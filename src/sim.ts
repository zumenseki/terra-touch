// CPU 地形シミュレーション (DESIGN-m1m2.md 準拠)
// 256² 格子で 岩盤/土/水 を保持し、ブラシ変形・安息角スランプ・浅水パイプモデルを回す。
// 結果は RGBA16F の DataTexture [bedrock, soil, water, wetness] に毎フレーム詰めて GPU 描画へ渡す。
// 全量は CPU 配列の総和で直接検証できる (GPU 読み戻し不要)。

import * as THREE from 'three/webgpu';
import { fbm, ridged } from './noise';
import { digProfile, depositProfile, computeBrushNorm, type BrushNorm } from './brush-profile';

export const SIM_N = 192;   // CPU シム格子 (60fps 目標での実測最適・M4 で worker+256² 化予定)
export const WORLD = 600;   // m
export const HEIGHT_M = 90;  // 最大岩盤標高 m

const CELL = WORLD / SIM_N;             // 2.34 m
const AREA = CELL * CELL;               // セル面積 m²
const G = 9.81;
const DAMP = 0.995;                     // 水フラックス減衰
const TALUS = Math.tan((34 * Math.PI) / 180) * CELL; // 安息角の段差閾値 (m/セル)
const SLUMP_K = 0.5;                    // スランプ緩和係数
const SIM_DT = 1 / 120;                 // 物理ステップ
const SOURCE_Q = 12;                    // 湧き水 m³/s

export interface SimTotals {
  solid: number;   // Σ(bedrock+soil)·AREA
  water: number;   // Σwater·AREA
  drained: number; // 端から出た総量
  injected: number;// 湧き総量
  driftPct: number;// 土の初期比 drift%
  nan: boolean;
}

// 谷の中心線 (world x)。j: 上端(0)→下端(N-1)。緩い蛇行・場内に収まる
function channelX(j: number): number {
  const f = j / (SIM_N - 1);
  return WORLD * 0.5 + Math.sin(f * Math.PI * 1.5) * 55 + Math.sin(f * 7.0) * 14;
}
// 谷底の標高 (m)。上端 58m → 下端 8m へ厳密に単調降下 (勾配 ~8%)
function channelFloor(j: number): number {
  return 58 - (j / (SIM_N - 1)) * 50;
}

export class TerrainSim {
  readonly n = SIM_N;
  readonly cell = CELL;
  bedrock: Float32Array;
  soil: Float32Array;
  water: Float32Array;
  wet: Float32Array;
  // 非負4フラックス (Mei パイプモデル)。fR[c]=c→右(i+1) 等
  private fL: Float32Array;
  private fR: Float32Array;
  private fU: Float32Array;
  private fD: Float32Array;
  private surf: Float32Array; // 再利用スクラッチ (bedrock+soil[+water])
  private packed: Uint16Array;
  private terrainDirty = true; // bedrock/soil を再パックする必要
  tex: THREE.DataTexture;

  initialSolid = 0;
  injected = 0;
  drained = 0;
  sourceOn = true;
  brushRadius = 15;
  digRate = 6; // m/s (中心)

  private sourceI: number;
  private sourceJ = 10;
  private norm: BrushNorm;

  constructor() {
    const N = SIM_N;
    const len = N * N;
    this.bedrock = new Float32Array(len);
    this.soil = new Float32Array(len);
    this.water = new Float32Array(len);
    this.wet = new Float32Array(len);
    this.fL = new Float32Array(len);
    this.fR = new Float32Array(len);
    this.fU = new Float32Array(len);
    this.fD = new Float32Array(len);
    this.surf = new Float32Array(len);
    this.packed = new Uint16Array(len * 4);

    // 初期地形: 山肌 + 上端→下端へ単調降下する谷を刻む (メートル)
    const smooth = (e0: number, e1: number, x: number) => {
      const t = Math.min(Math.max((x - e0) / (e1 - e0), 0), 1);
      return t * t * (3 - 2 * t);
    };
    for (let j = 0; j < N; j++) {
      const z = (j / N) * WORLD;
      const zf = j / (N - 1);
      const chanX = channelX(j);
      const floor = channelFloor(j);
      for (let i = 0; i < N; i++) {
        const x = (i / N) * WORLD;
        const base = fbm(x * 0.004, z * 0.004);
        const mount = ridged(x * 0.008, z * 0.008);
        // 山肌 25..85m + 全体を下流へ緩く傾ける
        let h = 25 + (base * 0.45 + mount * 0.55) * 60 - zf * 16;
        // 谷を刻む: 中心±6m で floor まで下げ、28m で山肌へブレンド
        const d = Math.abs(x - chanX);
        const mask = smooth(28, 6, d); // 1=中心, 0=谷外
        const carved = Math.min(h, floor + (1 - mask) * 6);
        h = h * (1 - mask) + carved * mask;
        this.bedrock[j * N + i] = h;
        this.wet[j * N + i] = 0;
      }
    }

    let s = 0;
    for (let k = 0; k < len; k++) s += this.bedrock[k];
    this.initialSolid = s * AREA;

    this.sourceI = Math.round((channelX(this.sourceJ) / WORLD) * (N - 1));

    // 川を初期シード: 谷に浅い水を張って開始直後から流れを見せる
    for (let j = 0; j < N; j++) {
      const chanX = channelX(j);
      for (let i = 0; i < N; i++) {
        const x = (i / N) * WORLD;
        const d = Math.abs(x - chanX);
        const mask = smooth(24, 4, d);
        if (mask > 0.25) this.water[j * N + i] = 0.6 * mask;
      }
    }
    this.norm = computeBrushNorm(this.brushRadius, CELL);

    this.tex = new THREE.DataTexture(this.packed, N, N, THREE.RGBAFormat, THREE.HalfFloatType);
    this.tex.minFilter = THREE.LinearFilter;
    this.tex.magFilter = THREE.LinearFilter;
    this.tex.wrapS = THREE.ClampToEdgeWrapping;
    this.tex.wrapT = THREE.ClampToEdgeWrapping;
    this.sync();
  }

  setBrushRadius(r: number) {
    this.brushRadius = r;
    this.norm = computeBrushNorm(r, CELL);
  }

  // ── バイリニア標高サンプル (レイキャスト用・メートル) ──
  surfaceHeightUV(u: number, v: number): number {
    const N = SIM_N;
    const fx = Math.min(Math.max(u, 0), 1) * (N - 1);
    const fy = Math.min(Math.max(v, 0), 1) * (N - 1);
    const i0 = Math.floor(fx);
    const j0 = Math.floor(fy);
    const i1 = Math.min(i0 + 1, N - 1);
    const j1 = Math.min(j0 + 1, N - 1);
    const tx = fx - i0;
    const ty = fy - j0;
    const h = (i: number, j: number) => this.bedrock[j * N + i] + this.soil[j * N + i];
    const a = h(i0, j0) * (1 - tx) + h(i1, j0) * tx;
    const b = h(i0, j1) * (1 - tx) + h(i1, j1) * tx;
    return a * (1 - ty) + b * ty;
  }

  // ── ブラシ: 押した体積を再配分 (完全保存) ──
  //   'dig'   = 中心を掘り (w) 縁へ盛る (g)  … 指を押し込む
  //   'raise' = 縁から借り (g) 中心へ盛る (w)  … 土手を押し上げる
  brush(u: number, v: number, dt: number, mode: 'dig' | 'raise' = 'dig') {
    const N = SIM_N;
    const ext = Math.ceil(this.norm.outerRadius / CELL) + 1;
    const ci = Math.min(Math.max(u * (N - 1), ext), N - 1 - ext);
    const cj = Math.min(Math.max(v * (N - 1), ext), N - 1 - ext);
    const R = this.brushRadius;
    const amt = this.digRate * dt;
    const ri = Math.round(ci);
    const rj = Math.round(cj);

    // 取る/盛るプロファイルをモードで入れ替え
    const takeProf = mode === 'dig' ? digProfile : depositProfile;
    const dropProf = mode === 'dig' ? depositProfile : digProfile;
    const dropSum = mode === 'dig' ? this.norm.sumG : this.norm.sumW;

    // pass1: takeProf に沿って掘削 (soil 優先→bedrock)、実際に取れた量を集計
    let removed = 0;
    for (let dj = -ext; dj <= ext; dj++) {
      for (let di = -ext; di <= ext; di++) {
        const i = ri + di;
        const j = rj + dj;
        if (i < 0 || j < 0 || i >= N || j >= N) continue;
        const r = Math.hypot(ri + di - ci, rj + dj - cj) * CELL;
        const w = takeProf(r, R);
        if (w <= 0) continue;
        let need = w * amt;
        const idx = j * N + i;
        const takeSoil = Math.min(this.soil[idx], need);
        this.soil[idx] -= takeSoil;
        need -= takeSoil;
        const takeRock = Math.min(this.bedrock[idx], need);
        this.bedrock[idx] -= takeRock;
        removed += takeSoil + takeRock;
      }
    }
    if (removed <= 0) return;
    this.terrainDirty = true;

    // pass2: dropProf/Σ で配分 → Σdrop = removed (厳密保存)
    if (dropSum <= 0) {
      this.soil[rj * N + ri] += removed;
      return;
    }
    for (let dj = -ext; dj <= ext; dj++) {
      for (let di = -ext; di <= ext; di++) {
        const i = ri + di;
        const j = rj + dj;
        if (i < 0 || j < 0 || i >= N || j >= N) continue;
        const r = Math.hypot(ri + di - ci, rj + dj - cj) * CELL;
        const g = dropProf(r, R);
        if (g <= 0) continue;
        this.soil[j * N + i] += (g / dropSum) * removed;
      }
    }
  }

  // ── 安息角スランプ (土のみ移動・各辺1回処理で厳密保存) ──
  private slump() {
    const N = SIM_N;
    const B = this.bedrock;
    const S = this.soil;
    const H = this.surf;
    for (let c = 0; c < H.length; c++) H[c] = B[c] + S[c]; // 高さ事前計算
    const kk = SLUMP_K * 0.5;
    let moved = false;
    for (let j = 0; j < N; j++) {
      const row = j * N;
      for (let i = 0; i < N; i++) {
        const c = row + i;
        const hc = H[c];
        if (i + 1 < N) {
          const r = c + 1;
          const diff = hc - H[r];
          if (diff > TALUS) {
            const move = kk * (diff - TALUS);
            const m = move < S[c] ? move : S[c];
            if (m > 0) { S[c] -= m; S[r] += m; H[c] -= m; H[r] += m; moved = true; }
          } else if (-diff > TALUS) {
            const move = kk * (-diff - TALUS);
            const m = move < S[r] ? move : S[r];
            if (m > 0) { S[r] -= m; S[c] += m; H[r] -= m; H[c] += m; moved = true; }
          }
        }
        if (j + 1 < N) {
          const d = c + N;
          const diff = H[c] - H[d];
          if (diff > TALUS) {
            const move = kk * (diff - TALUS);
            const m = move < S[c] ? move : S[c];
            if (m > 0) { S[c] -= m; S[d] += m; H[c] -= m; H[d] += m; moved = true; }
          } else if (-diff > TALUS) {
            const move = kk * (-diff - TALUS);
            const m = move < S[d] ? move : S[d];
            if (m > 0) { S[d] -= m; S[c] += m; H[d] -= m; H[c] += m; moved = true; }
          }
        }
      }
    }
    if (moved) this.terrainDirty = true;
  }

  // ── 浅水 1 サブステップ (パイプモデル・surface 配列を事前計算) ──
  private waterStep(dt: number) {
    const N = SIM_N;
    const B = this.bedrock;
    const S = this.soil;
    const W = this.water;
    const fL = this.fL, fR = this.fR, fU = this.fU, fD = this.fD;
    const surf = this.surf;

    // 湧き水注入
    if (this.sourceOn) {
      const add = SOURCE_Q * dt;
      const si = this.sourceI, sj = this.sourceJ;
      let wsum = 0;
      const rad = Math.ceil(3 / CELL);
      const wgt: number[] = [];
      const cells: number[] = [];
      for (let dj = -rad; dj <= rad; dj++) {
        for (let di = -rad; di <= rad; di++) {
          const i = si + di, j = sj + dj;
          if (i < 0 || j < 0 || i >= N || j >= N) continue;
          const d = Math.hypot(di, dj) * CELL;
          const wv = Math.max(0, 1 - d / 3);
          if (wv <= 0) continue;
          wgt.push(wv); cells.push(j * N + i); wsum += wv;
        }
      }
      if (wsum > 0) for (let k = 0; k < cells.length; k++) W[cells[k]] += (add * (wgt[k] / wsum)) / AREA;
      this.injected += add;
    }

    // 表面高 (bedrock+soil+water) を事前計算
    for (let c = 0; c < surf.length; c++) surf[c] = B[c] + S[c] + W[c];
    const co = dt * G * CELL;

    // flux 更新 (旧 surf から)
    for (let j = 0; j < N; j++) {
      const row = j * N;
      for (let i = 0; i < N; i++) {
        const c = row + i;
        const wc = surf[c];
        let l = fL[c] * DAMP + co * (wc - (i > 0 ? surf[c - 1] : 0));
        let r = fR[c] * DAMP + co * (wc - (i < N - 1 ? surf[c + 1] : 0));
        let u = fU[c] * DAMP + co * (wc - (j > 0 ? surf[c - N] : 0));
        let d = fD[c] * DAMP + co * (wc - (j < N - 1 ? surf[c + N] : 0));
        if (l < 0) l = 0;
        if (r < 0) r = 0;
        if (u < 0) u = 0;
        if (d < 0) d = 0;
        const out = (l + r + u + d) * dt;
        const avail = W[c] * AREA;
        if (out > avail && out > 0) {
          const K = avail / out;
          l *= K; r *= K; u *= K; d *= K;
        }
        fL[c] = l; fR[c] = r; fU[c] = u; fD[c] = d;
      }
    }

    // depth 更新 (flux から) + 端排水
    let drained = 0;
    for (let j = 0; j < N; j++) {
      const row = j * N;
      const atTop = j === 0, atBot = j === N - 1;
      for (let i = 0; i < N; i++) {
        const c = row + i;
        const outSum = fL[c] + fR[c] + fU[c] + fD[c];
        let inSum = 0;
        if (i > 0) inSum += fR[c - 1];
        if (i < N - 1) inSum += fL[c + 1];
        if (j > 0) inSum += fD[c - N];
        if (j < N - 1) inSum += fU[c + N];
        if (i === 0) drained += fL[c];
        if (i === N - 1) drained += fR[c];
        if (atTop) drained += fU[c];
        if (atBot) drained += fD[c];
        let w = W[c] + (dt * (inSum - outSum)) / AREA;
        if (w < 0) w = 0;
        W[c] = w;
      }
    }
    this.drained += drained * dt;
  }

  /** 1 フレーム分進める (brush は呼び出し側で別途 brush() 済み前提) */
  step(frameDt: number, waterIters = 2) {
    this.slump();
    const dt = SIM_DT;
    for (let k = 0; k < waterIters; k++) this.waterStep(dt);
    void frameDt;
    this.updateWetness();
  }

  private updateWetness() {
    const W = this.water;
    const wet = this.wet;
    for (let k = 0; k < wet.length; k++) {
      const target = Math.min(1, W[k] * 5);
      wet[k] = target > wet[k] ? target : wet[k] * 0.985 + target * 0.015;
    }
  }

  /** CPU 配列を RGBA16F テクスチャへ詰めて GPU へ反映
   *  水/濡れ (b,a) は毎回、岩盤/土 (r,g) は変形時のみ再パック (半float変換を半減) */
  sync() {
    const W = this.water, wet = this.wet;
    const p = this.packed;
    const toH = THREE.DataUtils.toHalfFloat;
    if (this.terrainDirty) {
      const B = this.bedrock, S = this.soil;
      for (let k = 0, o = 0; k < B.length; k++, o += 4) {
        p[o] = toH(B[k]);
        p[o + 1] = toH(S[k]);
        p[o + 2] = toH(W[k]);
        p[o + 3] = toH(wet[k]);
      }
      this.terrainDirty = false;
    } else {
      for (let k = 0, o = 2; k < W.length; k++, o += 4) {
        p[o] = toH(W[k]);
        p[o + 1] = toH(wet[k]);
      }
    }
    this.tex.needsUpdate = true;
  }

  totals(): SimTotals {
    const B = this.bedrock, S = this.soil, W = this.water;
    let solid = 0, water = 0;
    let nan = false;
    for (let k = 0; k < B.length; k++) {
      solid += B[k] + S[k];
      water += W[k];
      if (!nan && (Number.isNaN(B[k]) || Number.isNaN(S[k]) || Number.isNaN(W[k]))) nan = true;
    }
    solid *= AREA;
    water *= AREA;
    return {
      solid,
      water,
      drained: this.drained,
      injected: this.injected,
      driftPct: this.initialSolid > 0 ? ((solid - this.initialSolid) / this.initialSolid) * 100 : 0,
      nan,
    };
  }
}
