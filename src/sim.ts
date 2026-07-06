// CPU 地形シミュレーション (DESIGN-m1m2.md 準拠)
// 岩盤/土/水 を格子で保持し、ブラシ変形・安息角スランプ・浅水パイプモデルを回す。
// 結果は RGBA16F の DataTexture [bedrock, soil, water, wetness] に毎フレーム詰めて GPU 描画へ渡す。
// 全量は CPU 配列の総和で直接検証できる (GPU 読み戻し不要)。
//
// 2モード:
//   procedural (既定) = 内蔵の谷地形を生成 (サンドボックス index.html)
//   geo             = 外部の実 DEM を bedrock に注入 (geo.html)。実地形は bedrock=不動、
//                     押した所だけ bedrock→soil(可動) に変わり自然に崩れる。

import * as THREE from 'three/webgpu';
import { fbm, ridged } from './noise';
import { digProfile, depositProfile, computeBrushNorm, type BrushNorm } from './brush-profile';

export const SIM_N = 192;   // 既定シム格子
export const WORLD = 600;   // 既定ワールド m
export const HEIGHT_M = 90;  // 既定最大標高 m

const G = 9.81;
const DAMP = 0.995;
const SLUMP_K = 0.5;
const SIM_DT = 1 / 120;
const TALUS_ANGLE = (34 * Math.PI) / 180;

export interface SimOpts {
  n?: number;
  world?: number;          // m
  bedrock?: Float32Array;  // n*n メートル。渡すと geo モード
  brushRadius?: number;    // m
  sourceQ?: number;        // 点源 m³/s
}

export interface SimTotals {
  solid: number;
  water: number;
  drained: number;
  injected: number;
  driftPct: number;
  nan: boolean;
}

function channelX(j: number): number {
  const f = j / (SIM_N - 1);
  return WORLD * 0.5 + Math.sin(f * Math.PI * 1.5) * 55 + Math.sin(f * 7.0) * 14;
}
function channelFloor(j: number): number {
  return 58 - (j / (SIM_N - 1)) * 50;
}

export class TerrainSim {
  readonly n: number;
  readonly world: number;
  readonly cell: number;
  private readonly area: number;
  private readonly talus: number;
  bedrock: Float32Array;
  soil: Float32Array;
  water: Float32Array;
  wet: Float32Array;
  private fL: Float32Array;
  private fR: Float32Array;
  private fU: Float32Array;
  private fD: Float32Array;
  private surf: Float32Array;
  private packed: Uint16Array;
  private terrainDirty = true;
  tex: THREE.DataTexture;

  initialSolid = 0;
  injected = 0;
  drained = 0;
  sourceOn: boolean;
  sourceQ: number;
  rainRate = 0;         // m/s 全面に降らす (geo で実谷に水を集める)
  brushRadius: number;
  digRate: number;      // m/s (中心)
  private sourceI: number;
  private sourceJ: number;
  private norm: BrushNorm;

  constructor(opts: SimOpts = {}) {
    const geo = !!opts.bedrock;
    const N = opts.n ?? SIM_N;
    const world = opts.world ?? WORLD;
    this.n = N;
    this.world = world;
    this.cell = world / N;
    this.area = this.cell * this.cell;
    this.talus = Math.tan(TALUS_ANGLE) * this.cell;
    this.brushRadius = opts.brushRadius ?? (geo ? world / 40 : 15);
    this.digRate = this.brushRadius * 0.4; // 半径連動 (15m→6m/s で従来一致)
    this.sourceQ = opts.sourceQ ?? 12;

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

    if (geo) {
      this.bedrock.set(opts.bedrock!.subarray(0, len));
      this.sourceOn = false;
      this.sourceI = Math.floor(N / 2);
      this.sourceJ = Math.floor(N / 2);
    } else {
      this.generateProcedural();
      this.sourceOn = true;
      this.sourceI = Math.round((channelX(10) / WORLD) * (N - 1));
      this.sourceJ = 10;
    }

    let s = 0;
    for (let k = 0; k < len; k++) s += this.bedrock[k];
    this.initialSolid = s * this.area;

    this.norm = computeBrushNorm(this.brushRadius, this.cell);
    this.tex = new THREE.DataTexture(this.packed, N, N, THREE.RGBAFormat, THREE.HalfFloatType);
    this.tex.minFilter = THREE.LinearFilter;
    this.tex.magFilter = THREE.LinearFilter;
    this.tex.wrapS = THREE.ClampToEdgeWrapping;
    this.tex.wrapT = THREE.ClampToEdgeWrapping;
    this.sync();
  }

  private generateProcedural() {
    const N = this.n;
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
        let h = 25 + (base * 0.45 + mount * 0.55) * 60 - zf * 16;
        const d = Math.abs(x - chanX);
        const mask = smooth(28, 6, d);
        const carved = Math.min(h, floor + (1 - mask) * 6);
        h = h * (1 - mask) + carved * mask;
        this.bedrock[j * N + i] = h;
      }
    }
    for (let j = 0; j < N; j++) {
      const chanX = channelX(j);
      for (let i = 0; i < N; i++) {
        const x = (i / N) * WORLD;
        const d = Math.abs(x - chanX);
        const mask = smooth(24, 4, d);
        if (mask > 0.25) this.water[j * N + i] = 0.6 * mask;
      }
    }
  }

  setBrushRadius(r: number) {
    this.brushRadius = r;
    this.digRate = r * 0.4;
    this.norm = computeBrushNorm(r, this.cell);
  }

  /** 水源をワールド uv 位置へ設定 (geo で任意地点に湧水) */
  setSourceUV(u: number, v: number) {
    this.sourceI = Math.round(Math.min(Math.max(u, 0), 1) * (this.n - 1));
    this.sourceJ = Math.round(Math.min(Math.max(v, 0), 1) * (this.n - 1));
  }

  surfaceHeightUV(u: number, v: number): number {
    const N = this.n;
    const fx = Math.min(Math.max(u, 0), 1) * (N - 1);
    const fy = Math.min(Math.max(v, 0), 1) * (N - 1);
    const i0 = Math.floor(fx);
    const j0 = Math.floor(fy);
    const i1 = Math.min(i0 + 1, N - 1);
    const j1 = Math.min(j0 + 1, N - 1);
    const tx = fx - i0;
    const ty = fy - j0;
    const B = this.bedrock, S = this.soil;
    const h = (i: number, j: number) => B[j * N + i] + S[j * N + i];
    const a = h(i0, j0) * (1 - tx) + h(i1, j0) * tx;
    const b = h(i0, j1) * (1 - tx) + h(i1, j1) * tx;
    return a * (1 - ty) + b * ty;
  }

  // ── ブラシ: 押した体積を再配分 (完全保存) ──
  brush(u: number, v: number, dt: number, mode: 'dig' | 'raise' = 'dig') {
    const N = this.n;
    const cell = this.cell;
    const ext = Math.ceil(this.norm.outerRadius / cell) + 1;
    const ci = Math.min(Math.max(u * (N - 1), ext), N - 1 - ext);
    const cj = Math.min(Math.max(v * (N - 1), ext), N - 1 - ext);
    const R = this.brushRadius;
    const amt = this.digRate * dt;
    const ri = Math.round(ci);
    const rj = Math.round(cj);

    const takeProf = mode === 'dig' ? digProfile : depositProfile;
    const dropProf = mode === 'dig' ? depositProfile : digProfile;
    const dropSum = mode === 'dig' ? this.norm.sumG : this.norm.sumW;

    let removed = 0;
    for (let dj = -ext; dj <= ext; dj++) {
      for (let di = -ext; di <= ext; di++) {
        const i = ri + di;
        const j = rj + dj;
        if (i < 0 || j < 0 || i >= N || j >= N) continue;
        const r = Math.hypot(ri + di - ci, rj + dj - cj) * cell;
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

    if (dropSum <= 0) {
      this.soil[rj * N + ri] += removed;
      return;
    }
    for (let dj = -ext; dj <= ext; dj++) {
      for (let di = -ext; di <= ext; di++) {
        const i = ri + di;
        const j = rj + dj;
        if (i < 0 || j < 0 || i >= N || j >= N) continue;
        const r = Math.hypot(ri + di - ci, rj + dj - cj) * cell;
        const g = dropProf(r, R);
        if (g <= 0) continue;
        this.soil[j * N + i] += (g / dropSum) * removed;
      }
    }
  }

  private slump() {
    const N = this.n;
    const TALUS = this.talus;
    const B = this.bedrock;
    const S = this.soil;
    const H = this.surf;
    for (let c = 0; c < H.length; c++) H[c] = B[c] + S[c];
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

  private waterStep(dt: number) {
    const N = this.n;
    const cell = this.cell;
    const AREA = this.area;
    const B = this.bedrock;
    const S = this.soil;
    const W = this.water;
    const fL = this.fL, fR = this.fR, fU = this.fU, fD = this.fD;
    const surf = this.surf;

    // 全面降雨 (geo: 実谷に水が集まる)
    if (this.rainRate > 0) {
      const add = this.rainRate * dt;
      for (let k = 0; k < W.length; k++) W[k] += add;
      this.injected += add * AREA * W.length;
    }

    // 点源
    if (this.sourceOn) {
      const add = this.sourceQ * dt;
      const si = this.sourceI, sj = this.sourceJ;
      let wsum = 0;
      const rad = Math.max(1, Math.ceil((cell < 6 ? 3 : cell * 1.5) / cell));
      const wgt: number[] = [];
      const cells: number[] = [];
      for (let dj = -rad; dj <= rad; dj++) {
        for (let di = -rad; di <= rad; di++) {
          const i = si + di, j = sj + dj;
          if (i < 0 || j < 0 || i >= N || j >= N) continue;
          const dd = Math.hypot(di, dj);
          const wv = Math.max(0, 1 - dd / (rad + 0.5));
          if (wv <= 0) continue;
          wgt.push(wv); cells.push(j * N + i); wsum += wv;
        }
      }
      if (wsum > 0) for (let k = 0; k < cells.length; k++) W[cells[k]] += (add * (wgt[k] / wsum)) / AREA;
      this.injected += add;
    }

    for (let c = 0; c < surf.length; c++) surf[c] = B[c] + S[c] + W[c];
    const co = dt * G * cell;

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

  step(frameDt: number, waterIters = 2) {
    this.slump();
    for (let k = 0; k < waterIters; k++) this.waterStep(SIM_DT);
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
    solid *= this.area;
    water *= this.area;
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
