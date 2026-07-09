// @ts-nocheck — TSL ノード代数は動的型。TS 定義が過度に厳格なため本ファイルは型検査を外し、
//               正しさはブラウザ実機テスト(gpu-test.html)で検証する。
// terra-touch GPU シミュレーション (WebGPU compute / TSL storage buffer)。
// CPU版(sim.ts)と同じ浅水パイプモデル+安息角スランプ+体積保存ブラシを GPU で回す。
// storage buffer は同要素の read+write が可能(rgba32f storage texture は不可)なので採用。
// 状態を N² の vec4 バッファで保持し、格子=描画メッシュ=1024² を 1:1 対応させる。
//
// state[c] = vec4(bedrock, soil, water, wetness)   [m]
// flux[c]  = vec4(fL, fR, fU, fD)  非負4フラックス (Mei パイプモデル・momentum保持)
// slump[c] = vec4(sL, sR, sU, sD)  スランプ流出 (transient)

import * as THREE from 'three/webgpu';
import {
  Fn, instanceIndex, instancedArray, uniform, select, float, int, uint,
  vec2, vec4, uvec2, max, min, length, sin, textureStore,
} from 'three/tsl';
import { computeBrushNorm } from './brush-profile';

const G = 9.81;
const DAMP = 0.995;
const SLUMP_K = 0.5;
const SIM_DT = 1 / 120;
const TALUS_ANGLE = (34 * Math.PI) / 180;

export interface GpuSimOpts {
  n: number;
  world: number;
  bedrock: Float32Array; // n*n メートル
}

export class GpuSim {
  readonly n: number;
  readonly world: number;
  readonly cell: number;
  private renderer: THREE.WebGPURenderer;
  private initBed: Float32Array;

  // storage buffers (TSL StorageBufferNode)。TSLジェネリック型は追わず any 扱い。
  /* eslint-disable @typescript-eslint/no-explicit-any */
  state: any;
  private flux: any;
  private slump: any;
  private initStateBuf: any;
  private kReset: any;
  private stateAttr: THREE.StorageInstancedBufferAttribute;

  // uniforms
  private uCell = uniform(1);
  private uDt = uniform(SIM_DT);
  private uTalus = uniform(1);
  private uRain = uniform(0);
  private uBrush = uniform(new THREE.Vector4(0, 0, 0, 0)); // u, v, radius(m), amt
  private uBrushMode = uniform(1); // +1 dig / -1 raise
  private uSource = uniform(new THREE.Vector4(0, 0, 1, 0)); // u, v, radius(m), volume(m³ this dispatch)
  private uScaleDig = uniform(1);  // 掘り時: 縁堆積スケール ΣW/ΣG
  private uScaleRaise = uniform(1); // 盛り時: 中心堆積スケール ΣG/ΣW

  rainRate = 0;
  brushRadius: number;
  digRate: number;

  // 事前ビルドした compute ノード
  private kRain: any;
  private kFlux: any;
  private kDepth: any;
  private kSlump1: any;
  private kSlump2: any;
  private kBrush: any;
  private kSource: any;
  private kCopy: any;
  // 神の川ツール: 恒常水源(spring)。step 毎に注入。q=m³/s。
  springs: { u: number; v: number; q: number }[] = [];
  springRad = 1;
  // 表示用テクスチャ: [地表高(bed+soil), water, wet, 1]。毎ステップ末に buffer からコピー。
  dispTex!: THREE.StorageTexture;
  // 村ロジック用の粗い世界センサ (縮約 coarseN²・低頻度読戻し)
  readonly coarseN = 128;
  private coarseBuf: any;
  private coarseAttr!: THREE.StorageInstancedBufferAttribute;
  private kCoarse: any;
  coarse: Float32Array; // [height, water, wet, 0] × coarseN²

  constructor(renderer: THREE.WebGPURenderer, opts: GpuSimOpts) {
    this.renderer = renderer;
    const N = opts.n;
    this.n = N;
    this.world = opts.world;
    this.cell = opts.world / N;
    this.initBed = opts.bedrock;
    this.uCell.value = this.cell;
    this.uTalus.value = Math.tan(TALUS_ANGLE) * this.cell;
    this.brushRadius = opts.world / 40;
    this.digRate = this.brushRadius * 0.4;
    this.springRad = Math.max(2.5 * this.cell, opts.world / 160);

    // 初期 state バッファ (bedrock を x に)
    const len = N * N;
    const initState = new Float32Array(len * 4);
    for (let k = 0; k < len; k++) initState[k * 4] = opts.bedrock[k];
    this.state = instancedArray(initState as unknown as number, 'vec4');
    this.stateAttr = (this.state as unknown as { value: THREE.StorageInstancedBufferAttribute }).value;
    this.initStateBuf = instancedArray(initState.slice() as unknown as number, 'vec4'); // reset用の不変コピー
    this.flux = instancedArray(len, 'vec4');
    this.slump = instancedArray(len, 'vec4');
    this.coarseBuf = instancedArray(this.coarseN * this.coarseN, 'vec4');
    this.coarseAttr = (this.coarseBuf as unknown as { value: THREE.StorageInstancedBufferAttribute }).value;
    this.coarse = new Float32Array(this.coarseN * this.coarseN * 4);

    // 表示用 StorageTexture (rgba32f・textureLoad で整数座標読み=フィルタ不要でクッキリ)
    this.dispTex = new THREE.StorageTexture(N, N);
    this.dispTex.format = THREE.RGBAFormat;
    this.dispTex.type = THREE.FloatType;
    this.dispTex.minFilter = THREE.NearestFilter;
    this.dispTex.magFilter = THREE.NearestFilter;

    this.setBrushRadius(this.brushRadius);
    this.buildKernels(N);
  }

  setBrushRadius(r: number) {
    this.brushRadius = r;
    this.digRate = r * 0.4;
    const norm = computeBrushNorm(r, this.cell);
    this.uScaleDig.value = norm.sumG > 0 ? norm.sumW / norm.sumG : 0;
    this.uScaleRaise.value = norm.sumW > 0 ? norm.sumG / norm.sumW : 0;
  }

  private buildKernels(N: number) {
    const state = this.state, flux = this.flux, slump = this.slump;
    const uCell = this.uCell, uDt = this.uDt, uTalus = this.uTalus, uRain = this.uRain;
    const uBrush = this.uBrush, uBrushMode = this.uBrushMode;
    const uSource = this.uSource;
    const uScaleDig = this.uScaleDig, uScaleRaise = this.uScaleRaise;
    const Ni = int(N);
    const Nf = float(N);
    const co = float(G).mul(uDt).mul(uCell);
    const area = uCell.mul(uCell);

    // idx→(i,j) と 近傍 index (clamp)
    const ijOf = () => {
      const i = int(instanceIndex.mod(uint(N)));
      const j = int(instanceIndex.div(uint(N)));
      return { i, j };
    };
    const hi = Ni.sub(int(1));
    const nid = (i: ReturnType<typeof int>, j: ReturnType<typeof int>, di: number, dj: number) => {
      const ni = min(max(i.add(int(di)), int(0)), hi);
      const nj = min(max(j.add(int(dj)), int(0)), hi);
      return uint(nj.mul(Ni).add(ni));
    };
    const surfAt = (i: ReturnType<typeof int>, j: ReturnType<typeof int>, di: number, dj: number) => {
      const s = state.element(nid(i, j, di, dj));
      return s.x.add(s.y).add(s.z);
    };

    // ── 降雨 (全面に水を足す) ──
    this.kRain = Fn(() => {
      const s = state.element(instanceIndex);
      const w = s.z.add(uRain.mul(uDt));
      state.element(instanceIndex).assign(vec4(s.x, s.y, w, s.w));
    })().compute(N * N);

    // ── 水フラックス更新 (旧 surf から・自セルのみ書込) ──
    this.kFlux = Fn(() => {
      const { i, j } = ijOf();
      const s = state.element(instanceIndex);
      const wc = s.x.add(s.y).add(s.z);
      const f = flux.element(instanceIndex);
      // 境界は off-map surf=0 で排水
      const sL = select(i.equal(int(0)), float(0), surfAt(i, j, -1, 0));
      const sR = select(i.equal(Ni.sub(int(1))), float(0), surfAt(i, j, 1, 0));
      const sU = select(j.equal(int(0)), float(0), surfAt(i, j, 0, -1));
      const sD = select(j.equal(Ni.sub(int(1))), float(0), surfAt(i, j, 0, 1));
      const nl = max(float(0), f.x.mul(DAMP).add(co.mul(wc.sub(sL))));
      const nr = max(float(0), f.y.mul(DAMP).add(co.mul(wc.sub(sR))));
      const nu = max(float(0), f.z.mul(DAMP).add(co.mul(wc.sub(sU))));
      const nd = max(float(0), f.w.mul(DAMP).add(co.mul(wc.sub(sD))));
      const out = nl.add(nr).add(nu).add(nd).mul(uDt);
      const avail = s.z.mul(area);
      const K = min(float(1), avail.div(max(out, float(1e-6))));
      flux.element(instanceIndex).assign(vec4(nl.mul(K), nr.mul(K), nu.mul(K), nd.mul(K)));
    })().compute(N * N);

    // ── 水深更新 (flux から・自セル water を書込) ──
    this.kDepth = Fn(() => {
      const { i, j } = ijOf();
      const f = flux.element(instanceIndex);
      const outSum = f.x.add(f.y).add(f.z).add(f.w);
      const inL = select(i.equal(int(0)), float(0), flux.element(nid(i, j, -1, 0)).y);
      const inR = select(i.equal(Ni.sub(int(1))), float(0), flux.element(nid(i, j, 1, 0)).x);
      const inU = select(j.equal(int(0)), float(0), flux.element(nid(i, j, 0, -1)).w);
      const inD = select(j.equal(Ni.sub(int(1))), float(0), flux.element(nid(i, j, 0, 1)).z);
      const inSum = inL.add(inR).add(inU).add(inD);
      const s = state.element(instanceIndex);
      const w = max(float(0), s.z.add(uDt.mul(inSum.sub(outSum)).div(area)));
      const target = min(float(1), w.mul(5));
      const wet = select(target.greaterThan(s.w), target, s.w.mul(0.99));
      state.element(instanceIndex).assign(vec4(s.x, s.y, w, wet));
    })().compute(N * N);

    // ── スランプ流出 (土のみ) ──
    this.kSlump1 = Fn(() => {
      const { i, j } = ijOf();
      const s = state.element(instanceIndex);
      const hc = s.x.add(s.y);
      const kk = float(SLUMP_K * 0.5);
      const outDir = (di: number, dj: number) => {
        const sn = state.element(nid(i, j, di, dj));
        const diff = hc.sub(sn.x.add(sn.y)).sub(uTalus);
        return max(float(0), diff.mul(kk));
      };
      const oL = outDir(-1, 0), oR = outDir(1, 0), oU = outDir(0, -1), oD = outDir(0, 1);
      const tot = oL.add(oR).add(oU).add(oD);
      const K = min(float(1), s.y.div(max(tot, float(1e-6))));
      slump.element(instanceIndex).assign(vec4(oL.mul(K), oR.mul(K), oU.mul(K), oD.mul(K)));
    })().compute(N * N);

    // ── スランプ適用 (土移動) ──
    this.kSlump2 = Fn(() => {
      const { i, j } = ijOf();
      const sl = slump.element(instanceIndex);
      const outSum = sl.x.add(sl.y).add(sl.z).add(sl.w);
      const inL = select(i.equal(int(0)), float(0), slump.element(nid(i, j, -1, 0)).y);
      const inR = select(i.equal(Ni.sub(int(1))), float(0), slump.element(nid(i, j, 1, 0)).x);
      const inU = select(j.equal(int(0)), float(0), slump.element(nid(i, j, 0, -1)).w);
      const inD = select(j.equal(Ni.sub(int(1))), float(0), slump.element(nid(i, j, 0, 1)).z);
      const s = state.element(instanceIndex);
      const soil = max(float(0), s.y.add(inL.add(inR).add(inU).add(inD)).sub(outSum));
      state.element(instanceIndex).assign(vec4(s.x, soil, s.z, s.w));
    })().compute(N * N);

    // ── ブラシ (体積保存・解析的正規化) ──
    this.kBrush = Fn(() => {
      const { i, j } = ijOf();
      const R = uBrush.z;
      const amt = uBrush.w;
      const bi = uBrush.x.mul(Nf.sub(float(1)));
      const bj = uBrush.y.mul(Nf.sub(float(1)));
      const r = length(vec2(float(i).sub(bi), float(j).sub(bj))).mul(uCell);
      // dig profile w = (1-(r/R)^2)^2  (r<R)
      const rn = r.div(R);
      const t = float(1).sub(rn.mul(rn));
      const w = select(r.lessThan(R), max(float(0), t).mul(max(float(0), t)), float(0));
      // deposit profile g = sin^2(pi(r-R)/(0.6R))  (R<r<1.6R)
      const gg = sin(float(Math.PI).mul(r.sub(R)).div(R.mul(0.6)));
      const inRing = select(r.greaterThan(R), select(r.lessThan(R.mul(1.6)), float(1), float(0)), float(0));
      const g = gg.mul(gg).mul(inRing);
      // dig: take w, deposit g*scaleDig ; raise: take g, deposit w*scaleRaise
      const isDig = uBrushMode.greaterThan(float(0));
      const take = select(isDig, w, g).mul(amt);
      const drop = select(isDig, g.mul(uScaleDig), w.mul(uScaleRaise)).mul(amt);
      const s = state.element(instanceIndex);
      const takeSoil = min(s.y, take);
      const soilAfterTake = s.y.sub(takeSoil);
      const takeRock = min(s.x, take.sub(takeSoil));
      const bed = s.x.sub(takeRock);
      const soil = soilAfterTake.add(drop);
      state.element(instanceIndex).assign(vec4(bed, soil, s.z, s.w));
    })().compute(N * N);

    // ── 点源 (神の川ツール・恒常水源) ──
    // 円錐 (1-(r/R)²) を disk 上に注入。∫cone dA=πR²/2 で正規化し体積 Q(m³)を water(m) に。
    this.kSource = Fn(() => {
      const { i, j } = ijOf();
      const bi = uSource.x.mul(Nf.sub(float(1)));
      const bj = uSource.y.mul(Nf.sub(float(1)));
      const R = uSource.z, Q = uSource.w; // Q = 水深レート×dt (m)
      const r = length(vec2(float(i).sub(bi), float(j).sub(bj))).mul(uCell);
      const rn = r.div(R);
      const t = float(1).sub(rn.mul(rn));
      const wgt = select(r.lessThan(R), max(float(0), t), float(0)); // 円錐 0..1
      const s = state.element(instanceIndex);
      state.element(instanceIndex).assign(vec4(s.x, s.y, s.z.add(Q.mul(wgt)), s.w));
    })().compute(N * N);

    // ── 表示テクスチャへコピー ([bed+soil, water, wet, 1]) ──
    const dispTex = this.dispTex;
    this.kCopy = Fn(() => {
      const { i, j } = ijOf();
      const s = state.element(instanceIndex);
      textureStore(dispTex, uvec2(uint(i), uint(j)), vec4(s.x.add(s.y), s.z, s.w, float(1))).toWriteOnly();
    })().compute(N * N);

    // ── リセット (初期状態へ) ──
    const initBuf = this.initStateBuf;
    this.kReset = Fn(() => {
      state.element(instanceIndex).assign(initBuf.element(instanceIndex));
      flux.element(instanceIndex).assign(vec4(0, 0, 0, 0));
    })().compute(N * N);

    // ── 粗い世界センサへ縮約 (coarse cell → 対応する fine cell を代表サンプル) ──
    const coarseBuf = this.coarseBuf;
    const CN = int(this.coarseN);
    const scale = int(Math.floor(N / this.coarseN));
    // 縮約: 高さ=中心点(pick/カメラの回帰維持)。water/wet=ブロック内9点の MAX
    // (細い川がブロックのどこかにあれば村センサに映る=SPEC 8×8 max プーリング)。
    this.kCoarse = Fn(() => {
      const ci = int(instanceIndex.mod(uint(this.coarseN)));
      const cj = int(instanceIndex.div(uint(this.coarseN)));
      const bi = ci.mul(scale), bj = cj.mul(scale);
      const o0 = scale.div(int(6)), o1 = scale.div(int(2)), o2 = scale.mul(int(5)).div(int(6));
      const at = (ox: any, oy: any) => state.element(uint(min(bj.add(oy), Ni.sub(int(1))).mul(Ni).add(min(bi.add(ox), Ni.sub(int(1))))));
      const c = at(o1, o1); // 中心 = 高さ用
      const ss = [at(o0, o0), at(o1, o0), at(o2, o0), at(o0, o1), c, at(o2, o1), at(o0, o2), at(o1, o2), at(o2, o2)];
      let wMax = ss[0].z, wetMax = ss[0].w;
      for (let k = 1; k < 9; k++) { wMax = max(wMax, ss[k].z); wetMax = max(wetMax, ss[k].w); }
      coarseBuf.element(instanceIndex).assign(vec4(c.x.add(c.y), wMax, wetMax, float(0)));
    })().compute(this.coarseN * this.coarseN);
    void CN;
  }

  reset() {
    this.rainRate = 0;
    this.renderer.compute(this.kReset);
    this.renderer.compute(this.kCopy);
  }

  /** 1 フレーム進める */
  step(waterIters = 1, doSlump = true) {
    const r = this.renderer;
    if (doSlump) { r.compute(this.kSlump1); r.compute(this.kSlump2); }
    for (let k = 0; k < waterIters; k++) {
      if (this.rainRate > 0) { this.uRain.value = this.rainRate; r.compute(this.kRain); }
      for (const sp of this.springs) {
        (this.uSource.value as THREE.Vector4).set(sp.u, sp.v, this.springRad, sp.q * SIM_DT);
        r.compute(this.kSource);
      }
      r.compute(this.kFlux);
      r.compute(this.kDepth);
    }
    r.compute(this.kCopy); // 表示テクスチャ更新
  }

  /** ブラシ適用 (u,v=0..1) */
  brush(u: number, v: number, dt: number, mode: 'dig' | 'raise') {
    (this.uBrush.value as THREE.Vector4).set(u, v, this.brushRadius, this.digRate * dt);
    this.uBrushMode.value = mode === 'dig' ? 1 : -1;
    this.renderer.compute(this.kBrush);
    this.renderer.compute(this.kCopy);
  }

  /** dispTex を最新化 (初回描画前など) */
  syncDisp() { this.renderer.compute(this.kCopy); }

  /** 粗い世界センサを GPU→CPU へ読み戻す (低頻度で呼ぶ・~256KB) */
  async readCoarse(): Promise<Float32Array> {
    this.renderer.compute(this.kCoarse);
    const buf = await this.renderer.getArrayBufferAsync(this.coarseAttr as unknown as THREE.BufferAttribute);
    this.coarse.set(new Float32Array(buf));
    return this.coarse;
  }

  /** coarse を u,v(0..1) でバイリニアサンプル。ch: 0=height 1=water 2=wet */
  sampleCoarse(u: number, v: number, ch: number): number {
    const N = this.coarseN;
    const fx = Math.min(Math.max(u, 0), 1) * (N - 1);
    const fy = Math.min(Math.max(v, 0), 1) * (N - 1);
    const i0 = Math.floor(fx), j0 = Math.floor(fy);
    const i1 = Math.min(i0 + 1, N - 1), j1 = Math.min(j0 + 1, N - 1);
    const tx = fx - i0, ty = fy - j0;
    const c = this.coarse;
    const g = (i: number, j: number) => c[(j * N + i) * 4 + ch];
    const a = g(i0, j0) * (1 - tx) + g(i1, j0) * tx;
    const b = g(i0, j1) * (1 - tx) + g(i1, j1) * tx;
    return a * (1 - ty) + b * ty;
  }

  /** CPU へ state を読み戻して総量/NaN を確認 (検証用・重い) */
  async totals(): Promise<{ solid: number; water: number; nan: boolean }> {
    const buf = await this.renderer.getArrayBufferAsync(this.stateAttr as unknown as THREE.BufferAttribute);
    const a = new Float32Array(buf);
    const area = this.cell * this.cell;
    let solid = 0, water = 0, nan = false;
    for (let k = 0; k < this.n * this.n; k++) {
      const b = a[k * 4], s = a[k * 4 + 1], w = a[k * 4 + 2];
      solid += b + s; water += w;
      if (!nan && (Number.isNaN(b) || Number.isNaN(s) || Number.isNaN(w))) nan = true;
    }
    return { solid: solid * area, water: water * area, nan };
  }
}
