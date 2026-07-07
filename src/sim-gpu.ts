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
  vec2, vec4, max, min, length, sin,
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
  private stateAttr: THREE.StorageInstancedBufferAttribute;

  // uniforms
  private uCell = uniform(1);
  private uDt = uniform(SIM_DT);
  private uTalus = uniform(1);
  private uRain = uniform(0);
  private uBrush = uniform(new THREE.Vector4(0, 0, 0, 0)); // u, v, radius(m), amt
  private uBrushMode = uniform(1); // +1 dig / -1 raise
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

    // 初期 state バッファ (bedrock を x に)
    const len = N * N;
    const initState = new Float32Array(len * 4);
    for (let k = 0; k < len; k++) initState[k * 4] = opts.bedrock[k];
    this.state = instancedArray(initState as unknown as number, 'vec4');
    this.stateAttr = (this.state as unknown as { value: THREE.StorageInstancedBufferAttribute }).value;
    this.flux = instancedArray(len, 'vec4');
    this.slump = instancedArray(len, 'vec4');

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
  }

  /** 1 フレーム進める */
  step(waterIters = 1, doSlump = true) {
    const r = this.renderer;
    if (doSlump) { r.compute(this.kSlump1); r.compute(this.kSlump2); }
    for (let k = 0; k < waterIters; k++) {
      if (this.rainRate > 0) { this.uRain.value = this.rainRate; r.compute(this.kRain); }
      r.compute(this.kFlux);
      r.compute(this.kDepth);
    }
  }

  /** ブラシ適用 (u,v=0..1) */
  brush(u: number, v: number, dt: number, mode: 'dig' | 'raise') {
    (this.uBrush.value as THREE.Vector4).set(u, v, this.brushRadius, this.digRate * dt);
    this.uBrushMode.value = mode === 'dig' ? 1 : -1;
    this.renderer.compute(this.kBrush);
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
