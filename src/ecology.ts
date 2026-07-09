// terra-touch: 創世 — 生態システム (N6 魚+陸獣)。密度フィールド駆動(個体AIなし・決定論)。
// 容量 K は水深/平地/緑度から。logistic 成長+seed+線形decay+拡散。村が Holling-II で収穫し
// 魚/獣が多い水辺の適地度を上げて人を呼ぶ(誘引)。描画=密度から InstancedMesh を eco-tick 時に配置。
// 詳細裏設定: docs/SPEC-life-sim.md §7。

import * as THREE from 'three/webgpu';
import type { WorldSensor } from './village';

const ECO_N = 32;
const R_FISH = 1.4, SEED_FISH = 0.2, DECAY_FISH = 10;
const R_LAND = 0.5, SEED_LAND = 0.15, DECAY_LAND = 6;
const D_FISH = 1.0, D_LAND = 0.6;      // 拡散 /年
const K_FISH_MAX = 24, K_LAND_MAX = 16;
const CMAX_FISH = 6, CMAX_LAND = 3, NHALF_FISH = 6, NHALF_LAND = 4, MAX_TAKE = 0.4;
const FISH_WATER_MIN = 0.15;

const clamp = (x: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, x));
const smooth = (a: number, b: number, x: number) => { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };

export class EcologySystem {
  group = new THREE.Group();
  fish = new Float32Array(ECO_N * ECO_N);
  land = new Float32Array(ECO_N * ECO_N);
  kFish = new Float32Array(ECO_N * ECO_N);
  kLand = new Float32Array(ECO_N * ECO_N);
  private tmp = new Float32Array(ECO_N * ECO_N);
  private sensor: WorldSensor;
  private refreshed = false;
  // 収穫会計(検証用): 除去した魚/獣 == 供給した食料
  harvestedFish = 0;
  harvestedLand = 0;
  // 描画
  private fishMesh: THREE.InstancedMesh;
  private animalMesh: THREE.InstancedMesh;
  private fishCap: number;
  private animalCap: number;
  private dummy = new THREE.Object3D();

  constructor(sensor: WorldSensor, opts?: { fishCap?: number; animalCap?: number }) {
    this.sensor = sensor;
    this.fishCap = opts?.fishCap ?? 120;
    this.animalCap = opts?.animalCap ?? 60;
    // 魚 = 扁平ダイヤ(菱形)。獣 = 低い箱胴+脚(鹿/猪の抽象)。
    const fishGeo = new THREE.OctahedronGeometry(1, 0); fishGeo.scale(1.6, 0.5, 0.8);
    const fishMat = new THREE.MeshStandardNodeMaterial({ color: 0x8899aa, roughness: 0.5, metalness: 0.1 });
    this.fishMesh = new THREE.InstancedMesh(fishGeo, fishMat, this.fishCap);
    this.fishMesh.frustumCulled = false; this.fishMesh.count = this.fishCap;
    const animalGeo = new THREE.BoxGeometry(1.6, 0.9, 0.7); animalGeo.translate(0, 0.6, 0);
    const animalMat = new THREE.MeshStandardNodeMaterial({ color: 0x7a5a3a, roughness: 0.9 });
    this.animalMesh = new THREE.InstancedMesh(animalGeo, animalMat, this.animalCap);
    this.animalMesh.frustumCulled = false; this.animalMesh.count = this.animalCap;
    const z = new THREE.Matrix4().makeScale(0, 0, 0);
    for (let i = 0; i < this.fishCap; i++) this.fishMesh.setMatrixAt(i, z);
    for (let i = 0; i < this.animalCap; i++) this.animalMesh.setMatrixAt(i, z);
    this.group.add(this.fishMesh); this.group.add(this.animalMesh);
  }

  clear() {
    this.fish.fill(0); this.land.fill(0); this.refreshed = false;
    this.harvestedFish = 0; this.harvestedLand = 0;
  }

  private cellClamp(x: number) { return Math.min(ECO_N - 1, Math.max(0, Math.round(x * (ECO_N - 1)))); }

  private flat(u: number, v: number): number {
    const W = this.sensor.worldW, d = 3 / W;
    const h = this.sensor.heightUV.bind(this.sensor);
    const gx = (h(u + d, v) - h(u - d, v)) / (2 * d * W);
    const gz = (h(u, v + d) - h(u, v - d)) / (2 * d * W);
    return Math.max(0, 1 - Math.hypot(gx, gz) / 0.4);
  }

  // 容量 K を水深/地形から(低頻度で呼ぶ)。9点maxで細川の水深も拾う。
  refreshCapacity() {
    const range = Math.max(1, this.sensor.elevMax - this.sensor.elevMin);
    for (let cj = 0; cj < ECO_N; cj++) for (let ci = 0; ci < ECO_N; ci++) {
      const u = (ci + 0.5) / ECO_N, v = (cj + 0.5) / ECO_N;
      let depth = 0;
      for (let sj = -1; sj <= 1; sj++) for (let si = -1; si <= 1; si++) {
        const w = this.sensor.waterUV(u + si * 0.5 / ECO_N, v + sj * 0.5 / ECO_N);
        if (w > depth) depth = w;
      }
      depth = Math.round(depth / 0.05) * 0.05; // 量子化(膝のflutter防止)
      const kf = K_FISH_MAX * smooth(0.10, 0.60, depth);
      const flat = this.flat(u, v);
      const en = clamp((this.sensor.heightUV(u, v) - this.sensor.elevMin) / range, 0, 1);
      const wet = this.sensor.wetUV(u, v);
      const green = flat * (1 - en) * (0.3 + 0.7 * Math.min(1, wet * 2));
      const kl = K_LAND_MAX * flat * green * (1 - Math.min(1, depth / 0.5));
      const c = cj * ECO_N + ci;
      this.kFish[c] = kf; this.kLand[c] = kl;
    }
    this.refreshed = true;
  }

  tick(dtY: number) {
    if (!this.refreshed) this.refreshCapacity();
    this.stepField(this.fish, this.kFish, R_FISH, SEED_FISH, DECAY_FISH, D_FISH, 1.1 * K_FISH_MAX, dtY);
    this.stepField(this.land, this.kLand, R_LAND, SEED_LAND, DECAY_LAND, D_LAND, 1.1 * K_LAND_MAX, dtY);
  }

  private stepField(N: Float32Array, K: Float32Array, r: number, seed: number, decay: number, D: number, hardMax: number, dtY: number) {
    // logistic + seed + decay
    for (let i = 0; i < N.length; i++) {
      const k = K[i];
      if (k < 1e-3) { N[i] = Math.max(0, N[i] * (1 - decay * dtY)); }
      else { const n = N[i] + (r * N[i] * (1 - N[i] / k) + seed * k) * dtY; N[i] = clamp(n, 0, 1.05 * k); }
    }
    // 拡散(4近傍・明示スキーム)。拡散後は絶対上限でクランプ(低容量セルへの流入暴走防止)。
    if (D > 0) {
      const t = this.tmp; t.set(N);
      const a = D * dtY;
      for (let cj = 0; cj < ECO_N; cj++) for (let ci = 0; ci < ECO_N; ci++) {
        const c = cj * ECO_N + ci;
        const l = t[cj * ECO_N + Math.max(0, ci - 1)], rr = t[cj * ECO_N + Math.min(ECO_N - 1, ci + 1)];
        const u = t[Math.max(0, cj - 1) * ECO_N + ci], d = t[Math.min(ECO_N - 1, cj + 1) * ECO_N + ci];
        N[c] = clamp(t[c] + a * ((l + rr + u + d) * 0.25 - t[c]), 0, hardMax);
      }
    }
  }

  // ── 村クエリ ──
  private near(field: Float32Array, u: number, v: number, rad = 2): number {
    const ci = this.cellClamp(u), cj = this.cellClamp(v);
    let sum = 0, n = 0;
    for (let dj = -rad; dj <= rad; dj++) for (let di = -rad; di <= rad; di++) {
      const i = ci + di, j = cj + dj;
      if (i < 0 || j < 0 || i >= ECO_N || j >= ECO_N) continue;
      sum += field[j * ECO_N + i]; n++;
    }
    return n > 0 ? sum / n : 0;
  }
  fishNear(u: number, v: number) { return this.near(this.fish, u, v); }
  landNear(u: number, v: number) { return this.near(this.land, u, v); }

  // 誘引: 魚/獣が多い水辺の適地度を最大1.2倍。分母は実密度に較正(魚~12でrs≈1)。
  attractFactor(u: number, v: number): number {
    const rs = clamp((this.fishNear(u, v) + 0.6 * this.landNear(u, v)) / 12, 0, 1);
    return clamp(0.8 + 0.4 * rs, 0.8, 1.2);
  }

  // Holling-II 収穫。effort=漁/狩の労働者数。近傍セルから比例で除去し、除去量(=食料 fu)を返す。
  private harvest(field: Float32Array, u: number, v: number, effort: number, Cmax: number, Nhalf: number, dtY: number): number {
    if (effort <= 0) return 0;
    const ci = this.cellClamp(u), cj = this.cellClamp(v), rad = 1; // 収穫は近傍3×3に集中(乱獲可能に)
    let sum = 0; const cells: number[] = [];
    for (let dj = -rad; dj <= rad; dj++) for (let di = -rad; di <= rad; di++) {
      const i = ci + di, j = cj + dj;
      if (i < 0 || j < 0 || i >= ECO_N || j >= ECO_N) continue;
      const c = j * ECO_N + i; cells.push(c); sum += field[c];
    }
    if (sum <= 1e-6) return 0;
    const dens = sum / cells.length;
    let take = effort * Cmax * (dens / (dens + Nhalf)) * dtY;
    take = Math.min(take, MAX_TAKE * sum * dtY / Math.max(dtY, 1e-6)); // maxTakeFraction/年ガード
    take = Math.min(take, sum);
    if (!(take > 0) || !isFinite(take)) return 0;
    const frac = take / sum;
    for (const c of cells) field[c] = Math.max(0, field[c] * (1 - frac));
    return take;
  }
  harvestFish(u: number, v: number, effort: number, dtY: number): number { const t = this.harvest(this.fish, u, v, effort, CMAX_FISH, NHALF_FISH, dtY); this.harvestedFish += t; return t; }
  harvestLand(u: number, v: number, effort: number, dtY: number): number { const t = this.harvest(this.land, u, v, effort, CMAX_LAND, NHALF_LAND, dtY); this.harvestedLand += t; return t; }
  // 漁場ポテンシャル(村が漁師を割り当てるか)
  fishPot(u: number, v: number) { return this.fishNear(u, v) / K_FISH_MAX; }
  huntPot(u: number, v: number) { return this.landNear(u, v) / K_LAND_MAX; }

  totals() { let f = 0, l = 0; for (let i = 0; i < this.fish.length; i++) { f += this.fish[i]; l += this.land[i]; } return { fish: f, land: l }; }

  // ── 描画(eco-tick 時に密度から instance 配置) ──
  render() {
    const W = this.sensor.worldW, d = this.dummy;
    const posAt = (u: number, v: number, yOff: number) => {
      d.position.set(u * W - W / 2, this.sensor.heightUV(u, v) * this.sensor.vertExag + yOff, W / 2 - v * W);
    };
    const sz = W * 0.0016;
    // 魚: 水深十分なセルに密度比で配置
    let fi = 0;
    for (let cj = 0; cj < ECO_N && fi < this.fishCap; cj++) for (let ci = 0; ci < ECO_N && fi < this.fishCap; ci++) {
      const c = cj * ECO_N + ci; const dens = this.fish[c];
      if (dens < 3) continue;
      const u = (ci + 0.5) / ECO_N, v = (cj + 0.5) / ECO_N;
      const water = this.sensor.waterUV(u, v);
      if (water < FISH_WATER_MIN) continue;
      const cnt = Math.min(3, Math.floor(dens / 6) + 1);
      for (let m = 0; m < cnt && fi < this.fishCap; m++) {
        const hsh = ((c * 2654435761 + m * 40503) >>> 0) / 4294967296;
        const ju = (hsh - 0.5) * 0.9 / ECO_N, jv = (((c * 7 + m * 13) % 97) / 97 - 0.5) * 0.9 / ECO_N;
        posAt(u + ju, v + jv, this.sensor.vertExag * water * 0.5 + sz);
        d.rotation.set(0, hsh * 6.28, 0); d.scale.setScalar(sz); d.updateMatrix();
        this.fishMesh.setMatrixAt(fi++, d.matrix);
      }
    }
    for (let k = fi; k < this.fishCap; k++) this.fishMesh.setMatrixAt(k, new THREE.Matrix4().makeScale(0, 0, 0));
    this.fishMesh.count = this.fishCap; this.fishMesh.instanceMatrix.needsUpdate = true;
    // 陸獣
    let ai = 0;
    for (let cj = 0; cj < ECO_N && ai < this.animalCap; cj++) for (let ci = 0; ci < ECO_N && ai < this.animalCap; ci++) {
      const c = cj * ECO_N + ci; const dens = this.land[c];
      if (dens < 2) continue;
      const u = (ci + 0.5) / ECO_N, v = (cj + 0.5) / ECO_N;
      const cnt = Math.min(2, Math.floor(dens / 5) + 1);
      for (let m = 0; m < cnt && ai < this.animalCap; m++) {
        const hsh = ((c * 2246822519 + m * 3266489917) >>> 0) / 4294967296;
        const ju = (hsh - 0.5) * 0.9 / ECO_N, jv = (((c * 11 + m * 17) % 89) / 89 - 0.5) * 0.9 / ECO_N;
        posAt(u + ju, v + jv, 0);
        d.rotation.set(0, hsh * 6.28, 0); d.scale.setScalar(sz * 1.6); d.updateMatrix();
        this.animalMesh.setMatrixAt(ai++, d.matrix);
      }
    }
    for (let k = ai; k < this.animalCap; k++) this.animalMesh.setMatrixAt(k, new THREE.Matrix4().makeScale(0, 0, 0));
    this.animalMesh.count = this.animalCap; this.animalMesh.instanceMatrix.needsUpdate = true;
  }

  // 検証用
  _totals() { return this.totals(); }
  _kMax() { let kf = 0, kl = 0; for (let i = 0; i < this.kFish.length; i++) { if (this.kFish[i] > kf) kf = this.kFish[i]; if (this.kLand[i] > kl) kl = this.kLand[i]; } return { kf, kl }; }
}
