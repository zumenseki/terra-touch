// terra-touch: 創世 — 生態システム (N6 魚+陸獣)。密度フィールド駆動(個体AIなし・決定論)。
// 容量 K は水深/平地/緑度から。logistic 成長+seed+線形decay+拡散。村が Holling-II で収穫し
// 魚/獣が多い水辺の適地度を上げて人を呼ぶ(誘引)。描画=eco-tick が retarget でスロット目標を
// 割当て、animate が毎フレーム泳ぎ/歩きで追従(InstancedMesh)。詳細裏設定: docs/SPEC-life-sim.md §7。

import * as THREE from 'three/webgpu';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import type { WorldSensor } from './village';

const ECO_N = 32;
const R_FISH = 1.4, SEED_FISH = 0.2, DECAY_FISH = 10;
const R_LAND = 0.5, SEED_LAND = 0.15, DECAY_LAND = 6;
const D_FISH = 1.0, D_LAND = 0.6;      // 拡散 /年
const K_FISH_MAX = 24, K_LAND_MAX = 16;
const CMAX_FISH = 6, CMAX_LAND = 3, NHALF_FISH = 6, NHALF_LAND = 4, MAX_TAKE = 0.4;
const FISH_WATER_MIN = 0.15;
const LAND_SHOW_MIN = 2.2; // 獣を描く密度しきい。鹿は適正サイズなので数がいてよい(要望=動物を増やす)

const clamp = (x: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, x));
const smooth = (a: number, b: number, x: number) => { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };
const slotHash = (i: number, salt: number) => (((i + 1) * salt) >>> 0) / 4294967296;
const angWrap = (a: number) => a - Math.round(a / (Math.PI * 2)) * Math.PI * 2;

// 泳ぎ/歩きの永続スロット(view専用)。eco-tick が目標(tu,tv)を書き、animate が毎フレーム追従させる。
// 従来は eco-tick(0.5秒)毎に行列を置き直し=位置がワープしていた。
class SlotAnim {
  act: Uint8Array; fade: Float32Array;
  u: Float32Array; v: Float32Array; tu: Float32Array; tv: Float32Array;
  ph: Float32Array; yaw: Float32Array; rad: Float32Array;
  constructor(n: number) {
    this.act = new Uint8Array(n); this.fade = new Float32Array(n);
    this.u = new Float32Array(n); this.v = new Float32Array(n);
    this.tu = new Float32Array(n); this.tv = new Float32Array(n);
    this.ph = new Float32Array(n); this.yaw = new Float32Array(n); this.rad = new Float32Array(n);
  }
  target(i: number, tu: number, tv: number) {
    this.tu[i] = tu; this.tv[i] = tv;
    if (!this.act[i]) {
      this.act[i] = 1;
      if (this.fade[i] <= 0.01) { this.u[i] = tu; this.v[i] = tv; } // 消え切っていたらその場に湧く(フェードイン)
    }
  }
  deactivateFrom(n: number) { for (let i = n; i < this.act.length; i++) this.act[i] = 0; }
  reset() { this.act.fill(0); this.fade.fill(0); this.rad.fill(0); }
}

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
  private fishAnim: SlotAnim;
  private landAnim: SlotAnim;
  private animT = 0;
  private mZero = new THREE.Matrix4().makeScale(0, 0, 0);

  constructor(sensor: WorldSensor, opts?: { fishCap?: number; animalCap?: number }) {
    this.sensor = sensor;
    this.fishCap = opts?.fishCap ?? 120;
    this.animalCap = opts?.animalCap ?? 60;
    // 魚 = 扁平ダイヤ(菱形)+尾びれ。
    // 🔴 mergeGeometries は index の有無が揃っていないと null を返す。
    //    Octahedron=非インデックス / Cone=インデックス付き → 尾を toNonIndexed() で揃える。
    const fishBody = new THREE.OctahedronGeometry(1, 0); fishBody.scale(1.5, 0.45, 0.75);
    const fishTailI = new THREE.ConeGeometry(0.5, 0.7, 4); fishTailI.rotateZ(Math.PI / 2); fishTailI.scale(1, 1, 0.35); fishTailI.translate(-1.7, 0, 0);
    const fishGeo = mergeGeometries([fishBody, fishTailI.toNonIndexed()]);
    if (!fishGeo) throw new Error('fish geometry merge failed');
    const fishMat = new THREE.MeshStandardNodeMaterial({ color: 0x8899aa, roughness: 0.5, metalness: 0.1 });
    this.fishMesh = new THREE.InstancedMesh(fishGeo, fishMat, this.fishCap);
    this.fishMesh.frustumCulled = false; this.fishMesh.count = this.fishCap;
    // 獣 = 鹿の抽象(胴+4脚+首+頭)。単位長~1.2・高さ~1.15 で作り、描画時にスケール。
    const aBody = new THREE.BoxGeometry(1.0, 0.42, 0.38); aBody.translate(0, 0.58, 0);
    const aLeg = (x: number, z: number) => { const l = new THREE.BoxGeometry(0.09, 0.5, 0.09); l.translate(x, 0.25, z); return l; };
    const aNeck = new THREE.BoxGeometry(0.15, 0.36, 0.15); aNeck.translate(0.44, 0.84, 0);
    const aHead = new THREE.BoxGeometry(0.3, 0.17, 0.16); aHead.translate(0.57, 1.0, 0);
    const animalGeo = mergeGeometries([aBody, aLeg(-0.34, 0.13), aLeg(-0.34, -0.13), aLeg(0.34, 0.13), aLeg(0.34, -0.13), aNeck, aHead]);
    if (!animalGeo) throw new Error('animal geometry merge failed');
    const animalMat = new THREE.MeshStandardNodeMaterial({ color: 0x7a5a3a, roughness: 0.9 });
    this.animalMesh = new THREE.InstancedMesh(animalGeo, animalMat, this.animalCap);
    this.animalMesh.frustumCulled = false; this.animalMesh.count = this.animalCap;
    for (let i = 0; i < this.fishCap; i++) this.fishMesh.setMatrixAt(i, this.mZero);
    for (let i = 0; i < this.animalCap; i++) this.animalMesh.setMatrixAt(i, this.mZero);
    this.fishAnim = new SlotAnim(this.fishCap);
    this.landAnim = new SlotAnim(this.animalCap);
    this.group.add(this.fishMesh); this.group.add(this.animalMesh);
  }

  clear() {
    this.fish.fill(0); this.land.fill(0); this.refreshed = false;
    this.harvestedFish = 0; this.harvestedLand = 0;
    this.fishAnim.reset(); this.landAnim.reset();
    for (let i = 0; i < this.fishCap; i++) this.fishMesh.setMatrixAt(i, this.mZero);
    for (let i = 0; i < this.animalCap; i++) this.animalMesh.setMatrixAt(i, this.mZero);
    this.fishMesh.instanceMatrix.needsUpdate = true;
    this.animalMesh.instanceMatrix.needsUpdate = true;
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

  // セル内で実際に水がある地点を探す。🔴 eco セル(≈1km)に対し川幅は数百m ＝ セル中心や
  // ランダムジッタで置くと陸に乗る(実機FB「水ないところを泳いでいる」の原因)。4×4 を走査し、
  // 個体ごとに開始位置をずらす(同セルの魚が1点に固まらない)。見つからなければ null=出さない。
  private findWater(cu: number, cv: number, start: number): { u: number; v: number } | null {
    const S = 4, total = S * S;
    for (let n = 0; n < total; n++) {
      const m = (n + start) % total;
      const su = ((m % S) + 0.5) / S - 0.5, sv = (Math.floor(m / S) + 0.5) / S - 0.5;
      const u = cu + su * 0.95 / ECO_N, v = cv + sv * 0.95 / ECO_N;
      if (this.sensor.waterUV(u, v) >= FISH_WATER_MIN) return { u, v };
    }
    return null;
  }

  // ── eco-tick: 密度からスロット目標を割当(移動/行列書込は animate が毎フレーム行う) ──
  retarget() {
    // 魚: 密度のあるセルの「水がある地点」に目標を置く
    let fi = 0;
    for (let cj = 0; cj < ECO_N && fi < this.fishCap; cj++) for (let ci = 0; ci < ECO_N && fi < this.fishCap; ci++) {
      const c = cj * ECO_N + ci; const dens = this.fish[c];
      if (dens < 3) continue;
      const u = (ci + 0.5) / ECO_N, v = (cj + 0.5) / ECO_N;
      const cnt = Math.min(3, Math.floor(dens / 6) + 1);
      for (let m = 0; m < cnt && fi < this.fishCap; m++) {
        const hsh = ((c * 2654435761 + m * 40503) >>> 0) / 4294967296;
        const spot = this.findWater(u, v, Math.floor(hsh * 16) + m * 5);
        if (!spot) break; // このセルは水無し=魚を出さない
        this.fishAnim.target(fi++, spot.u, spot.v);
      }
    }
    this.fishAnim.deactivateFrom(fi);
    // 陸獣: 生息地に集める(全面に湧かせない)
    let ai = 0;
    for (let cj = 0; cj < ECO_N && ai < this.animalCap; cj++) for (let ci = 0; ci < ECO_N && ai < this.animalCap; ci++) {
      const c = cj * ECO_N + ci; const dens = this.land[c];
      if (dens < LAND_SHOW_MIN) continue;
      const u = (ci + 0.5) / ECO_N, v = (cj + 0.5) / ECO_N;
      const cnt = Math.min(2, Math.floor(dens / 6) + 1);
      for (let m = 0; m < cnt && ai < this.animalCap; m++) {
        const hsh = ((c * 2246822519 + m * 3266489917) >>> 0) / 4294967296;
        const ju = (hsh - 0.5) * 0.9 / ECO_N, jv = (((c * 11 + m * 17) % 89) / 89 - 0.5) * 0.9 / ECO_N;
        this.landAnim.target(ai++, u + ju, v + jv);
      }
    }
    this.landAnim.deactivateFrom(ai);
  }

  // ── 毎フレーム: 魚=アンカーへ滑らか接近+小周回遊泳、獣=ゆっくり歩き。出現/消滅はフェード ──
  // view専用(ハッシュ駆動・rng不使用)なので sim の決定論/cap非依存に影響しない。
  animate(dt: number) {
    this.animT += dt;
    const W = this.sensor.worldW, d = this.dummy, ex = this.sensor.vertExag;
    // 実寸基準: 人の背丈 H = W*0.004*1.35 ≈ 172。魚は人の1/5、鹿は人の0.55倍の背丈。
    const H = W * 0.004 * 1.35;
    const fishSz = H * 0.10;   // 菱形半径 → 魚の全長 ~ H*0.3
    const deerSz = H * 0.55;   // 単位獣(高さ~1.15)→ 鹿の背丈 ~ H*0.63
    const fs = this.fishAnim;
    for (let i = 0; i < this.fishCap; i++) {
      if (!fs.act[i] && fs.fade[i] <= 0) continue;
      const h1 = slotHash(i, 2654435761), h2 = slotHash(i, 2246822519), h3 = slotHash(i, 3266489917);
      const omg = (0.6 + h1 * 0.9) * (h2 < 0.5 ? 1 : -1);  // 周回角速度 rad/s
      const radMax = (0.04 + h2 * 0.10) / ECO_N;            // 周回半径の上限(川幅より小さめに取る)
      fs.ph[i] += omg * dt;
      let vu = 0, vv = 0;
      const du = fs.tu[i] - fs.u[i], dv = fs.tv[i] - fs.v[i], dist = Math.hypot(du, dv);
      if (dist > 1e-6) {
        const spd = Math.min(dist * 1.5, 0.05);             // uv/s: 距離比例+上限(遠い再割当も泳いで移動)
        vu = (du / dist) * spd; vv = (dv / dist) * spd;
        const step = Math.min(spd * dt, dist);
        fs.u[i] += (du / dist) * step; fs.v[i] += (dv / dist) * step;
      }
      fs.fade[i] = clamp(fs.fade[i] + (fs.act[i] ? dt : -dt) * 2.5, 0, 1);
      const th = fs.ph[i];
      // 周回は水がある間だけ広げる。はみ出す位置ならアンカー(retargetが水と確認済)へ退避。
      let rad = Math.min(fs.rad[i] + radMax * dt, radMax);
      let pu = fs.u[i] + Math.cos(th) * rad, pv = fs.v[i] + Math.sin(th) * rad;
      let water = this.sensor.waterUV(pu, pv);
      if (water < FISH_WATER_MIN) {
        rad = 0; pu = fs.u[i]; pv = fs.v[i];
        water = this.sensor.waterUV(pu, pv);
        if (water < FISH_WATER_MIN) fs.act[i] = 0; // アンカーも干上がった=フェードアウト
      }
      fs.rad[i] = rad;
      vu -= Math.sin(th) * rad * omg; vv += Math.cos(th) * rad * omg;
      if (Math.hypot(vu, vv) > 1e-9) fs.yaw[i] += angWrap(Math.atan2(vv, vu) - fs.yaw[i]) * Math.min(1, dt * 5);
      const sc = fishSz * (0.8 + 0.4 * h3) * fs.fade[i];
      if (sc <= 1e-4) { this.fishMesh.setMatrixAt(i, this.mZero); continue; }
      const bed = this.sensor.heightUV(pu, pv) * ex, half = sc * 0.45;
      d.position.set(
        pu * W - W / 2,
        Math.max(bed + half, bed + ex * water * 0.45 + Math.sin(th * 2.3 + h3 * 6.28) * fishSz * 0.15),
        W / 2 - pv * W);
      d.rotation.set(Math.sin(th * 2 + h1 * 6.28) * 0.15, fs.yaw[i], 0);
      d.scale.setScalar(sc); d.updateMatrix();
      this.fishMesh.setMatrixAt(i, d.matrix);
    }
    this.fishMesh.instanceMatrix.needsUpdate = true;
    // 獣: 目標へゆっくり歩き(歩幅bob+進行方向へ向く)。待機中は草を食む微小な首振りのみ。
    const as = this.landAnim;
    for (let i = 0; i < this.animalCap; i++) {
      if (!as.act[i] && as.fade[i] <= 0) continue;
      const h1 = slotHash(i, 2654435761), h2 = slotHash(i, 2246822519), h3 = slotHash(i, 3266489917);
      let moving = 0;
      const du = as.tu[i] - as.u[i], dv = as.tv[i] - as.v[i], dist = Math.hypot(du, dv);
      if (dist > 1e-5) {
        const spd = Math.min(dist * 1.2, 0.012);
        const step = Math.min(spd * dt, dist);
        as.u[i] += (du / dist) * step; as.v[i] += (dv / dist) * step;
        as.ph[i] += (step * W) / (deerSz * 0.6);            // 歩幅~鹿の6割で bob 位相を進める
        as.yaw[i] += angWrap(Math.atan2(dv, du) - as.yaw[i]) * Math.min(1, dt * 4);
        moving = Math.min(1, dist * ECO_N * 3);
      }
      as.fade[i] = clamp(as.fade[i] + (as.act[i] ? dt : -dt) * 2.5, 0, 1);
      const sc = deerSz * (0.85 + 0.3 * h3) * as.fade[i];
      if (sc <= 1e-4) { this.animalMesh.setMatrixAt(i, this.mZero); continue; }
      const bob = Math.abs(Math.sin(as.ph[i])) * deerSz * 0.05 * moving;
      d.position.set(as.u[i] * W - W / 2, this.sensor.heightUV(as.u[i], as.v[i]) * ex + bob, W / 2 - as.v[i] * W);
      d.rotation.set(0, as.yaw[i] + Math.sin(this.animT * (0.25 + h1 * 0.3) + h2 * 6.28) * 0.08, 0);
      d.scale.setScalar(sc); d.updateMatrix();
      this.animalMesh.setMatrixAt(i, d.matrix);
    }
    this.animalMesh.instanceMatrix.needsUpdate = true;
  }

  // 検証用
  _totals() { return this.totals(); }
  _kMax() { let kf = 0, kl = 0; for (let i = 0; i < this.kFish.length; i++) { if (this.kFish[i] > kf) kf = this.kFish[i]; if (this.kLand[i] > kl) kl = this.kLand[i]; } return { kf, kl }; }
}
