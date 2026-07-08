// terra-touch: 創世 — 村システム (G0)。
// 神(=地形/水の変形)に対して住民は自律反応。個体は管理せず「村単位」で抽象化し、
// 見た目用に代表の人を数体だけ動かす。band=定住前のさまよう一団 → 適地で Village 化。

import * as THREE from 'three/webgpu';

export interface WorldSensor {
  worldW: number;
  vertExag: number;
  /** 生標高(m・VERT_EXAG適用前) */
  heightUV(u: number, v: number): number;
  /** 水深(m) */
  waterUV(u: number, v: number): number;
  /** 湿り 0..1 */
  wetUV(u: number, v: number): number;
}

interface Village {
  u: number; v: number;
  pop: number;
  huts: number;
  age: number;
}
interface Band {
  u: number; v: number;
  wander: number;
}

const MAX_HUTS = 3000;
const MAX_PEOPLE = 400;

export class VillageSystem {
  group = new THREE.Group();
  villages: Village[] = [];
  private bands: Band[] = [];
  private sensor: WorldSensor;
  private hutMesh: THREE.InstancedMesh;
  private peopleMesh: THREE.InstancedMesh;
  private dummy = new THREE.Object3D();
  private hutSize: number;
  private peopleSize: number;
  private people: { u: number; v: number; a: number; r: number; vi: number }[] = [];
  private settleThresh = 0.55;

  constructor(sensor: WorldSensor) {
    this.sensor = sensor;
    const W = sensor.worldW;
    this.hutSize = W * 0.006;
    this.peopleSize = W * 0.0032;

    // 小屋 = 円錐(テント/竪穴)。石器時代の見た目。
    const hutGeo = new THREE.ConeGeometry(this.hutSize * 0.7, this.hutSize * 1.3, 6);
    hutGeo.translate(0, this.hutSize * 0.65, 0);
    const hutMat = new THREE.MeshStandardNodeMaterial({ color: 0x8a6b4a, roughness: 0.9 });
    this.hutMesh = new THREE.InstancedMesh(hutGeo, hutMat, MAX_HUTS);
    this.hutMesh.count = 0;
    this.hutMesh.frustumCulled = false;
    this.group.add(this.hutMesh);

    // 人 = 小さな暗い円錐
    const pGeo = new THREE.ConeGeometry(this.peopleSize * 0.5, this.peopleSize * 1.6, 5);
    pGeo.translate(0, this.peopleSize * 0.8, 0);
    const pMat = new THREE.MeshStandardNodeMaterial({ color: 0x3a2c22, roughness: 1 });
    this.peopleMesh = new THREE.InstancedMesh(pGeo, pMat, MAX_PEOPLE);
    this.peopleMesh.count = 0;
    this.peopleMesh.frustumCulled = false;
    this.group.add(this.peopleMesh);
  }

  /** (u,v) に原始人の一団を置く */
  seed(u: number, v: number) {
    this.bands.push({ u, v, wander: 0 });
  }

  clear() {
    this.villages = []; this.bands = []; this.people = [];
    this.hutMesh.count = 0; this.peopleMesh.count = 0;
  }

  // world(u,v) → シーン座標
  private pos(u: number, v: number, out: THREE.Vector3) {
    const W = this.sensor.worldW;
    out.set(u * W - W / 2, this.sensor.heightUV(u, v) * this.sensor.vertExag, W / 2 - v * W);
  }

  // 平坦度 0..1 (生標高の近傍差から)
  private flat(u: number, v: number): number {
    const W = this.sensor.worldW;
    const d = 3 / this.sensor.worldW; // ~3px相当の微小オフセット(uv)
    const h = this.sensor.heightUV.bind(this.sensor);
    const gx = (h(u + d, v) - h(u - d, v)) / (2 * d * W);
    const gz = (h(u, v + d) - h(u, v - d)) / (2 * d * W);
    const slope = Math.hypot(gx, gz);
    return Math.max(0, 1 - slope / 0.4); // 勾配0.4(~22°)で0
  }

  // 適地スコア: 平地主体 + 水/湿りで加点
  private suitability(u: number, v: number): number {
    const flat = this.flat(u, v);
    if (flat <= 0) return 0;
    const water = this.sensor.waterUV(u, v);
    const wet = this.sensor.wetUV(u, v);
    const waterScore = Math.min(1, wet * 2 + water * 0.4);
    return flat * (0.55 + 0.45 * waterScore);
  }

  update(dt: number) {
    const stepUv = (this.sensor.worldW * 0.03 * dt) / this.sensor.worldW; // 移動速度
    // ── band: 適地へ hill-climb ──
    for (let b = this.bands.length - 1; b >= 0; b--) {
      const band = this.bands[b];
      const here = this.suitability(band.u, band.v);
      // 近傍8方向で最良を探す
      let bestS = here, bu = band.u, bv = band.v;
      const probe = stepUv * 3;
      for (let k = 0; k < 8; k++) {
        const a = (k / 8) * Math.PI * 2;
        const nu = band.u + Math.cos(a) * probe, nv = band.v + Math.sin(a) * probe;
        if (nu < 0.02 || nu > 0.98 || nv < 0.02 || nv > 0.98) continue;
        const s = this.suitability(nu, nv);
        if (s > bestS) { bestS = s; bu = nu; bv = nv; }
      }
      band.wander += dt;
      if (bestS > here + 1e-4) {
        // より良い方へ進む
        band.u += Math.sign(bu - band.u) * Math.min(Math.abs(bu - band.u), stepUv);
        band.v += Math.sign(bv - band.v) * Math.min(Math.abs(bv - band.v), stepUv);
      } else if (here >= this.settleThresh || band.wander > 8) {
        // 極大に到達 or 探し疲れ → 定住
        this.villages.push({ u: band.u, v: band.v, pop: 3, huts: 0, age: 0 });
        this.bands.splice(b, 1);
        this.spawnPeople(band.u, band.v, 3);
      }
    }

    // ── village: 人口増減 + 建物 ──
    for (const vg of this.villages) {
      vg.age += dt;
      const s = this.suitability(vg.u, vg.v);
      if (s > 0.4) vg.pop += dt * s * 2.0; // 適地で増える
      else vg.pop = Math.max(1, vg.pop - dt * 0.5);
      vg.pop = Math.min(vg.pop, 120);
      // 小屋数 = pop に応じて (5人/軒 目安)
      const wantHuts = Math.min(30, Math.floor(vg.pop / 5) + 1);
      if (wantHuts > vg.huts) { vg.huts = wantHuts; }
      // 人口に応じて代表の人を増やす(cap)
      const wantPeople = Math.min(6, Math.ceil(vg.pop / 12));
      const cur = this.people.filter((p) => p.vi === this.villages.indexOf(vg)).length;
      if (cur < wantPeople) this.spawnPeople(vg.u, vg.v, wantPeople - cur);
    }

    this.updateMeshes(dt);
  }

  private spawnPeople(u: number, v: number, n: number) {
    const vi = this.findVillage(u, v);
    for (let k = 0; k < n; k++) {
      if (this.people.length >= MAX_PEOPLE) break;
      this.people.push({ u, v, a: Math.random() * Math.PI * 2, r: 0, vi });
    }
  }
  private findVillage(u: number, v: number): number {
    let best = -1, bd = 1e9;
    for (let i = 0; i < this.villages.length; i++) {
      const dd = (this.villages[i].u - u) ** 2 + (this.villages[i].v - v) ** 2;
      if (dd < bd) { bd = dd; best = i; }
    }
    return best;
  }

  private updateMeshes(dt: number) {
    const W = this.sensor.worldW;
    const d = this.dummy;
    // 小屋: 各村の周りに螺旋配置
    let hi = 0;
    const hutRadius = this.hutSize * 2.2;
    for (const vg of this.villages) {
      for (let k = 0; k < vg.huts && hi < MAX_HUTS; k++) {
        const ang = k * 2.399963; // 黄金角
        const rad = hutRadius * Math.sqrt(k) * 0.6;
        const du = (Math.cos(ang) * rad) / W, dv = (Math.sin(ang) * rad) / W;
        const u = vg.u + du, v = vg.v + dv;
        this.pos(u, v, d.position);
        d.rotation.set(0, ang, 0);
        const sc = 0.7 + 0.3 * ((k * 13) % 7) / 7;
        d.scale.setScalar(sc);
        d.updateMatrix();
        this.hutMesh.setMatrixAt(hi++, d.matrix);
      }
    }
    this.hutMesh.count = hi;
    this.hutMesh.instanceMatrix.needsUpdate = true;

    // 人: 村の周りをうろつく
    let pi = 0;
    for (const p of this.people) {
      if (pi >= MAX_PEOPLE) break;
      const vg = this.villages[p.vi] ?? this.villages[0];
      if (!vg) continue;
      p.a += dt * (0.6 + (pi % 5) * 0.15);
      const rad = this.hutSize * (2.5 + 2 * ((pi % 4) / 4));
      const u = vg.u + (Math.cos(p.a) * rad) / W;
      const v = vg.v + (Math.sin(p.a) * rad) / W;
      this.pos(u, v, d.position);
      d.rotation.set(0, -p.a, 0);
      d.scale.setScalar(1);
      d.updateMatrix();
      this.peopleMesh.setMatrixAt(pi++, d.matrix);
    }
    this.peopleMesh.count = pi;
    this.peopleMesh.instanceMatrix.needsUpdate = true;
  }

  stats() {
    let pop = 0;
    for (const v of this.villages) pop += v.pop;
    return { villages: this.villages.length, pop: Math.round(pop), bands: this.bands.length };
  }
}
