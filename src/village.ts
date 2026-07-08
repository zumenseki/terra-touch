// terra-touch: 創世 — 村システム (G0.3 自律的に動く人)。
// 村は人口で抽象化。見た目の代表=個別エージェントの原始人。各人が自分の目標を持ち
// 村の周りを自律徘徊(目標へ歩く→休む→次の目標)。歩いている時だけ手足が動く。
// band(定住前の一団)は隊列で目的地へ随伴。定住時に民を新しい村へ引き継ぐ。

import * as THREE from 'three/webgpu';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';

export interface WorldSensor {
  worldW: number;
  vertExag: number;
  elevMin: number;
  elevMax: number;
  heightUV(u: number, v: number): number;
  waterUV(u: number, v: number): number;
  wetUV(u: number, v: number): number;
}

interface Village { u: number; v: number; pop: number; huts: number; age: number; }
interface Band { u: number; v: number; wander: number; tu?: number; tv?: number; }

interface Person {
  group: THREE.Group;
  legL: THREE.Group; legR: THREE.Group;
  armL: THREE.Group; armR: THREE.Group;
  body: THREE.Object3D;
  active: boolean;
  homeRef: Village | Band | null;
  homeKind: 'village' | 'band' | null;
  u: number; v: number; tu: number; tv: number;
  state: 'walk' | 'pause';
  timer: number; speed: number; phase: number; facing: number;
  fx: number; fy: number; // band 隊列オフセット
}

const MAX_HUTS = 1500;
const MAX_FIRES = 200;

function paint(geo: THREE.BufferGeometry, hex: number): THREE.BufferGeometry {
  const c = new THREE.Color(hex);
  const n = geo.attributes.position.count;
  const col = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) { col[i * 3] = c.r; col[i * 3 + 1] = c.g; col[i * 3 + 2] = c.b; }
  geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
  return geo;
}

export class VillageSystem {
  group = new THREE.Group();
  villages: Village[] = [];
  private bands: Band[] = [];
  private sensor: WorldSensor;
  private hutMesh: THREE.InstancedMesh;
  private fireMesh: THREE.InstancedMesh;
  private dummy = new THREE.Object3D();
  private U: number;
  private t = 0;
  private pool: Person[] = [];
  private maxPeople: number;
  private stride: number; // 一歩あたりの脚の位相進み(移動距離基準)

  constructor(sensor: WorldSensor, opts?: { maxPeople?: number }) {
    this.sensor = sensor;
    const W = sensor.worldW;
    const U = W * 0.004;
    this.U = U;
    this.stride = U * 0.42;
    this.maxPeople = opts?.maxPeople ?? 40;

    const wall = new THREE.CylinderGeometry(U * 0.55, U * 0.7, U * 0.7, 7); wall.translate(0, U * 0.35, 0);
    const roof = new THREE.ConeGeometry(U * 0.95, U * 0.85, 7); roof.translate(0, U * 0.7 + U * 0.42, 0);
    const hutGeo = mergeGeometries([paint(wall, 0xb39e7d), paint(roof, 0x6f4a2e)])!;
    this.hutMesh = new THREE.InstancedMesh(hutGeo, new THREE.MeshStandardNodeMaterial({ vertexColors: true, roughness: 0.9 }), MAX_HUTS);
    this.hutMesh.count = 0; this.hutMesh.frustumCulled = false; this.group.add(this.hutMesh);

    const fGeo = new THREE.ConeGeometry(U * 0.3, U * 0.7, 6); fGeo.translate(0, U * 0.35, 0);
    this.fireMesh = new THREE.InstancedMesh(fGeo, new THREE.MeshBasicNodeMaterial({ color: 0xff7a26 }), MAX_FIRES);
    this.fireMesh.count = 0; this.fireMesh.frustumCulled = false; this.group.add(this.fireMesh);

    this.buildPeople();
  }

  private buildPeople() {
    const U = this.U;
    const H = U * 1.35;
    const legLen = H * 0.48, legR = H * 0.042;
    const armLen = H * 0.40, armR = H * 0.032;
    const torsoH = H * 0.30;
    const hipY = legLen;
    const skins = [0xb98a5e, 0xa9764a, 0xc79b6e].map((c) => new THREE.MeshStandardNodeMaterial({ color: c, roughness: 0.85 }));
    const hairMat = new THREE.MeshStandardNodeMaterial({ color: 0x201510, roughness: 1 });
    const clothMat = new THREE.MeshStandardNodeMaterial({ color: 0x6b4a2e, roughness: 1 });

    const legGeo = new THREE.CapsuleGeometry(legR, legLen - 2 * legR, 4, 7); legGeo.translate(0, -legLen / 2 + legR, 0);
    const armGeo = new THREE.CapsuleGeometry(armR, armLen - 2 * armR, 4, 7); armGeo.translate(0, -armLen / 2 + armR, 0);
    const torsoGeo = new THREE.CapsuleGeometry(H * 0.095, torsoH - H * 0.095, 5, 9);
    const headGeo = new THREE.SphereGeometry(H * 0.088, 12, 10);
    const hairGeo = new THREE.SphereGeometry(H * 0.096, 12, 8, 0, Math.PI * 2, 0, Math.PI * 0.62);
    const clothGeo = new THREE.CylinderGeometry(H * 0.11, H * 0.125, H * 0.14, 9);
    const footGeo = new THREE.BoxGeometry(legR * 2, legR * 1.1, legR * 2.6); footGeo.translate(0, -legLen + legR * 0.5, legR * 0.6);

    for (let i = 0; i < this.maxPeople; i++) {
      const skin = skins[i % skins.length];
      const g = new THREE.Group();
      const mkLeg = (x: number) => { const p = new THREE.Group(); p.position.set(x, hipY, 0); p.add(new THREE.Mesh(legGeo, skin)); p.add(new THREE.Mesh(footGeo, skin)); g.add(p); return p; };
      const legL = mkLeg(-H * 0.045), legR2 = mkLeg(H * 0.045);
      const body = new THREE.Group();
      const torso = new THREE.Mesh(torsoGeo, skin); torso.position.y = hipY + torsoH * 0.5; body.add(torso);
      const cloth = new THREE.Mesh(clothGeo, clothMat); cloth.position.y = hipY + H * 0.02; body.add(cloth);
      const head = new THREE.Mesh(headGeo, skin); head.position.y = hipY + torsoH + H * 0.1; body.add(head);
      const hair = new THREE.Mesh(hairGeo, hairMat); hair.position.y = hipY + torsoH + H * 0.12; body.add(hair);
      const mkArm = (x: number) => { const p = new THREE.Group(); p.position.set(x, hipY + torsoH * 0.92, 0); p.add(new THREE.Mesh(armGeo, skin)); body.add(p); return p; };
      const armL = mkArm(-H * 0.12), armR3 = mkArm(H * 0.12);
      g.add(body); g.visible = false; this.group.add(g);
      this.pool.push({
        group: g, legL, legR: legR2, armL, armR: armR3, body,
        active: false, homeRef: null, homeKind: null,
        u: 0, v: 0, tu: 0, tv: 0, state: 'walk', timer: 0, speed: 0.007, phase: i * 1.7, facing: 0, fx: 0, fy: 0,
      });
    }
  }

  seed(u: number, v: number) { this.bands.push({ u, v, wander: 0 }); }

  clear() {
    this.villages = []; this.bands = [];
    this.hutMesh.count = 0; this.fireMesh.count = 0;
    for (const p of this.pool) { p.active = false; p.homeRef = null; p.group.visible = false; }
  }

  private pos(u: number, v: number, out: THREE.Vector3) {
    const W = this.sensor.worldW;
    out.set(u * W - W / 2, this.sensor.heightUV(u, v) * this.sensor.vertExag, W / 2 - v * W);
  }
  private flat(u: number, v: number): number {
    const W = this.sensor.worldW; const d = 3 / W;
    const h = this.sensor.heightUV.bind(this.sensor);
    const gx = (h(u + d, v) - h(u - d, v)) / (2 * d * W);
    const gz = (h(u, v + d) - h(u, v - d)) / (2 * d * W);
    return Math.max(0, 1 - Math.hypot(gx, gz) / 0.4);
  }
  private suitability(u: number, v: number): number {
    const flat = this.flat(u, v);
    if (flat <= 0) return 0;
    const waterScore = Math.min(1, this.sensor.wetUV(u, v) * 2 + this.sensor.waterUV(u, v) * 0.4);
    const range = Math.max(1, this.sensor.elevMax - this.sensor.elevMin);
    const en = Math.min(1, Math.max(0, (this.sensor.heightUV(u, v) - this.sensor.elevMin) / range));
    return flat * (0.55 + 0.45 * waterScore) * (0.45 + 0.55 * (1 - en * 0.7));
  }

  private countHome(ref: Village | Band): number {
    let n = 0; for (const p of this.pool) if (p.active && p.homeRef === ref) n++; return n;
  }
  private assignFree(ref: Village | Band, kind: 'village' | 'band', u: number, v: number): Person | null {
    const p = this.pool.find((x) => !x.active);
    if (!p) return null;
    p.active = true; p.homeRef = ref; p.homeKind = kind;
    p.u = u; p.v = v; p.tu = u; p.tv = v; p.state = 'pause'; p.timer = Math.random() * 1.5;
    p.speed = 0.005 + Math.random() * 0.006;
    p.fx = (Math.random() - 0.5) * this.U * 0.7 / this.sensor.worldW;
    p.fy = (Math.random() - 0.5) * this.U * 0.7 / this.sensor.worldW;
    p.group.visible = true;
    return p;
  }
  private release(p: Person) { p.active = false; p.homeRef = null; p.homeKind = null; p.group.visible = false; }

  private wanderTarget(p: Person, vg: Village) {
    const W = this.sensor.worldW;
    const r = (this.U * (0.8 + Math.random() * 3)) / W;
    const a = Math.random() * Math.PI * 2;
    p.tu = Math.min(0.98, Math.max(0.02, vg.u + Math.cos(a) * r));
    p.tv = Math.min(0.98, Math.max(0.02, vg.v + Math.sin(a) * r));
  }

  update(dt: number) {
    this.t += dt;
    const stepUv = 0.06 * dt;

    // ── band: 適地スキャン→直進→定住 ──
    for (let b = this.bands.length - 1; b >= 0; b--) {
      const band = this.bands[b];
      if (band.tu === undefined) {
        let best = this.suitability(band.u, band.v) + 0.02, tu = band.u, tv = band.v;
        for (let ri = 1; ri <= 5; ri++) {
          const rad = 0.05 * ri;
          for (let a = 0; a < 12; a++) {
            const ang = (a / 12) * Math.PI * 2 + ri * 0.7;
            const nu = band.u + Math.cos(ang) * rad, nv = band.v + Math.sin(ang) * rad;
            if (nu < 0.04 || nu > 0.96 || nv < 0.04 || nv > 0.96) continue;
            const s = this.suitability(nu, nv);
            if (s > best) { best = s; tu = nu; tv = nv; }
          }
        }
        band.tu = tu; band.tv = tv;
      }
      // 民を 3 人確保
      while (this.countHome(band) < 3 && this.assignFree(band, 'band', band.u, band.v)) { /* */ }
      band.wander += dt;
      const du = band.tu - band.u, dv = (band.tv ?? band.v) - band.v;
      const dist = Math.hypot(du, dv);
      if (dist < stepUv * 1.5 || band.wander > 14) {
        const vg: Village = { u: band.u, v: band.v, pop: 4, huts: 1, age: 0 };
        this.villages.push(vg);
        // band の民を村へ引き継ぐ
        for (const p of this.pool) if (p.homeRef === band) { p.homeRef = vg; p.homeKind = 'village'; this.wanderTarget(p, vg); }
        this.bands.splice(b, 1);
        continue;
      }
      band.u += (du / dist) * stepUv; band.v += (dv / dist) * stepUv;
    }

    // ── village: 人口・小屋・民の割当 ──
    const capPerV = Math.max(2, Math.floor(this.maxPeople / Math.max(1, this.villages.length)));
    for (const vg of this.villages) {
      vg.age += dt;
      const s = this.suitability(vg.u, vg.v);
      if (s > 0.4) vg.pop += dt * s * 2.2; else vg.pop = Math.max(1, vg.pop - dt * 0.4);
      vg.pop = Math.min(vg.pop, 140);
      vg.huts = Math.min(12, Math.max(1, Math.floor(vg.pop / 6) + 1));
      // 民の数を pop に合わせる
      const want = Math.min(capPerV, Math.max(3, Math.ceil(vg.pop / 7)));
      let have = this.countHome(vg);
      while (have < want) { const np = this.assignFree(vg, 'village', vg.u, vg.v); if (!np) break; this.wanderTarget(np, vg); np.state = 'walk'; have++; }
      while (have > want) { const p = this.pool.find((x) => x.homeRef === vg); if (!p) break; this.release(p); have--; }
    }

    // ── 各人を自律更新 ──
    for (const p of this.pool) if (p.active) this.updatePerson(p, dt);

    this.renderStatic();
  }

  private updatePerson(p: Person, dt: number) {
    const W = this.sensor.worldW, U = this.U, d = this.dummy;
    let moved = 0;
    if (p.homeKind === 'band' && p.homeRef) {
      const band = p.homeRef as Band;
      p.tu = band.u + p.fx; p.tv = band.v + p.fy;
      moved = this.stepToward(p, p.speed * 1.6 * dt);
    } else {
      const vg = p.homeRef as Village;
      if (p.state === 'pause') {
        p.timer -= dt;
        if (p.timer <= 0) { this.wanderTarget(p, vg); p.state = 'walk'; }
      } else {
        moved = this.stepToward(p, p.speed * dt);
        const du = p.tu - p.u, dv = p.tv - p.v;
        if (Math.hypot(du, dv) < p.speed * dt * 1.2) { p.state = 'pause'; p.timer = 0.6 + Math.random() * 3; }
      }
    }
    // 歩行アニメ (移動量に同期=止まると足も止まる)
    p.phase += (moved * W) / this.stride;
    const sw = Math.sin(p.phase) * 0.6;
    p.legL.rotation.x = sw; p.legR.rotation.x = -sw;
    p.armL.rotation.x = -sw * 0.7; p.armR.rotation.x = sw * 0.7;
    p.body.position.y = Math.abs(Math.cos(p.phase)) * U * 0.03;
    // 配置
    this.pos(p.u, p.v, d.position);
    p.group.position.copy(d.position);
    p.group.rotation.y = p.facing;
  }

  // 目標へ一歩。戻り値=移動した uv 距離
  private stepToward(p: Person, step: number): number {
    const du = p.tu - p.u, dv = p.tv - p.v;
    const dist = Math.hypot(du, dv);
    if (dist < 1e-6) return 0;
    p.facing = Math.atan2(du, -dv);
    const s = Math.min(step, dist);
    p.u += (du / dist) * s; p.v += (dv / dist) * s;
    return s;
  }

  private renderStatic() {
    const W = this.sensor.worldW, U = this.U, d = this.dummy;
    let hi = 0;
    for (const vg of this.villages) {
      for (let k = 0; k < vg.huts && hi < MAX_HUTS; k++) {
        const ang = k * 2.399963;
        const rad = U * 1.3 * Math.sqrt(k);
        const u = vg.u + (Math.cos(ang) * rad) / W, v = vg.v + (Math.sin(ang) * rad) / W;
        this.pos(u, v, d.position); d.rotation.set(0, ang * 2.1, 0);
        d.scale.setScalar(0.8 + 0.25 * ((k * 7) % 5) / 5); d.updateMatrix();
        this.hutMesh.setMatrixAt(hi++, d.matrix);
      }
    }
    this.hutMesh.count = hi; this.hutMesh.instanceMatrix.needsUpdate = true;
    let fi = 0;
    for (const vg of this.villages) {
      if (fi >= MAX_FIRES) break;
      this.pos(vg.u, vg.v, d.position); d.rotation.set(0, 0, 0);
      d.scale.set(1, 0.85 + 0.25 * Math.sin(this.t * 6 + fi), 1); d.updateMatrix();
      this.fireMesh.setMatrixAt(fi++, d.matrix);
    }
    this.fireMesh.count = fi; this.fireMesh.instanceMatrix.needsUpdate = true;
  }

  stats() {
    let pop = 0; for (const v of this.villages) pop += v.pop;
    return { villages: this.villages.length, pop: Math.round(pop), bands: this.bands.length };
  }
}
