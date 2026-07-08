// terra-touch: 創世 — 村システム (G0.2 リアルな原始人)。
// 住民は村単位で抽象化。見た目用の代表は「関節のある歩く原始人」= 頭/髪/胴/腰布/腕/脚/足を
// 共有ジオメトリで組み、股・肩から手足を振る歩行アニメ。プールで表示数を cap。

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
  phase: number; baseY: number;
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
  private t = 0; private dt = 1 / 60;

  private pool: Person[] = [];
  private maxPeople: number;
  private tmp = new THREE.Vector3();

  constructor(sensor: WorldSensor, opts?: { maxPeople?: number }) {
    this.sensor = sensor;
    const W = sensor.worldW;
    const U = W * 0.004;
    this.U = U;
    this.maxPeople = opts?.maxPeople ?? 40;

    // 小屋 = 壁+屋根 (頂点色2トーン)
    const wall = new THREE.CylinderGeometry(U * 0.55, U * 0.7, U * 0.7, 7); wall.translate(0, U * 0.35, 0);
    const roof = new THREE.ConeGeometry(U * 0.95, U * 0.85, 7); roof.translate(0, U * 0.7 + U * 0.42, 0);
    const hutGeo = mergeGeometries([paint(wall, 0xb39e7d), paint(roof, 0x6f4a2e)])!;
    this.hutMesh = new THREE.InstancedMesh(hutGeo, new THREE.MeshStandardNodeMaterial({ vertexColors: true, roughness: 0.9 }), MAX_HUTS);
    this.hutMesh.count = 0; this.hutMesh.frustumCulled = false; this.group.add(this.hutMesh);

    // 焚き火 = 常時オレンジ
    const fGeo = new THREE.ConeGeometry(U * 0.3, U * 0.7, 6); fGeo.translate(0, U * 0.35, 0);
    this.fireMesh = new THREE.InstancedMesh(fGeo, new THREE.MeshBasicNodeMaterial({ color: 0xff7a26 }), MAX_FIRES);
    this.fireMesh.count = 0; this.fireMesh.frustumCulled = false; this.group.add(this.fireMesh);

    this.buildPeople();
  }

  // ── 原始人モデル (共有ジオメトリ+材質・肌色3種) ──
  private buildPeople() {
    const U = this.U;
    const H = U * 1.35;                 // 身長 (少し高く=スラッと)
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
      // 脚 (股ピボット)
      const mkLeg = (x: number) => {
        const p = new THREE.Group(); p.position.set(x, hipY, 0);
        p.add(new THREE.Mesh(legGeo, skin));
        const foot = new THREE.Mesh(footGeo, skin); p.add(foot);
        g.add(p); return p;
      };
      const legL = mkLeg(-H * 0.045), legR2 = mkLeg(H * 0.045);
      // 胴 (体幹=上下ボブ用に body ラッパ)
      const body = new THREE.Group();
      const torso = new THREE.Mesh(torsoGeo, skin); torso.position.y = hipY + torsoH * 0.5; body.add(torso);
      const cloth = new THREE.Mesh(clothGeo, clothMat); cloth.position.y = hipY + H * 0.02; body.add(cloth);
      const head = new THREE.Mesh(headGeo, skin); head.position.y = hipY + torsoH + H * 0.1; body.add(head);
      const hair = new THREE.Mesh(hairGeo, hairMat); hair.position.y = hipY + torsoH + H * 0.12; body.add(hair);
      // 腕 (肩ピボット)
      const mkArm = (x: number) => {
        const p = new THREE.Group(); p.position.set(x, hipY + torsoH * 0.92, 0);
        p.add(new THREE.Mesh(armGeo, skin)); body.add(p); return p;
      };
      const armL = mkArm(-H * 0.12), armR3 = mkArm(H * 0.12);
      g.add(body);
      g.visible = false;
      this.group.add(g);
      this.pool.push({ group: g, legL, legR: legR2, armL, armR: armR3, body, phase: i * 1.7, baseY: 0 });
    }
  }

  seed(u: number, v: number) { this.bands.push({ u, v, wander: 0 }); }

  clear() {
    this.villages = []; this.bands = [];
    this.hutMesh.count = 0; this.fireMesh.count = 0;
    for (const p of this.pool) p.group.visible = false;
  }

  private pos(u: number, v: number, out: THREE.Vector3) {
    const W = this.sensor.worldW;
    out.set(u * W - W / 2, this.sensor.heightUV(u, v) * this.sensor.vertExag, W / 2 - v * W);
  }

  private flat(u: number, v: number): number {
    const W = this.sensor.worldW;
    const d = 3 / W;
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
    const lowland = 1 - en * 0.7;
    return flat * (0.55 + 0.45 * waterScore) * (0.45 + 0.55 * lowland);
  }

  update(dt: number) {
    this.t += dt; this.dt = dt;
    const stepUv = 0.06 * dt;
    const probe = 0.012; void probe;

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
      band.wander += dt;
      const du = band.tu - band.u, dv = (band.tv ?? band.v) - band.v;
      const dist = Math.hypot(du, dv);
      if (dist < stepUv * 1.5 || band.wander > 14) {
        this.villages.push({ u: band.u, v: band.v, pop: 4, huts: 1, age: 0 });
        this.bands.splice(b, 1);
        continue;
      }
      band.u += (du / dist) * stepUv;
      band.v += (dv / dist) * stepUv;
    }

    for (const vg of this.villages) {
      vg.age += dt;
      const s = this.suitability(vg.u, vg.v);
      if (s > 0.4) vg.pop += dt * s * 2.2;
      else vg.pop = Math.max(1, vg.pop - dt * 0.4);
      vg.pop = Math.min(vg.pop, 140);
      vg.huts = Math.min(12, Math.max(1, Math.floor(vg.pop / 6) + 1));
    }

    this.render();
  }

  private render() {
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

    // ── 人 (プールから割当・歩行アニメ) ──
    let pi = 0;
    const usePerson = (u: number, v: number, facing: number, speed: number) => {
      if (pi >= this.pool.length) return;
      const P = this.pool[pi++];
      P.group.visible = true;
      this.pos(u, v, P.group.position);
      P.group.rotation.y = facing;
      P.phase += this.dt * speed;
      const sw = Math.sin(P.phase) * 0.6;
      P.legL.rotation.x = sw; P.legR.rotation.x = -sw;
      P.armL.rotation.x = -sw * 0.7; P.armR.rotation.x = sw * 0.7;
      P.body.position.y = Math.abs(Math.cos(P.phase)) * U * 0.03; // 上下ボブ
    };
    // band=移動中(速く歩く・進行方向を向く)
    for (const band of this.bands) {
      const facing = Math.atan2((band.tu ?? band.u) - band.u, -(((band.tv ?? band.v) - band.v)));
      for (let k = 0; k < 3; k++) {
        const off = (k - 1) * U * 0.35;
        usePerson(band.u + (Math.cos(facing) * off) / W, band.v + (Math.sin(facing) * off) / W, facing, 9);
      }
    }
    // village=焚き火の周りをゆっくり歩く
    for (const vg of this.villages) {
      const nP = Math.min(6, Math.max(2, Math.ceil(vg.pop / 14)));
      for (let k = 0; k < nP; k++) {
        const ang = this.t * (0.35 + (k % 4) * 0.08) + k * 1.7;
        const rad = U * (1.4 + 1.3 * ((k % 3) / 3));
        usePerson(vg.u + (Math.cos(ang) * rad) / W, vg.v + (Math.sin(ang) * rad) / W, ang + Math.PI / 2, 5);
      }
    }
    for (let k = pi; k < this.pool.length; k++) this.pool[k].group.visible = false;
  }

  stats() {
    let pop = 0;
    for (const v of this.villages) pop += v.pop;
    return { villages: this.villages.length, pop: Math.round(pop), bands: this.bands.length };
  }
}
