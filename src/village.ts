// terra-touch: 創世 — 村システム (G0.1 見た目刷新)。
// 神(=地形/水の変形)に対して住民は自律反応。個体は管理せず「村単位」で抽象化。
// 置いた瞬間から原始人(人)が見え、適地へ歩いて定住→焚き火を囲む家(壁+屋根)の村に育つ。

import * as THREE from 'three/webgpu';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';

export interface WorldSensor {
  worldW: number;
  vertExag: number;
  elevMin: number; // マップ標高レンジ (低地選好に使う)
  elevMax: number;
  heightUV(u: number, v: number): number; // 生標高(m)
  waterUV(u: number, v: number): number;   // 水深(m)
  wetUV(u: number, v: number): number;     // 湿り 0..1
}

interface Village { u: number; v: number; pop: number; huts: number; age: number; }
interface Band { u: number; v: number; wander: number; tu?: number; tv?: number; }

const MAX_HUTS = 1500;
const MAX_PEOPLE = 500;
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
  private personMesh: THREE.InstancedMesh;
  private fireMesh: THREE.InstancedMesh;
  private dummy = new THREE.Object3D();
  private U: number;      // 基本単位長 (worldW依存)
  private t = 0;

  constructor(sensor: WorldSensor) {
    this.sensor = sensor;
    const W = sensor.worldW;
    const U = W * 0.004; // 見やすいアイコン尺度
    this.U = U;

    // 小屋 = 壁(円柱)+屋根(円錐)。頂点色で2トーン。
    const wall = new THREE.CylinderGeometry(U * 0.55, U * 0.7, U * 0.7, 7); wall.translate(0, U * 0.35, 0);
    const roof = new THREE.ConeGeometry(U * 0.95, U * 0.85, 7); roof.translate(0, U * 0.7 + U * 0.42, 0);
    const hutGeo = mergeGeometries([paint(wall, 0xb39e7d), paint(roof, 0x6f4a2e)])!;
    const hutMat = new THREE.MeshStandardNodeMaterial({ vertexColors: true, roughness: 0.9 });
    this.hutMesh = new THREE.InstancedMesh(hutGeo, hutMat, MAX_HUTS);
    this.hutMesh.count = 0; this.hutMesh.frustumCulled = false;
    this.group.add(this.hutMesh);

    // 人 = カプセル(明るいタン=地面に映える)
    const pGeo = new THREE.CapsuleGeometry(U * 0.22, U * 0.5, 3, 6); pGeo.translate(0, U * 0.45, 0);
    const pMat = new THREE.MeshStandardNodeMaterial({ color: 0xd8c39c, roughness: 1 });
    this.personMesh = new THREE.InstancedMesh(pGeo, pMat, MAX_PEOPLE);
    this.personMesh.count = 0; this.personMesh.frustumCulled = false;
    this.group.add(this.personMesh);

    // 焚き火 = 常に明るいオレンジ(村の中心マーカー・生活感)
    const fGeo = new THREE.ConeGeometry(U * 0.3, U * 0.7, 6); fGeo.translate(0, U * 0.35, 0);
    const fMat = new THREE.MeshBasicNodeMaterial({ color: 0xff7a26 });
    this.fireMesh = new THREE.InstancedMesh(fGeo, fMat, MAX_FIRES);
    this.fireMesh.count = 0; this.fireMesh.frustumCulled = false;
    this.group.add(this.fireMesh);
  }

  seed(u: number, v: number) { this.bands.push({ u, v, wander: 0 }); }

  clear() {
    this.villages = []; this.bands = [];
    this.hutMesh.count = 0; this.personMesh.count = 0; this.fireMesh.count = 0;
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
    // 低地選好: settlers は谷/平野へ下る (山頂の局所平坦の罠を避ける)
    const range = Math.max(1, this.sensor.elevMax - this.sensor.elevMin);
    const en = Math.min(1, Math.max(0, (this.sensor.heightUV(u, v) - this.sensor.elevMin) / range));
    const lowland = 1 - en * 0.7;
    return flat * (0.55 + 0.45 * waterScore) * (0.45 + 0.55 * lowland);
  }

  update(dt: number) {
    this.t += dt;
    const stepUv = 0.06 * dt; // 移動速度 (uv/s)
    const probe = 0.012;      // 近傍探索半径 (~勾配を感じる距離)

    void probe;
    // ── band: 置いた時に周囲を広くスキャンして最良の適地を目標化 → 直進して定住 ──
    for (let b = this.bands.length - 1; b >= 0; b--) {
      const band = this.bands[b];
      if (band.tu === undefined) {
        // 半径0.25uv 内を同心リングでスキャンし最良点を目標に
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

    // ── village: 人口増減 + 小屋 ──
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

    // 小屋 (村ごとに黄金角螺旋・コンパクト)
    let hi = 0;
    for (const vg of this.villages) {
      for (let k = 0; k < vg.huts && hi < MAX_HUTS; k++) {
        const ang = k * 2.399963;
        const rad = U * 1.3 * Math.sqrt(k);
        const u = vg.u + (Math.cos(ang) * rad) / W, v = vg.v + (Math.sin(ang) * rad) / W;
        this.pos(u, v, d.position);
        d.rotation.set(0, ang * 2.1, 0);
        d.scale.setScalar(0.8 + 0.25 * ((k * 7) % 5) / 5);
        d.updateMatrix();
        this.hutMesh.setMatrixAt(hi++, d.matrix);
      }
    }
    this.hutMesh.count = hi; this.hutMesh.instanceMatrix.needsUpdate = true;

    // 焚き火 (村中心・ゆらぎ)
    let fi = 0;
    for (const vg of this.villages) {
      if (fi >= MAX_FIRES) break;
      this.pos(vg.u, vg.v, d.position);
      d.rotation.set(0, 0, 0);
      d.scale.set(1, 0.85 + 0.25 * Math.sin(this.t * 6 + fi), 1);
      d.updateMatrix();
      this.fireMesh.setMatrixAt(fi++, d.matrix);
    }
    this.fireMesh.count = fi; this.fireMesh.instanceMatrix.needsUpdate = true;

    // 人 (band=移動中3人 / village=人口に応じ数人・焚き火の周りをうろつく)
    let pi = 0;
    const placePerson = (u: number, v: number, ang: number) => {
      if (pi >= MAX_PEOPLE) return;
      this.pos(u, v, d.position);
      d.rotation.set(0, -ang, 0); d.scale.setScalar(1);
      d.updateMatrix();
      this.personMesh.setMatrixAt(pi++, d.matrix);
    };
    for (const band of this.bands) {
      for (let k = 0; k < 3; k++) {
        const ang = this.t * 1.5 + k * 2.1;
        placePerson(band.u + (Math.cos(ang) * U * 0.5) / W, band.v + (Math.sin(ang) * U * 0.5) / W, ang);
      }
    }
    for (const vg of this.villages) {
      const nP = Math.min(7, Math.max(2, Math.ceil(vg.pop / 12)));
      for (let k = 0; k < nP; k++) {
        const ang = this.t * (0.7 + (k % 4) * 0.12) + k * 1.7;
        const rad = U * (1.6 + 1.4 * ((k % 3) / 3));
        placePerson(vg.u + (Math.cos(ang) * rad) / W, vg.v + (Math.sin(ang) * rad) / W, ang);
      }
    }
    this.personMesh.count = pi; this.personMesh.instanceMatrix.needsUpdate = true;
  }

  stats() {
    let pop = 0;
    for (const v of this.villages) pop += v.pop;
    return { villages: this.villages.length, pop: Math.round(pop), bands: this.bands.length };
  }
}
