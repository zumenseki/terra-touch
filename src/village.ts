// terra-touch: 創世 — 村システム (N1: 個体台帳 LifeAgent + クロック + 加齢/寿命/死)。
// 人口の真実 = 村ごとの離散個体台帳 agents[]。体(Person)は表示cap内のレンタル資源。
// 生命/経済 = rngSim(cap非依存)、表示サンプル = rngView。→ maxPeople 18/44 で人口動態がbit一致。
// 詳細裏設定: docs/SPEC-life-sim.md。

import * as THREE from 'three/webgpu';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { mulberry32 } from './rng';

export interface WorldSensor {
  worldW: number;
  vertExag: number;
  elevMin: number;
  elevMax: number;
  heightUV(u: number, v: number): number;
  waterUV(u: number, v: number): number;
  wetUV(u: number, v: number): number;
}

// ── 生命シム定数 (SPEC-life-sim) ──
const YEAR_SEC = 20;               // 1 gameYear = 実 20 秒
const SIM_TICK = 0.5;              // 実秒: 生命シムの低頻度 tick 間隔
const DT_Y = SIM_TICK / YEAR_SEC;  // 0.025 gameYear / tick
const ADULT_AGE = 15, ELDER_AGE = 50;
const FERT_MIN = 15, FERT_MAX = 40;
const SPAN_MIN = 45, SPAN_RANGE = 25; // 寿命 = 45 + rngSim*25 (45-70)
const PREG_CD = 1.75;              // 受胎クールダウン(gameYear)
const MAX_AGE = 70;
const FERT_BASE = 0.50;
const LINE_MUT = 0.06;             // 出産時に新家系となる確率
const VILLAGE_MIN_POP = 2;
const MAX_AGENTS = 1024;           // 村あたり個体上限(暴走防止)
const DEATH_FADE = 2.0;            // 死亡フェード(実秒)
const INFANT_HIDE = 3;             // age<3 は非表示

function hazardAt(ay: number): number {
  if (ay <= 0) return 0.12;
  if (ay <= 4) return 0.03;
  if (ay <= 14) return 0.008;
  return 0.006;
}

// 個体台帳。1体≈48B。同一性=id。
interface LifeAgent {
  id: number;
  age: number;   // gameYear (float)
  sex: 0 | 1;    // 0=F, 1=M
  span: number;  // 誕生時確定寿命 (gameYear)
  preg: number;  // 受胎CD残 (0=可)
  line: number;  // 母系家系ID
  body: number;  // pool 添字 (-1=非表示)
}

interface Village {
  u: number; v: number; huts: number; age: number;
  agents: LifeAgent[]; birthAcc: number;
  kids: number; adultsF: number; adultsM: number; elders: number;
}
interface Band { u: number; v: number; wander: number; tu?: number; tv?: number; agents: LifeAgent[]; }

interface Person {
  group: THREE.Group;
  legL: THREE.Group; legR: THREE.Group;
  armL: THREE.Group; armR: THREE.Group;
  body: THREE.Object3D;
  idx: number;               // pool 内の固定添字
  agent: LifeAgent | null;   // この体が表す個体
  active: boolean;
  dying: boolean;            // 死亡フェード中
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
  private dyingList: { p: Person; t: number }[] = [];
  private maxPeople: number;
  private stride: number;
  private rngSim: () => number;   // 生命/経済/出産/死 (cap非依存)
  private rngView: () => number;  // 表示サンプル(徘徊/体割当)

  // クロック & カウンタ
  private simAcc = 0;
  private ticksDone = 0;
  private nextId = 1;
  private nextLine = 1;
  private totalFounders = 0;
  private totalBirths = 0;
  private totalDeaths = 0;
  private recordDeaths = false;
  private deathLog: number[] = [];

  constructor(sensor: WorldSensor, opts?: { maxPeople?: number; seed?: number }) {
    this.sensor = sensor;
    const W = sensor.worldW;
    const U = W * 0.004;
    this.U = U;
    this.stride = U * 0.42;
    this.maxPeople = opts?.maxPeople ?? 40;
    const seed = opts?.seed ?? Math.floor(Math.random() * 0xffffffff);
    this.rngSim = mulberry32(seed);
    this.rngView = mulberry32((seed ^ 0x9e3779b9) >>> 0);

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
        idx: i, agent: null,
        active: false, dying: false, homeRef: null, homeKind: null,
        u: 0, v: 0, tu: 0, tv: 0, state: 'walk', timer: 0, speed: 0.007, phase: i * 1.7, facing: 0, fx: 0, fy: 0,
      });
    }
  }

  // ── 個体生成 ──
  private newFounder(age: number, sex: 0 | 1): LifeAgent {
    this.totalFounders++;
    return { id: this.nextId++, age, sex, span: SPAN_MIN + this.rngSim() * SPAN_RANGE, preg: 0, line: this.nextLine++, body: -1 };
  }

  seed(u: number, v: number) {
    const agents: LifeAgent[] = [];
    for (let i = 0; i < 4; i++) agents.push(this.newFounder(18 + this.rngSim() * 4, (i % 2 === 0 ? 0 : 1)));
    this.bands.push({ u, v, wander: 0, agents });
  }

  // 決定論再シード(検証・A/B比較用)。生命/表示の両乱数列を seed で固定。
  reseed(seed: number) {
    this.rngSim = mulberry32(seed >>> 0);
    this.rngView = mulberry32((seed ^ 0x9e3779b9) >>> 0);
  }

  clock(): number { return this.ticksDone * DT_Y + this.simAcc / YEAR_SEC; }

  clear() {
    this.villages = []; this.bands = []; this.dyingList = [];
    this.hutMesh.count = 0; this.fireMesh.count = 0;
    for (const p of this.pool) { p.active = false; p.dying = false; p.homeRef = null; p.homeKind = null; p.agent = null; p.group.visible = false; p.group.scale.setScalar(1); }
    this.totalFounders = 0; this.totalBirths = 0; this.totalDeaths = 0;
    this.nextId = 1; this.nextLine = 1; this.simAcc = 0; this.ticksDone = 0;
    this.recordDeaths = false; this.deathLog = [];
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

  // ── メイン更新: 毎フレーム(band移動/歩行/描画) + 低頻度tick(生命シム) ──
  update(dt: number) {
    this.t += dt;
    this.updateBands(dt);

    this.simAcc += dt;
    let guard = 0;
    while (this.simAcc >= SIM_TICK && guard++ < 4096) { this.simAcc -= SIM_TICK; this.simTick(); this.ticksDone++; }

    this.updateDying(dt);
    for (const p of this.pool) if (p.active && !p.dying && p.agent) this.updatePerson(p, dt);
    this.renderStatic();
  }

  // ── band: 適地スキャン→直進→定住(毎フレーム・決定論=rng不使用) ──
  private updateBands(dt: number) {
    const stepUv = 0.06 * dt;
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
      const du = (band.tu ?? band.u) - band.u, dv = (band.tv ?? band.v) - band.v;
      const dist = Math.hypot(du, dv);
      if (dist < stepUv * 1.5 || band.wander > 14) {
        const vg: Village = { u: band.u, v: band.v, huts: 1, age: 0, agents: band.agents, birthAcc: 0, kids: 0, adultsF: 0, adultsM: 0, elders: 0 };
        this.recomputeDemo(vg);
        vg.huts = Math.min(12, Math.max(1, Math.floor(vg.agents.length / 6) + 1));
        this.villages.push(vg);
        // band に紐付いた体を村へ移管
        for (const a of band.agents) if (a.body >= 0) { const p = this.pool[a.body]; if (p) { p.homeRef = vg; p.homeKind = 'village'; } }
        this.bands.splice(b, 1);
        continue;
      }
      band.u += (du / dist) * stepUv; band.v += (dv / dist) * stepUv;
    }
  }

  // ── 生命シム 1 tick (dtY gameYear) ──
  private simTick() {
    for (const vg of this.villages) {
      vg.agents = this.ageAndDie(vg.agents);
      this.birthsIn(vg);
      this.recomputeDemo(vg);
      vg.huts = Math.min(12, Math.max(1, Math.floor(vg.agents.length / 6) + 1));
      vg.age += DT_Y;
    }
    for (const bd of this.bands) bd.agents = this.ageAndDie(bd.agents);
    // 消滅村/空band を除去
    for (let i = this.villages.length - 1; i >= 0; i--) {
      if (this.villages[i].agents.length < VILLAGE_MIN_POP) { this.dissolveVillage(this.villages[i]); this.villages.splice(i, 1); }
    }
    for (let i = this.bands.length - 1; i >= 0; i--) if (this.bands[i].agents.length === 0) this.bands.splice(i, 1);
    this.reconcileBodies();
  }

  // 加齢 + 死(自然=age≥span / 運死=誕生日にhazardロール)。生存者を返す。
  private ageAndDie(agents: LifeAgent[]): LifeAgent[] {
    const survivors: LifeAgent[] = [];
    for (const a of agents) {
      const prevY = Math.floor(a.age);
      a.age += DT_Y;
      if (a.preg > 0) a.preg = Math.max(0, a.preg - DT_Y);
      const ageY = Math.floor(a.age);
      let dead = false;
      // 生き抜いた年(prevY)の死亡率でロール = 乳児(prevY=0)は q0=0.12 が効く
      if (ageY > prevY && this.rngSim() < hazardAt(Math.min(prevY, MAX_AGE))) dead = true;
      if (!dead && a.age >= a.span) dead = true;
      if (dead) { this.totalDeaths++; if (this.recordDeaths) this.deathLog.push(a.age); if (a.body >= 0) this.killBody(a); }
      else survivors.push(a);
    }
    return survivors;
  }

  // 出産(N1簡易版: B = fertBase * eligibleF * E_pair・環境ゲート無し)
  private birthsIn(vg: Village) {
    let m15 = 0, f15 = 0, eligF = 0;
    for (const a of vg.agents) {
      if (a.age >= ADULT_AGE) { if (a.sex === 1) m15++; else f15++; }
      if (a.sex === 0 && a.age >= FERT_MIN && a.age <= FERT_MAX && a.preg <= 0) eligF++;
    }
    const ePair = (m15 + f15 > 0) ? 2 * Math.min(m15, f15) / (m15 + f15) : 0;
    vg.birthAcc += FERT_BASE * eligF * ePair * DT_Y;
    const diff = m15 - f15;
    while (vg.birthAcc >= 1 && vg.agents.length < MAX_AGENTS) {
      vg.birthAcc -= 1;
      let sex: 0 | 1;
      if (Math.abs(diff) > 3) { const minority: 0 | 1 = diff > 0 ? 0 : 1; sex = this.rngSim() < 0.7 ? minority : (minority === 0 ? 1 : 0); }
      else sex = this.rngSim() < 0.5 ? 0 : 1;
      const span = SPAN_MIN + this.rngSim() * SPAN_RANGE;
      let motherLine = -1;
      for (const a of vg.agents) { if (a.sex === 0 && a.age >= FERT_MIN && a.age <= FERT_MAX && a.preg <= 0) { a.preg = PREG_CD; motherLine = a.line; break; } }
      const line = (this.rngSim() < LINE_MUT || motherLine < 0) ? this.nextLine++ : motherLine;
      vg.agents.push({ id: this.nextId++, age: 0, sex, span, preg: 0, line, body: -1 });
      this.totalBirths++;
    }
  }

  private recomputeDemo(vg: Village) {
    let k = 0, af = 0, am = 0, e = 0;
    for (const a of vg.agents) {
      if (a.age < ADULT_AGE) k++;
      else if (a.age >= ELDER_AGE) e++;
      else if (a.sex === 0) af++; else am++;
    }
    vg.kids = k; vg.adultsF = af; vg.adultsM = am; vg.elders = e;
  }

  private dissolveVillage(vg: Village) {
    for (const a of vg.agents) { this.totalDeaths++; if (this.recordDeaths) this.deathLog.push(a.age); if (a.body >= 0) this.killBody(a); }
    vg.agents = [];
  }

  // ── 体レンタル ──
  private assignBody(ref: Village | Band, kind: 'village' | 'band', a: LifeAgent): boolean {
    const p = this.pool.find((x) => !x.active);
    if (!p) return false;
    p.active = true; p.dying = false; p.homeRef = ref; p.homeKind = kind; p.agent = a;
    a.body = p.idx;
    p.u = ref.u; p.v = ref.v; p.tu = ref.u; p.tv = ref.v; p.state = 'pause'; p.timer = this.rngView() * 1.5;
    p.speed = 0.005 + this.rngView() * 0.006;
    p.fx = (this.rngView() - 0.5) * this.U * 0.7 / this.sensor.worldW;
    p.fy = (this.rngView() - 0.5) * this.U * 0.7 / this.sensor.worldW;
    p.group.visible = true;
    return true;
  }
  private releaseBody(a: LifeAgent) {
    if (a.body < 0) return;
    const p = this.pool[a.body];
    if (p) { p.active = false; p.dying = false; p.agent = null; p.homeRef = null; p.homeKind = null; p.group.visible = false; p.group.scale.setScalar(1); }
    a.body = -1;
  }
  // 死亡した個体の体は 2 秒かけてフェード(rng不使用)
  private killBody(a: LifeAgent) {
    if (a.body < 0) return;
    const p = this.pool[a.body];
    if (p) { p.agent = null; p.dying = true; p.homeRef = null; p.homeKind = null; this.dyingList.push({ p, t: DEATH_FADE }); }
    a.body = -1;
  }
  private updateDying(dt: number) {
    for (let i = this.dyingList.length - 1; i >= 0; i--) {
      const d = this.dyingList[i]; d.t -= dt;
      const s = Math.max(0, d.t / DEATH_FADE);
      d.p.group.scale.setScalar(s);
      if (d.t <= 0) {
        d.p.active = false; d.p.dying = false; d.p.group.visible = false; d.p.group.scale.setScalar(1);
        this.dyingList.splice(i, 1);
      }
    }
  }

  private reconcileBodies() {
    const groups = this.villages.length + this.bands.length;
    const capPerV = Math.max(2, Math.floor(this.maxPeople / Math.max(1, groups)));
    for (const vg of this.villages) this.reconcileGroup(vg, vg.agents, 'village', Math.min(capPerV, vg.agents.length));
    for (const bd of this.bands) this.reconcileGroup(bd, bd.agents, 'band', Math.min(capPerV, bd.agents.length));
  }
  // 表示体数を want に合わせる。優先=大人>子供(最大40%)。幼児(age<3)は非表示。
  private reconcileGroup(ref: Village | Band, agents: LifeAgent[], kind: 'village' | 'band', want: number) {
    let bodied = 0;
    for (const a of agents) if (a.body >= 0) bodied++;
    // trim
    for (let i = agents.length - 1; i >= 0 && bodied > want; i--) {
      const a = agents[i];
      if (a.body >= 0) { this.releaseBody(a); bodied--; }
    }
    if (bodied >= want) return;
    // 大人優先
    for (const a of agents) { if (bodied >= want) break; if (a.body < 0 && a.age >= ADULT_AGE) if (this.assignBody(ref, kind, a)) bodied++; }
    // 子供(3-15歳)を最大 40%
    const kidCap = Math.floor(want * 0.4);
    let kidsShown = 0; for (const a of agents) if (a.body >= 0 && a.age < ADULT_AGE) kidsShown++;
    for (const a of agents) {
      if (bodied >= want || kidsShown >= kidCap) break;
      if (a.body < 0 && a.age >= INFANT_HIDE && a.age < ADULT_AGE) if (this.assignBody(ref, kind, a)) { bodied++; kidsShown++; }
    }
  }

  private wanderTarget(p: Person, vg: Village) {
    const W = this.sensor.worldW;
    const r = (this.U * (0.8 + this.rngView() * 3)) / W;
    const a = this.rngView() * Math.PI * 2;
    p.tu = Math.min(0.98, Math.max(0.02, vg.u + Math.cos(a) * r));
    p.tv = Math.min(0.98, Math.max(0.02, vg.v + Math.sin(a) * r));
  }

  private updatePerson(p: Person, dt: number) {
    const W = this.sensor.worldW, U = this.U, d = this.dummy;
    let moved = 0;
    if (p.homeKind === 'band' && p.homeRef) {
      const band = p.homeRef as Band;
      p.tu = band.u + p.fx; p.tv = band.v + p.fy;
      moved = this.stepToward(p, p.speed * 1.6 * dt);
    } else if (p.homeRef) {
      const vg = p.homeRef as Village;
      if (p.state === 'pause') {
        p.timer -= dt;
        if (p.timer <= 0) { this.wanderTarget(p, vg); p.state = 'walk'; }
      } else {
        moved = this.stepToward(p, p.speed * dt);
        const du = p.tu - p.u, dv = p.tv - p.v;
        if (Math.hypot(du, dv) < p.speed * dt * 1.2) { p.state = 'pause'; p.timer = 0.6 + this.rngView() * 3; }
      }
    }
    // 歩行アニメ
    p.phase += (moved * W) / this.stride;
    const sw = Math.sin(p.phase) * 0.6;
    p.legL.rotation.x = sw; p.legR.rotation.x = -sw;
    p.armL.rotation.x = -sw * 0.7; p.armR.rotation.x = sw * 0.7;
    p.body.position.y = Math.abs(Math.cos(p.phase)) * U * 0.03;
    // 配置 + 年齢スケール(子は小さい)
    this.pos(p.u, p.v, d.position);
    p.group.position.copy(d.position);
    p.group.rotation.y = p.facing;
    if (p.agent) p.group.scale.setScalar(0.4 + 0.6 * Math.min(p.agent.age, 15) / 15);
  }

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
    let pop = 0, kids = 0, adults = 0, elders = 0;
    for (const v of this.villages) { pop += v.agents.length; kids += v.kids; adults += v.adultsF + v.adultsM; elders += v.elders; }
    for (const b of this.bands) pop += b.agents.length;
    return {
      villages: this.villages.length, pop, bands: this.bands.length,
      year: Math.round(this.clock() * 1000) / 1000,
      births: this.totalBirths, deaths: this.totalDeaths,
      kids, adults, elders,
    };
  }

  // ── ヘッドレス検証用デバッグAPI ──
  _live(): number { let n = 0; for (const v of this.villages) n += v.agents.length; for (const b of this.bands) n += b.agents.length; return n; }
  _counts() { return { founders: this.totalFounders, births: this.totalBirths, deaths: this.totalDeaths, live: this._live() }; }
  _injectVillage(u: number, v: number, specs: { age: number; sex: 0 | 1 }[]) {
    const agents = specs.map((s) => this.newFounder(s.age, s.sex));
    const vg: Village = { u, v, huts: 1, age: 0, agents, birthAcc: 0, kids: 0, adultsF: 0, adultsM: 0, elders: 0 };
    this.recomputeDemo(vg); this.villages.push(vg); return vg;
  }
  _startDeathLog() { this.recordDeaths = true; this.deathLog = []; }
  _deathAges(): number[] { return this.deathLog; }
  _ages(): number[] { const out: number[] = []; for (const v of this.villages) for (const a of v.agents) out.push(a.age); for (const b of this.bands) for (const a of b.agents) out.push(a.age); return out; }
  _demoOK(): boolean {
    for (const v of this.villages) {
      let k = 0, af = 0, am = 0, e = 0;
      for (const a of v.agents) { if (a.age < ADULT_AGE) k++; else if (a.age >= ELDER_AGE) e++; else if (a.sex === 0) af++; else am++; }
      if (k !== v.kids || af !== v.adultsF || am !== v.adultsM || e !== v.elders) return false;
    }
    return true;
  }
  _ageRangeOK(): boolean { for (const v of this.villages) for (const a of v.agents) if (!(a.age >= 0 && a.age < MAX_AGE + DT_Y)) return false; return true; }
}
