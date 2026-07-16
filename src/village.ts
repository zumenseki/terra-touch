// terra-touch: 創世 — 村システム (N1: 個体台帳 LifeAgent + クロック + 加齢/寿命/死)。
// 人口の真実 = 村ごとの離散個体台帳 agents[]。体(Person)は表示cap内のレンタル資源。
// 生命/経済 = rngSim(cap非依存)、表示サンプル = rngView。→ maxPeople 18/44 で人口動態がbit一致。
// 詳細裏設定: docs/SPEC-life-sim.md。

import * as THREE from 'three/webgpu';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { mulberry32 } from './rng';
import { EcologySystem } from './ecology';

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

// ── N2 食料経済 (SPEC-life-sim §3,4) ──
// GATHER_BASE: 実地形較正値。富士は乾燥急峻で suitability が最大0.5/平均0.25。
// SPEC理論値1.6(損益分岐s=0.625)は実地形で全村即絶滅→3.2に較正(損益分岐s≈0.31・
// 最良flat地~40人上限)。水(川/雨)で suitability が上がるほど繁栄=設計意図どおり。
const GATHER_BASE = 3.2;           // fu / gatherer年 (× suitability)
const STORE_PER_HUT = 4.0;         // 家1軒あたり食料貯蔵上限 (fu)
const M_STARVE = 0.4;              // 飢餓死係数 / 年
const HUNGER_RATE = 3;             // hungerY += dtY*3*(1-nutrition)
const CONS_KID = 0.5, CONS_ADULT = 1.0, CONS_ELDER = 0.7; // 消費 fu/年
const LABOR_ELDER = 0.5;           // 老人の労働係数

// ── N4 建築 (SPEC-life-sim §5) ──
// 段階別コスト: テント=丸太不要(枝と皮)／藁小屋=4／丸太家=8。
// 🔴 新村は huts=0 から始めて必ず「建てる過程」を見せる(実機FB「最初から家がある」)。
// テントが丸太不要なのはそのための必然: 木が無い土地でも最初の一軒は張れる=詰まない。
const LOGS_BY_STAGE = [0, 4, 8];         // 1軒に必要な丸太(段階別)
const WORK_BY_STAGE = [0.15, 0.35, 0.5]; // 建築 worker年 / 軒(段階別)
const LOGS_PER_TREE = 2;           // 木1本=2丸太
const LUMBER_RATE = 24;            // 伐採+運搬 logs / worker年
const TREE_REGROW = 4;             // 木の再生 本/年
const TREES_CAP_BASE = 24;         // 木の上限 = 24 * vegScore
// 村の見た目/建築段階: テント(<8人) → 藁小屋(8-20) → 丸太家(20+)
const stageOf = (pop: number) => (pop < 8 ? 0 : pop < 20 ? 1 : 2);

function hazardAt(ay: number): number {
  if (ay <= 0) return 0.12;
  if (ay <= 4) return 0.03;
  if (ay <= 14) return 0.008;
  return 0.006;
}
const clamp = (x: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, x));

// ── N7 移住/分村/気候/respawn (SPEC-life-sim §8) ──
const MIG_THRESHOLD = 1.5;   // hungerY がこれ以上で移住
const MIG_COOLDOWN = 3;      // 移住後クールダウン(gameYear)
const FISSION_POP = 45, FISSION_SURPLUS = 2; // 分村条件
const RESPAWN_POP = 8, RESPAWN_SEC = 45;     // 全滅回避リスポーン
const CLIMATE_AMP = 0.30, CLIMATE_PERIOD = 12; // 気候振動 ±30%・周期12年
const BAND_FOOD_INIT = 6;    // seed/respawn band の初期携行食料
// 気候 sineLUT (Math.sin は init のみ・tick 内は禁止=クロスプラットフォーム決定論)
const CLIMATE_LUT = new Float32Array(256);
for (let i = 0; i < 256; i++) CLIMATE_LUT[i] = Math.sin((i / 256) * Math.PI * 2);
const hash01 = (n: number) => ((n * 2654435761) >>> 0) / 4294967296;
// 家数(N2自動)。人口より先行させ E_house≈1 に保つ(鶏卵回避)。家の律速は N4 で実装。
const autoHuts = (pop: number) => Math.min(12, Math.max(2, Math.ceil(pop / 6) + 1));

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

// need/work = 着工時に確定(建築中に人口が段階を跨いでもコストが変わらない=木材保存則が壊れない)
interface BuildSite { logs: number; workY: number; stage: number; need: number; work: number; }
interface Village {
  u: number; v: number; huts: number; age: number; // huts = 建築済み軒数(N4=hutsBuilt)
  agents: LifeAgent[]; birthAcc: number;
  kids: number; adultsF: number; adultsM: number; elders: number;
  foodStock: number; hungerY: number; surplusY: number; deathAcc: number;
  trees: number; site: BuildSite | null; // N4 建築
  migCd: number; phi: number;            // N7 移住クールダウン(gameYear) / 気候位相
}
interface Band {
  u: number; v: number; wander: number; tu?: number; tv?: number; agents: LifeAgent[];
  food: number; migCd: number;           // N7 携行食料 / 移住由来クールダウン
  ox?: number; oy?: number;              // 移住元(移住bandはここから離れて再定住)
}

// N3: 体は部位別InstancedMesh(全人物で共有)。Person は論理状態+外見パラメータのみ。
interface Person {
  idx: number;               // pool 内=instance 添字(固定)
  agent: LifeAgent | null;   // この体が表す個体
  active: boolean;
  dying: boolean;            // 死亡フェード中
  fade: number;              // 表示スケール係数 (1=通常, 0=消滅)
  homeRef: Village | Band | null;
  homeKind: 'village' | 'band' | null;
  u: number; v: number; tu: number; tv: number;
  state: 'walk' | 'pause';
  timer: number; speed: number; phase: number; facing: number;
  fx: number; fy: number; // band 隊列オフセット
  // 外見(体を借りた時に agent から決定・見た目の個体差/性差)
  sex: 0 | 1; skinIdx: number; hairLong: boolean;
  heightScale: number; shoulderW: number; hipW: number; gait: number;
}

const MAX_HUTS = 1500;
const MAX_FIRES = 200;
const SMOKE_PER_FIRE = 3;

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
  private hutMeshes: THREE.InstancedMesh[] = []; // 3段: テント/藁小屋/丸太家
  private fireMesh: THREE.InstancedMesh;
  private smokeMesh: THREE.InstancedMesh;
  private dummy = new THREE.Object3D();
  private U: number;
  private t = 0;
  private pool: Person[] = [];
  private dyingList: { p: Person; t: number }[] = [];

  // N3: 部位別InstancedMesh(全人物共有・draw call ~10)
  // 膝肘2セグ: 腿/脛(+足)・上腕/前腕
  private partThighL!: THREE.InstancedMesh; private partThighR!: THREE.InstancedMesh;
  private partShinL!: THREE.InstancedMesh; private partShinR!: THREE.InstancedMesh;
  private partUpArmL!: THREE.InstancedMesh; private partUpArmR!: THREE.InstancedMesh;
  private partForeL!: THREE.InstancedMesh; private partForeR!: THREE.InstancedMesh;
  private thighLen = 0; private upArmLen = 0;
  private partTorso!: THREE.InstancedMesh; private partHead!: THREE.InstancedMesh;
  private partHairS!: THREE.InstancedMesh; private partHairL!: THREE.InstancedMesh;
  private partClothM!: THREE.InstancedMesh; private partClothF!: THREE.InstancedMesh;
  private skinParts: THREE.InstancedMesh[] = [];
  private allParts: THREE.InstancedMesh[] = [];
  private personsDirty = true;
  private hutSig = '';
  // 人体寸法(buildPeople で確定)
  private H = 0; private hipY = 0; private torsoH = 0;
  // 再利用行列
  private mRoot = new THREE.Matrix4(); private mLocal = new THREE.Matrix4(); private mOut = new THREE.Matrix4();
  private mLocal2 = new THREE.Matrix4();
  private qTmp = new THREE.Quaternion(); private eTmp = new THREE.Euler();
  private vTmp = new THREE.Vector3(); private vScl = new THREE.Vector3(); private vPos = new THREE.Vector3();
  private cTmp = new THREE.Color();
  private mZero = new THREE.Matrix4().makeScale(0, 0, 0);
  private skinCols = [0xb98a5e, 0xa9764a, 0xc79b6e];
  private maxPeople: number;
  private stride: number;
  private rngSim: () => number;   // 生命/経済/出産/死 (cap非依存)
  private rngView: () => number;  // 表示サンプル(徘徊/体割当)
  eco: EcologySystem;             // N6 生態(魚+陸獣)
  private ecoRefreshAcc = 0;

  // クロック & カウンタ
  private simAcc = 0;
  private ticksDone = 0;
  private nextId = 1;
  private nextLine = 1;
  private totalFounders = 0;
  private totalBirths = 0;
  private totalDeaths = 0;
  // N4 建築の保存則検証用
  private totalChopped = 0;      // 累計伐採丸太
  private totalTreesConsumed = 0;
  private hutsCompleted = 0;     // 建築で完成した軒数
  private totalLogsUsed = 0;     // 完成した家が消費した丸太(段階別コスト・保存則の右辺)
  private migrations = 0;        // N7 移住回数
  private respawnAcc = 0;        // N7 リスポーン蓄積(gameYear)
  atCapacity = false;            // MAX_AGENTS 到達(HUD表示用)
  private recordDeaths = false;
  private deathLog: number[] = [];
  // 検証用フラグ(通常は無効)
  private testHuts = -1;      // >=0 で huts 固定
  private testInfFood = false; // true で食料無限(飢餓なし・nutrition=1)

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

    // 家3段成長: テント(cone) → 藁小屋(壁+茅葺) → 丸太家(箱+切妻)。村の人口で段階。
    const mkTent = () => { const c = new THREE.ConeGeometry(U * 0.55, U * 0.95, 6); c.translate(0, U * 0.47, 0); return paint(c, 0xa89478); };
    const mkStraw = () => {
      const wall = new THREE.CylinderGeometry(U * 0.55, U * 0.7, U * 0.7, 7); wall.translate(0, U * 0.35, 0);
      const roof = new THREE.ConeGeometry(U * 0.95, U * 0.85, 7); roof.translate(0, U * 0.7 + U * 0.42, 0);
      return mergeGeometries([paint(wall, 0xc9b07a), paint(roof, 0x8a6a3a)])!;
    };
    const mkLog = () => {
      const body = new THREE.BoxGeometry(U * 1.15, U * 0.8, U * 0.95); body.translate(0, U * 0.4, 0);
      const roof = new THREE.ConeGeometry(U * 0.98, U * 0.55, 4); roof.rotateY(Math.PI / 4); roof.translate(0, U * 0.8 + U * 0.27, 0);
      return mergeGeometries([paint(body, 0x8a6a45), paint(roof, 0x5a4530)])!;
    };
    const hutMat = new THREE.MeshStandardNodeMaterial({ vertexColors: true, roughness: 0.9 });
    for (const g of [mkTent(), mkStraw(), mkLog()]) {
      const m = new THREE.InstancedMesh(g, hutMat, MAX_HUTS);
      m.count = 0; m.frustumCulled = false; this.group.add(m); this.hutMeshes.push(m);
    }

    const fGeo = new THREE.ConeGeometry(U * 0.3, U * 0.7, 6); fGeo.translate(0, U * 0.35, 0);
    this.fireMesh = new THREE.InstancedMesh(fGeo, new THREE.MeshBasicNodeMaterial({ color: 0xff7a26 }), MAX_FIRES);
    this.fireMesh.count = 0; this.fireMesh.frustumCulled = false; this.group.add(this.fireMesh);

    // 焚き火の煙(立ち昇る半透明パフ)。村に生活感。
    const smGeo = new THREE.SphereGeometry(U * 0.28, 6, 5);
    const smMat = new THREE.MeshBasicNodeMaterial({ color: 0x9a9488, transparent: true, opacity: 0.28, depthWrite: false });
    this.smokeMesh = new THREE.InstancedMesh(smGeo, smMat, MAX_FIRES * SMOKE_PER_FIRE);
    this.smokeMesh.count = 0; this.smokeMesh.frustumCulled = false; this.smokeMesh.renderOrder = 1; this.group.add(this.smokeMesh);

    this.buildPeople();

    this.eco = new EcologySystem(sensor, { fishCap: this.maxPeople * 2, animalCap: this.maxPeople });
    this.group.add(this.eco.group);
  }

  private buildPeople() {
    const U = this.U;
    const H = U * 1.35;
    const legLen = H * 0.48, legR = H * 0.042;
    const armLen = H * 0.40, armR = H * 0.032;
    const torsoH = H * 0.30;
    const hipY = legLen;
    this.H = H; this.hipY = hipY; this.torsoH = torsoH;

    // ── 部位ジオメトリ(各パーツは自身の pivot 原点で作る) ──
    // 膝肘2セグ: 腿(pivot=股)/脛+足(pivot=膝)・上腕(pivot=肩)/前腕(pivot=肘)
    const thighLen = legLen * 0.52, shinLen = legLen * 0.48;
    const upArmLen = armLen * 0.5, foreLen = armLen * 0.5;
    this.thighLen = thighLen; this.upArmLen = upArmLen;
    const thighGeo = new THREE.CapsuleGeometry(legR, thighLen - 2 * legR, 4, 7); thighGeo.translate(0, -thighLen / 2 + legR, 0);
    const shinR = legR * 0.85;
    const shinCap = new THREE.CapsuleGeometry(shinR, shinLen - 2 * shinR, 4, 7); shinCap.translate(0, -shinLen / 2 + shinR, 0);
    const footGeo = new THREE.BoxGeometry(legR * 2, legR * 1.1, legR * 2.6); footGeo.translate(0, -shinLen + legR * 0.5, legR * 0.6);
    const shinGeo = mergeGeometries([shinCap, footGeo])!;               // 脛+足(pivot=膝)
    const upArmGeo = new THREE.CapsuleGeometry(armR, upArmLen - 2 * armR, 4, 7); upArmGeo.translate(0, -upArmLen / 2 + armR, 0);
    const foreR = armR * 0.85;
    const foreGeo = new THREE.CapsuleGeometry(foreR, foreLen - 2 * foreR, 4, 7); foreGeo.translate(0, -foreLen / 2 + foreR, 0);
    const torsoGeo = new THREE.CapsuleGeometry(H * 0.095, torsoH - H * 0.095, 5, 9);
    const headGeo = new THREE.SphereGeometry(H * 0.088, 12, 10);
    const hairSGeo = new THREE.SphereGeometry(H * 0.096, 12, 8, 0, Math.PI * 2, 0, Math.PI * 0.62); // 短髪(浅い帽)
    const hairLGeo = new THREE.SphereGeometry(H * 0.10, 12, 10, 0, Math.PI * 2, 0, Math.PI * 0.95);  // 長髪(深い帽=後ろ髪)
    const clothMGeo = new THREE.CylinderGeometry(H * 0.11, H * 0.125, H * 0.14, 9);                  // 腰巻(男)
    const clothFGeo = new THREE.CylinderGeometry(H * 0.10, H * 0.17, H * 0.24, 10); clothFGeo.translate(0, -H * 0.05, 0); // スカート(女)

    const skinMat = new THREE.MeshStandardNodeMaterial({ roughness: 0.85 });   // 肌=instanceColor で個体差
    const hairMat = new THREE.MeshStandardNodeMaterial({ color: 0x201510, roughness: 1 });
    const clothMat = new THREE.MeshStandardNodeMaterial({ color: 0x6b4a2e, roughness: 1 });

    const N = this.maxPeople;
    const mk = (geo: THREE.BufferGeometry, mat: THREE.Material, skin: boolean): THREE.InstancedMesh => {
      const m = new THREE.InstancedMesh(geo, mat, N);
      m.frustumCulled = false; m.count = N;
      for (let i = 0; i < N; i++) m.setMatrixAt(i, this.mZero);
      m.instanceMatrix.needsUpdate = true;
      if (skin) { for (let i = 0; i < N; i++) m.setColorAt(i, this.cTmp.setHex(0xb98a5e)); if (m.instanceColor) m.instanceColor.needsUpdate = true; }
      this.group.add(m); this.allParts.push(m);
      return m;
    };
    this.partThighL = mk(thighGeo, skinMat, true); this.partThighR = mk(thighGeo, skinMat, true);
    this.partShinL = mk(shinGeo, skinMat, true); this.partShinR = mk(shinGeo, skinMat, true);
    this.partUpArmL = mk(upArmGeo, skinMat, true); this.partUpArmR = mk(upArmGeo, skinMat, true);
    this.partForeL = mk(foreGeo, skinMat, true); this.partForeR = mk(foreGeo, skinMat, true);
    this.partTorso = mk(torsoGeo, skinMat, true); this.partHead = mk(headGeo, skinMat, true);
    this.partHairS = mk(hairSGeo, hairMat, false); this.partHairL = mk(hairLGeo, hairMat, false);
    this.partClothM = mk(clothMGeo, clothMat, false); this.partClothF = mk(clothFGeo, clothMat, false);
    this.skinParts = [this.partThighL, this.partThighR, this.partShinL, this.partShinR,
      this.partUpArmL, this.partUpArmR, this.partForeL, this.partForeR, this.partTorso, this.partHead];

    for (let i = 0; i < N; i++) {
      this.pool.push({
        idx: i, agent: null,
        active: false, dying: false, fade: 1,
        homeRef: null, homeKind: null,
        u: 0, v: 0, tu: 0, tv: 0, state: 'walk', timer: 0, speed: 0.007, phase: i * 1.7, facing: 0, fx: 0, fy: 0,
        sex: 1, skinIdx: 0, hairLong: false, heightScale: 1, shoulderW: 1, hipW: 1, gait: 1,
      });
    }
  }

  // agent の同一性から決定論的に外見を導出(rng不使用=cap非依存・体を借り直しても不変)
  private setAppearance(p: Person, a: LifeAgent) {
    const h1 = (a.id * 2654435761) >>> 0, h2 = (a.id * 40503 + 12345) >>> 0, h3 = (a.id * 2246822519) >>> 0;
    p.sex = a.sex;
    p.skinIdx = h1 % 3;
    p.hairLong = a.sex === 0 ? (h2 % 4 !== 0) : (h2 % 5 === 0); // 女=長髪多め/男=たまに
    p.heightScale = 0.92 + (h3 % 100) / 100 * 0.16;             // 0.92-1.08
    p.shoulderW = a.sex === 1 ? 1.12 : 0.85;                    // 男=肩広
    p.hipW = a.sex === 0 ? 1.22 : 1.0;                          // 女=腰広
    p.gait = a.sex === 1 ? 1.15 : 0.85;                         // 男=大股/女=小股
    // 肌色を各 skin パーツへ
    const col = this.cTmp.setHex(this.skinCols[p.skinIdx]);
    for (const m of this.skinParts) { m.setColorAt(p.idx, col); if (m.instanceColor) m.instanceColor.needsUpdate = true; }
  }

  // ── 個体生成 ──
  private newFounder(age: number, sex: 0 | 1): LifeAgent {
    this.totalFounders++;
    return { id: this.nextId++, age, sex, span: SPAN_MIN + this.rngSim() * SPAN_RANGE, preg: 0, line: this.nextLine++, body: -1 };
  }

  seed(u: number, v: number) {
    const agents: LifeAgent[] = [];
    for (let i = 0; i < 4; i++) agents.push(this.newFounder(18 + this.rngSim() * 4, (i % 2 === 0 ? 0 : 1)));
    this.bands.push({ u, v, wander: 0, agents, food: BAND_FOOD_INIT, migCd: 0 });
  }

  // 決定論再シード(検証・A/B比較用)。生命/表示の両乱数列を seed で固定。
  reseed(seed: number) {
    this.rngSim = mulberry32(seed >>> 0);
    this.rngView = mulberry32((seed ^ 0x9e3779b9) >>> 0);
  }

  clock(): number { return this.ticksDone * DT_Y + this.simAcc / YEAR_SEC; }

  clear() {
    this.villages = []; this.bands = []; this.dyingList = [];
    this.eco.clear(); this.ecoRefreshAcc = 0;
    for (const m of this.hutMeshes) m.count = 0; this.fireMesh.count = 0; this.smokeMesh.count = 0; this.hutSig = '';
    for (const p of this.pool) { p.active = false; p.dying = false; p.fade = 1; p.homeRef = null; p.homeKind = null; p.agent = null; }
    for (const m of this.allParts) { for (let i = 0; i < this.maxPeople; i++) m.setMatrixAt(i, this.mZero); m.instanceMatrix.needsUpdate = true; }
    this.totalFounders = 0; this.totalBirths = 0; this.totalDeaths = 0;
    this.totalChopped = 0; this.totalTreesConsumed = 0; this.hutsCompleted = 0; this.totalLogsUsed = 0;
    this.migrations = 0; this.respawnAcc = 0; this.atCapacity = false;
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
    // 誘引: 魚/獣が多い水辺は最大1.2倍(人が狩り/漁に来る)
    return flat * (0.55 + 0.45 * waterScore) * (0.45 + 0.55 * (1 - en * 0.7)) * this.eco.attractFactor(u, v);
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
    this.renderPeople();
    this.eco.animate(dt); // 魚の泳ぎ/獣の歩き(view専用・simに影響しない)
    this.renderStatic();
  }

  // ── band: 適地スキャン→直進→定住(毎フレーム・決定論=rng不使用) ──
  private updateBands(dt: number) {
    const stepUv = 0.06 * dt;
    for (let b = this.bands.length - 1; b >= 0; b--) {
      const band = this.bands[b];
      if (band.tu === undefined) {
        const isMig = band.ox !== undefined;
        let best = isMig ? 0 : this.suitability(band.u, band.v) + 0.02, tu = band.u, tv = band.v;
        for (let ri = 1; ri <= 5; ri++) {
          const rad = 0.05 * ri;
          for (let a = 0; a < 12; a++) {
            const ang = (a / 12) * Math.PI * 2 + ri * 0.7;
            const nu = band.u + Math.cos(ang) * rad, nv = band.v + Math.sin(ang) * rad;
            if (nu < 0.04 || nu > 0.96 || nv < 0.04 || nv > 0.96) continue;
            if (isMig && Math.hypot(nu - band.ox!, nv - band.oy!) < 0.08) continue; // 元地の近くは避ける
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
        const vg = this.makeVillage(band.u, band.v, band.agents, band.migCd, Math.max(band.food, autoHuts(band.agents.length) * STORE_PER_HUT * 0.5));
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
    // 生態: 容量は低頻度更新(水/地形は緩変化)、密度は毎tick。
    if (++this.ecoRefreshAcc >= 20) { this.ecoRefreshAcc = 0; this.eco.refreshCapacity(); }
    this.eco.tick(DT_Y);
    for (const vg of this.villages) {
      vg.agents = this.ageAndDie(vg.agents);
      if (this.testHuts >= 0) vg.huts = this.testHuts;
      this.recomputeDemo(vg);
      const econ = this.economy(vg);       // 採集/消費/貯蔵/飢餓死 + builders予約
      this.buildStep(vg, econ.builders);   // 伐採→運搬→建築(huts++)
      this.recomputeDemo(vg);              // 飢餓死後の demo
      this.birthsIn(vg, econ.intakeRate, econ.upkeep); // 完全B式
      this.recomputeDemo(vg);              // 出産後の最終 demo
      vg.age += DT_Y; vg.migCd = Math.max(0, vg.migCd - DT_Y);
    }
    // band: 加齢 + 携行食料消費(尽きたら餓死) + クールダウン
    for (const bd of this.bands) {
      bd.agents = this.ageAndDie(bd.agents);
      bd.migCd = Math.max(0, bd.migCd - DT_Y);
      bd.food -= bd.agents.length * CONS_ADULT * DT_Y;
      if (bd.food < 0) { bd.food = 0; this.starveBand(bd); }
    }
    // 消滅村/空band を除去
    for (let i = this.villages.length - 1; i >= 0; i--) {
      if (this.villages[i].agents.length < VILLAGE_MIN_POP) { this.dissolveVillage(this.villages[i]); this.villages.splice(i, 1); }
    }
    for (let i = this.bands.length - 1; i >= 0; i--) if (this.bands[i].agents.length === 0) this.bands.splice(i, 1);

    this.migrateAndFission();  // N7 破綻→移住 / 繁栄→分村
    this.respawnIfEmpty();     // N7 全滅回避リスポーン

    this.reconcileBodies();
    this.eco.retarget();
  }

  private starveBand(bd: Band) {
    let victim = -1, bestAge = -1;
    for (let i = 0; i < bd.agents.length; i++) if (bd.agents[i].age > bestAge) { bestAge = bd.agents[i].age; victim = i; }
    if (victim >= 0) { const a = bd.agents[victim]; this.totalDeaths++; if (a.body >= 0) this.killBody(a); bd.agents.splice(victim, 1); }
  }

  // 破綻→移住(村→band退行) / 繁栄→分村(30%を新band・新line)
  private migrateAndFission() {
    for (let i = this.villages.length - 1; i >= 0; i--) {
      const vg = this.villages[i];
      if (vg.migCd > 0) continue;
      if (vg.hungerY >= MIG_THRESHOLD && vg.agents.length >= VILLAGE_MIN_POP) {
        for (const a of vg.agents) if (a.body >= 0) this.releaseBody(a);
        this.bands.push({ u: vg.u, v: vg.v, wander: 0, agents: vg.agents, food: vg.foodStock, migCd: MIG_COOLDOWN, ox: vg.u, oy: vg.v });
        this.villages.splice(i, 1); this.migrations++;
      } else if (vg.agents.length >= FISSION_POP && vg.surplusY >= FISSION_SURPLUS) {
        const take = Math.floor(vg.agents.length * 0.3);
        if (take >= 2) {
          const movers = vg.agents.splice(vg.agents.length - take, take);
          for (const a of movers) { a.line = this.nextLine++; if (a.body >= 0) this.releaseBody(a); }
          this.bands.push({ u: vg.u, v: vg.v, wander: 0, agents: movers, food: vg.foodStock * 0.3, migCd: MIG_COOLDOWN, ox: vg.u, oy: vg.v });
          vg.foodStock *= 0.7; vg.surplusY = 0; this.recomputeDemo(vg);
        }
      }
    }
  }

  // 全滅回避: 総人口 < RESPAWN_POP が続いたら最適水辺に4人band(4新line)を約45実秒毎
  private respawnIfEmpty() {
    if (this._live() < RESPAWN_POP) {
      this.respawnAcc += DT_Y;
      if (this.respawnAcc >= RESPAWN_SEC / YEAR_SEC) {
        this.respawnAcc = 0;
        // 最適地(水辺=suitability高)を粗探索
        let bu = 0.5, bv = 0.5, best = -1;
        for (let j = 0; j < 16; j++) for (let k = 0; k < 16; k++) {
          const u = (k + 0.5) / 16, v = (j + 0.5) / 16, s = this.suitability(u, v);
          if (s > best) { best = s; bu = u; bv = v; }
        }
        const agents: LifeAgent[] = [];
        for (let m = 0; m < 4; m++) agents.push(this.newFounder(18 + this.rngSim() * 4, (m % 2 === 0 ? 0 : 1)));
        this.bands.push({ u: bu, v: bv, wander: 0, agents, food: BAND_FOOD_INIT, migCd: 0 });
      }
    } else this.respawnAcc = 0;
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

  // 食料経済。builders を労働から予約し残りが採集。飢餓死まで処理。
  private economy(vg: Village): { intakeRate: number; upkeep: number; builders: number } {
    const kids = vg.kids, adults = vg.adultsF + vg.adultsM, elders = vg.elders;
    const labor = adults + LABOR_ELDER * elders;
    const upkeep = kids * CONS_KID + adults * CONS_ADULT + elders * CONS_ELDER; // fu/年
    // 家0軒(定住直後)でも 1軒分は野天に積める。これが無いと携行食料が全捨てされ即飢える。
    const storageCap = Math.max(1, vg.huts) * STORE_PER_HUT;
    // 建築者を予約(食料が薄いと0=全員食料へ)。残りが採集。
    const foodGate = this.testInfFood ? 1 : clamp((vg.foodStock / Math.max(1e-6, upkeep) - 0.1) / 0.3, 0, 1);
    // 家0軒なら空腹でも最低1人は建てる(雨露をしのぐのが先)。テントは丸太不要ゆえ伐採は起きない。
    const base = Math.min(2, Math.floor(labor * 0.35 * foodGate));
    const builders = (vg.huts === 0 && labor > 0) ? Math.max(1, base) : base;
    let avail = Math.max(0, labor - builders);
    // 漁/狩を割当(ポテンシャルがあれば)。残りが採集。
    const fishPot = this.eco.fishPot(vg.u, vg.v), huntPot = this.eco.huntPot(vg.u, vg.v);
    const fishers = fishPot > 0.2 ? Math.min(avail, 8, Math.max(1, Math.round(avail * 0.5))) : 0;
    avail -= fishers;
    const hunters = huntPot > 0.2 ? Math.min(avail, 6, Math.max(1, Math.round(avail * 0.4))) : 0;
    avail -= hunters;
    const Wg = avail;
    const effW = Math.min(Wg, 12) + 0.25 * Math.max(0, Wg - 12); // 収穫逓減
    const s = this.suitability(vg.u, vg.v);
    // 収穫(魚/獣を除去し食料へ)。intakeRate は 年率に換算。
    const takeFish = this.eco.harvestFish(vg.u, vg.v, fishers, DT_Y);
    const takeAnimal = this.eco.harvestLand(vg.u, vg.v, hunters, DT_Y);
    const harvestRate = (takeFish + takeAnimal) / DT_Y; // fu/年
    // 気候振動 ±30%・周期12年・地域位相(放置でも凶作/豊作が巡る=創発の核)
    const ph = (this.clock() / CLIMATE_PERIOD + vg.phi);
    const climate = 1 + CLIMATE_AMP * CLIMATE_LUT[(((Math.floor(ph * 256) % 256) + 256) % 256)];
    const intakeRate = (GATHER_BASE * effW * s + harvestRate) * climate; // fu/年
    if (this.testInfFood) {
      vg.foodStock = storageCap; vg.hungerY = 0; vg.deathAcc = 0; vg.surplusY += DT_Y;
      return { intakeRate: Math.max(intakeRate, upkeep * 1.2), upkeep, builders };
    }
    const intake = intakeRate * DT_Y;
    const need = upkeep * DT_Y;
    const supply = vg.foodStock + intake;
    const eaten = Math.min(need, supply);
    const nutrition = need > 0 ? eaten / need : 1;
    vg.foodStock = Math.min(storageCap, supply - eaten);
    if (nutrition < 1) vg.hungerY += DT_Y * HUNGER_RATE * (1 - nutrition);
    else vg.hungerY = Math.max(0, vg.hungerY - DT_Y);
    vg.surplusY = (intake >= 1.1 * need) ? vg.surplusY + DT_Y : 0;
    // 飢餓死(高齢順・繁殖適齢Fは他が尽きるまで温存)
    vg.deathAcc += vg.agents.length * M_STARVE * (1 - nutrition) * DT_Y;
    let guard = 0;
    while (vg.deathAcc >= 1 && vg.agents.length > 0 && guard++ < MAX_AGENTS) { vg.deathAcc -= 1; this.starve(vg); }
    return { intakeRate, upkeep, builders };
  }

  // 伐採→運搬→建築(1村1サイト)。木を消費し、丸太8+建築0.5worker年で1軒完成。
  private buildStep(vg: Village, builders: number) {
    const cap = TREES_CAP_BASE * this.vegScore(vg.u, vg.v);
    vg.trees = Math.min(cap, vg.trees + TREE_REGROW * DT_Y); // 木の再生
    if (this.testHuts >= 0 || builders <= 0) return;         // huts固定テスト時は建てない
    // 着工: 空き & 目標軒数(autoHuts)に足りない & (丸太不要 or 木がある)
    // 🔴 目標は autoHuts = 旧「家credit」と同じ水準。credit を廃した今、そこへ建てて到達する。
    //    旧条件 pop > huts*6-2 は credit で2軒ある前提の較正値で、huts=0 始まりだと
    //    1軒目の直後に偽になり E_house=0.33 で出産が止まる → 人口が増えず永久に2軒目が建たない
    //    (credit が隠していた鶏卵問題がそのまま出る)。実測で pop が 20年後 7人に停滞して発覚。
    if (!vg.site && vg.huts < autoHuts(vg.agents.length)) {
      const st = stageOf(vg.agents.length);
      const need = LOGS_BY_STAGE[st];
      if (need === 0 || vg.trees >= 4) vg.site = { logs: 0, workY: 0, stage: 0, need, work: WORK_BY_STAGE[st] };
    }
    const site = vg.site;
    if (!site) return;
    if (site.logs < site.need) {
      // 伐採+運搬(木を消費・2丸太/本)
      const add = Math.min(site.need - site.logs, builders * LUMBER_RATE * DT_Y, vg.trees * LOGS_PER_TREE);
      if (add > 0) {
        site.logs += add; const treesUsed = add / LOGS_PER_TREE; vg.trees -= treesUsed;
        this.totalChopped += add; this.totalTreesConsumed += treesUsed;
      }
    } else {
      site.workY = Math.min(site.work, site.workY + builders * DT_Y); // 建築
    }
    site.stage = site.logs < site.need ? 0 : (site.workY < site.work * 0.5 ? 1 : 2);
    if (site.logs >= site.need && site.workY >= site.work) { // 完成
      vg.huts = Math.min(12, vg.huts + 1); this.hutsCompleted++;
      this.totalLogsUsed += site.need; vg.site = null;
    }
  }

  private starve(vg: Village) {
    let victim = -1, bestAge = -1;
    for (let i = 0; i < vg.agents.length; i++) {
      const a = vg.agents[i];
      if (a.sex === 0 && a.age >= FERT_MIN && a.age <= FERT_MAX) continue; // 繁殖適齢Fは温存
      if (a.age > bestAge) { bestAge = a.age; victim = i; }
    }
    if (victim < 0) for (let i = 0; i < vg.agents.length; i++) if (vg.agents[i].age > bestAge) { bestAge = vg.agents[i].age; victim = i; }
    if (victim >= 0) {
      const a = vg.agents[victim];
      this.totalDeaths++; if (this.recordDeaths) this.deathLog.push(a.age); if (a.body >= 0) this.killBody(a);
      vg.agents.splice(victim, 1);
    }
  }

  // 出産(N2 完全版: B = fertBase*eligibleF*E_food*E_house*E_pair*E_size*E_div)
  private birthsIn(vg: Village, intakeRate: number, upkeep: number) {
    const agents = vg.agents;
    let m15 = 0, f15 = 0, eligF = 0;
    const lineSet = new Set<number>();
    for (const a of agents) {
      lineSet.add(a.line);
      if (a.age >= ADULT_AGE) { if (a.sex === 1) m15++; else f15++; }
      if (a.sex === 0 && a.age >= FERT_MIN && a.age <= FERT_MAX && a.preg <= 0) eligF++;
    }
    const pop = agents.length, adults = m15 + f15, lines = lineSet.size;
    const up = Math.max(1e-6, upkeep);
    const ePair = (m15 + f15 > 0) ? 2 * Math.min(m15, f15) / (m15 + f15) : 0;
    const eHouse = clamp((vg.huts * 6 - pop) / 6, 0, 1);
    const eSize = clamp(adults / 4, 0, 1);
    const eDiv = clamp(0.5 + 0.166 * (lines - 1), 0.5, 1.0);
    const eFood = this.testInfFood ? 1
      : clamp((vg.foodStock / up + 0.5 * Math.max(0, intakeRate / up - 1) - 0.1) / 0.3, 0, 1);
    vg.birthAcc += FERT_BASE * eligF * eFood * eHouse * ePair * eSize * eDiv * DT_Y;
    const diff = m15 - f15;
    while (vg.birthAcc >= 1 && agents.length < MAX_AGENTS) {
      vg.birthAcc -= 1;
      let sex: 0 | 1;
      if (Math.abs(diff) > 3) { const minority: 0 | 1 = diff > 0 ? 0 : 1; sex = this.rngSim() < 0.7 ? minority : (minority === 0 ? 1 : 0); }
      else sex = this.rngSim() < 0.5 ? 0 : 1;
      const span = SPAN_MIN + this.rngSim() * SPAN_RANGE;
      let motherLine = -1;
      for (const a of agents) { if (a.sex === 0 && a.age >= FERT_MIN && a.age <= FERT_MAX && a.preg <= 0) { a.preg = PREG_CD; motherLine = a.line; break; } }
      const line = (this.rngSim() < LINE_MUT || motherLine < 0) ? this.nextLine++ : motherLine;
      agents.push({ id: this.nextId++, age: 0, sex, span, preg: 0, line, body: -1 });
      this.totalBirths++;
    }
    if (vg.birthAcc >= 1 && agents.length >= MAX_AGENTS) { this.atCapacity = true; vg.birthAcc = 0; } // 収容力到達=出生を静かに棄却
  }

  // 植生スコア(M3植生未実装のfallback): 平地×低地×湿り。木の上限に使う。川を掘ると森が濃い。
  private vegScore(u: number, v: number): number {
    const flat = this.flat(u, v);
    const range = Math.max(1, this.sensor.elevMax - this.sensor.elevMin);
    const en = Math.min(1, Math.max(0, (this.sensor.heightUV(u, v) - this.sensor.elevMin) / range));
    const wet = this.sensor.wetUV(u, v);
    return flat * (1 - en) * (0.3 + 0.7 * Math.min(1, wet * 2));
  }
  private makeVillage(u: number, v: number, agents: LifeAgent[], migCd = 0, food?: number): Village {
    // 🔴 家credit なし(huts=0)。定住した人が自分でテントを張る過程を必ず見せる。
    const phi = hash01(Math.round(u * 32) + Math.round(v * 32) * 37 + 1); // 地域ごとの気候位相
    const vg: Village = {
      u, v, huts: 0, age: 0, agents, birthAcc: 0, kids: 0, adultsF: 0, adultsM: 0, elders: 0,
      foodStock: food ?? STORE_PER_HUT, hungerY: 0, surplusY: 0, deathAcc: 0,
      trees: TREES_CAP_BASE * this.vegScore(u, v), site: null,
      migCd, phi,
    };
    this.recomputeDemo(vg);
    return vg;
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
    p.active = true; p.dying = false; p.fade = 1; p.homeRef = ref; p.homeKind = kind; p.agent = a;
    a.body = p.idx;
    p.u = ref.u; p.v = ref.v; p.tu = ref.u; p.tv = ref.v; p.state = 'pause'; p.timer = this.rngView() * 1.5;
    p.speed = 0.005 + this.rngView() * 0.006;
    p.fx = (this.rngView() - 0.5) * this.U * 0.7 / this.sensor.worldW;
    p.fy = (this.rngView() - 0.5) * this.U * 0.7 / this.sensor.worldW;
    this.setAppearance(p, a);
    return true;
  }
  private releaseBody(a: LifeAgent) {
    if (a.body < 0) return;
    const p = this.pool[a.body];
    if (p) { p.active = false; p.dying = false; p.agent = null; p.fade = 1; p.homeRef = null; p.homeKind = null; }
    a.body = -1;
  }
  // 死亡した個体の体は 2 秒かけてフェード(rng不使用)。agent 参照は縮小描画用に保持。
  private killBody(a: LifeAgent) {
    if (a.body < 0) return;
    const p = this.pool[a.body];
    if (p) { p.dying = true; p.homeRef = null; p.homeKind = null; this.dyingList.push({ p, t: DEATH_FADE }); }
    a.body = -1;
  }
  private updateDying(dt: number) {
    for (let i = this.dyingList.length - 1; i >= 0; i--) {
      const d = this.dyingList[i]; d.t -= dt;
      d.p.fade = Math.max(0, d.t / DEATH_FADE);
      if (d.t <= 0) {
        d.p.active = false; d.p.dying = false; d.p.agent = null; d.p.fade = 1;
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

  // 論理更新のみ(移動/位相/向き)。描画は renderPeople が instance 行列へ。
  private updatePerson(p: Person, dt: number) {
    const W = this.sensor.worldW;
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
    p.phase += (moved * W) / this.stride;
  }

  // 部位別 InstancedMesh へ全人物の行列を書き込む(毎フレーム=歩行アニメ)
  private renderPeople() {
    for (const p of this.pool) {
      const i = p.idx;
      if (!(p.active && p.agent)) { this.zeroPersonParts(i); continue; }
      const ageScale = 0.4 + 0.6 * Math.min(p.agent.age, ADULT_AGE) / ADULT_AGE;
      const s = p.fade * p.heightScale * ageScale;
      if (s <= 0.001) { this.zeroPersonParts(i); continue; }
      this.pos(p.u, p.v, this.vPos);
      this.vPos.y += Math.abs(Math.cos(p.phase)) * this.U * 0.03 * s;
      this.qTmp.setFromEuler(this.eTmp.set(0, p.facing, 0));
      this.mRoot.compose(this.vPos, this.qTmp, this.vScl.set(s, s, s));
      const sw = Math.sin(p.phase) * 0.6 * p.gait;
      const H = this.H, hipY = this.hipY, torsoH = this.torsoH;
      // 脚: 腿(股で振る)+脛(膝で後ろへ曲がる。後ろに振れた側ほど曲げて踵を上げる)
      const hipL = sw, hipR = -sw;
      const kneeL = -Math.max(0, -hipL) * 1.4, kneeR = -Math.max(0, -hipR) * 1.4;
      const legX = H * 0.045 * p.hipW;
      this.setPart(this.partThighL, i, -legX, hipY, 0, hipL, 1, 1, 1);
      this.setPart(this.partThighR, i, legX, hipY, 0, hipR, 1, 1, 1);
      this.setPart2(this.partShinL, i, -legX, hipY, 0, hipL, this.thighLen, kneeL);
      this.setPart2(this.partShinR, i, legX, hipY, 0, hipR, this.thighLen, kneeR);
      // 腕: 上腕(肩で振る)+前腕(肘は常に軽く曲げ、前振り時に更に曲げる)
      const shY = hipY + torsoH * 0.92, armX = H * 0.12 * p.shoulderW;
      const armL = -sw * 0.7, armR2 = sw * 0.7;
      const elbL = 0.25 + Math.max(0, armL) * 0.5, elbR = 0.25 + Math.max(0, armR2) * 0.5;
      this.setPart(this.partUpArmL, i, -armX, shY, 0, armL, 1, 1, 1);
      this.setPart(this.partUpArmR, i, armX, shY, 0, armR2, 1, 1, 1);
      this.setPart2(this.partForeL, i, -armX, shY, 0, armL, this.upArmLen, elbL);
      this.setPart2(this.partForeR, i, armX, shY, 0, armR2, this.upArmLen, elbR);
      this.setPart(this.partTorso, i, 0, hipY + torsoH * 0.5, 0, 0, p.shoulderW, 1, 1);
      this.setPart(this.partHead, i, 0, hipY + torsoH + H * 0.1, 0, 0, 1, 1, 1);
      if (p.hairLong) { this.setPart(this.partHairL, i, 0, hipY + torsoH + H * 0.11, 0, 0, 1, 1, 1); this.partHairS.setMatrixAt(i, this.mZero); }
      else { this.setPart(this.partHairS, i, 0, hipY + torsoH + H * 0.12, 0, 0, 1, 1, 1); this.partHairL.setMatrixAt(i, this.mZero); }
      if (p.sex === 1) { this.setPart(this.partClothM, i, 0, hipY + H * 0.02, 0, 0, p.hipW, 1, p.hipW); this.partClothF.setMatrixAt(i, this.mZero); }
      else { this.setPart(this.partClothF, i, 0, hipY + H * 0.02, 0, 0, p.hipW, 1, p.hipW); this.partClothM.setMatrixAt(i, this.mZero); }
    }
    for (const m of this.allParts) m.instanceMatrix.needsUpdate = true;
  }
  // 2セグ関節: local = T(off) * Rx(a1) * T(0,-len,0) * Rx(a2)。下位セグ(脛/前腕)の world 行列。
  private setPart2(mesh: THREE.InstancedMesh, i: number, ox: number, oy: number, oz: number, a1: number, len: number, a2: number) {
    this.qTmp.setFromEuler(this.eTmp.set(a1, 0, 0));
    this.mLocal.compose(this.vTmp.set(ox, oy, oz), this.qTmp, this.vScl.set(1, 1, 1));
    this.qTmp.setFromEuler(this.eTmp.set(a2, 0, 0));
    this.mLocal2.compose(this.vTmp.set(0, -len, 0), this.qTmp, this.vScl.set(1, 1, 1));
    this.mOut.multiplyMatrices(this.mLocal, this.mLocal2);
    this.mOut.premultiply(this.mRoot);
    mesh.setMatrixAt(i, this.mOut);
  }
  private setPart(mesh: THREE.InstancedMesh, i: number, ox: number, oy: number, oz: number, rotX: number, sx: number, sy: number, sz: number) {
    this.qTmp.setFromEuler(this.eTmp.set(rotX, 0, 0));
    this.mLocal.compose(this.vTmp.set(ox, oy, oz), this.qTmp, this.vScl.set(sx, sy, sz));
    this.mOut.multiplyMatrices(this.mRoot, this.mLocal);
    mesh.setMatrixAt(i, this.mOut);
  }
  private zeroPersonParts(i: number) { for (const m of this.allParts) m.setMatrixAt(i, this.mZero); }

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
    // 家: 村構成/建築ステージ/成長段階が変わった時だけ再構築(ダーティフラグ)。段階=module stageOf
    let sig = this.villages.length + '|';
    for (const vg of this.villages) sig += vg.huts + '/' + (vg.site ? vg.site.stage + 1 : 0) + '/' + stageOf(vg.agents.length) + ',' + Math.round(vg.u * 1e4) + ',' + Math.round(vg.v * 1e4) + ';';
    if (sig !== this.hutSig) {
      this.hutSig = sig;
      const hc = [0, 0, 0]; // 段階別カウンタ
      for (const vg of this.villages) {
        const st = stageOf(vg.agents.length);
        const mesh = this.hutMeshes[st];
        for (let k = 0; k < vg.huts && hc[st] < MAX_HUTS; k++) {
          const ang = k * 2.399963;
          const rad = U * 1.3 * Math.sqrt(k);
          const u = vg.u + (Math.cos(ang) * rad) / W, v = vg.v + (Math.sin(ang) * rad) / W;
          this.pos(u, v, d.position); d.rotation.set(0, ang * 2.1, 0);
          d.scale.setScalar(0.8 + 0.25 * ((k * 7) % 5) / 5); d.updateMatrix();
          mesh.setMatrixAt(hc[st]++, d.matrix);
        }
        // 建築中サイト = 次のスパイラル位置に段階スケールの小屋(村の現段階メッシュで)
        if (vg.site && hc[st] < MAX_HUTS) {
          const k = vg.huts, ang = k * 2.399963, rad = U * 1.3 * Math.sqrt(k);
          const u = vg.u + (Math.cos(ang) * rad) / W, v = vg.v + (Math.sin(ang) * rad) / W;
          this.pos(u, v, d.position); d.rotation.set(0, ang * 2.1, 0);
          d.scale.setScalar([0.35, 0.6, 0.85][vg.site.stage] ?? 0.35); d.updateMatrix();
          mesh.setMatrixAt(hc[st]++, d.matrix);
        }
      }
      for (let s = 0; s < 3; s++) { this.hutMeshes[s].count = hc[s]; this.hutMeshes[s].instanceMatrix.needsUpdate = true; }
    }
    let fi = 0, si = 0;
    for (const vg of this.villages) {
      if (fi >= MAX_FIRES) break;
      this.pos(vg.u, vg.v, d.position);
      const baseY = d.position.y;
      d.rotation.set(0, 0, 0);
      d.scale.set(1, 0.85 + 0.25 * Math.sin(this.t * 6 + fi), 1); d.updateMatrix();
      this.fireMesh.setMatrixAt(fi, d.matrix);
      // 立ち昇る煙: 各パフが上昇→肥大→頂上でフェード(scale0)
      for (let p = 0; p < SMOKE_PER_FIRE; p++) {
        const ph = (this.t * 0.35 + p / SMOKE_PER_FIRE + fi * 0.37) % 1;
        const fade = ph < 0.8 ? 1 : Math.max(0, (1 - ph) / 0.2);
        // 🔴 scale は無単位倍率(ジオメトリが既に U*0.28 の実寸)。U を掛けると画面を埋める球になる。
        const sc = (0.35 + ph * 0.75) * fade;
        d.position.y = baseY + ph * U * 2.2;
        d.rotation.set(0, 0, 0);
        d.scale.setScalar(sc); d.updateMatrix();
        this.smokeMesh.setMatrixAt(si++, d.matrix);
      }
      fi++;
    }
    this.fireMesh.count = fi; this.fireMesh.instanceMatrix.needsUpdate = true;
    this.smokeMesh.count = si; this.smokeMesh.instanceMatrix.needsUpdate = true;
  }

  stats() {
    let pop = 0, kids = 0, adults = 0, elders = 0, food = 0, huts = 0, trees = 0;
    for (const v of this.villages) { pop += v.agents.length; kids += v.kids; adults += v.adultsF + v.adultsM; elders += v.elders; food += v.foodStock; huts += v.huts; trees += v.trees; }
    for (const b of this.bands) pop += b.agents.length;
    return {
      villages: this.villages.length, pop, bands: this.bands.length,
      year: Math.round(this.clock() * 1000) / 1000,
      births: this.totalBirths, deaths: this.totalDeaths,
      kids, adults, elders,
      food: Math.round(food * 100) / 100,
      huts, trees: Math.round(trees),
      fish: Math.round(this.eco.totals().fish), animals: Math.round(this.eco.totals().land),
      migrations: this.migrations, atCapacity: this.atCapacity,
    };
  }

  // ── ヘッドレス検証用デバッグAPI ──
  _live(): number { let n = 0; for (const v of this.villages) n += v.agents.length; for (const b of this.bands) n += b.agents.length; return n; }
  _counts() { return { founders: this.totalFounders, births: this.totalBirths, deaths: this.totalDeaths, live: this._live() }; }
  _injectVillage(u: number, v: number, specs: { age: number; sex: 0 | 1 }[]) {
    const agents = specs.map((s) => this.newFounder(s.age, s.sex));
    const vg = this.makeVillage(u, v, agents); this.villages.push(vg); return vg;
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
  _setTestHuts(n: number) { this.testHuts = n; }
  _setInfFood(b: boolean) { this.testInfFood = b; }
  _villageFood(i: number): number { return this.villages[i]?.foodStock ?? 0; }
  _villageHunger(i: number): number { return this.villages[i]?.hungerY ?? 0; }
  _lines(i: number): number { const v = this.villages[i]; if (!v) return 0; const s = new Set<number>(); for (const a of v.agents) s.add(a.line); return s.size; }
  _suit(u: number, v: number): number { return this.suitability(u, v); }
  _villageDemo(i: number) { const v = this.villages[i]; if (!v) return null; return { pop: v.agents.length, kids: v.kids, af: v.adultsF, am: v.adultsM, elders: v.elders, food: Math.round(v.foodStock * 100) / 100, hunger: Math.round(v.hungerY * 100) / 100, huts: v.huts }; }
  _villageBuild(i: number) { const v = this.villages[i]; if (!v) return null; return { huts: v.huts, trees: Math.round(v.trees * 100) / 100, site: v.site ? { logs: Math.round(v.site.logs * 100) / 100, workY: Math.round(v.site.workY * 1000) / 1000, stage: v.site.stage } : null }; }
  _villageMig(i: number) { const v = this.villages[i]; if (!v) return null; return { pop: v.agents.length, hungerY: Math.round(v.hungerY * 1000) / 1000, migCd: Math.round(v.migCd * 100) / 100, surplusY: Math.round(v.surplusY * 100) / 100 }; }
  _forceHunger(i: number, y: number) { const v = this.villages[i]; if (v) { v.hungerY = y; v.migCd = 0; } }
  _forceSurplus(i: number, y: number) { const v = this.villages[i]; if (v) { v.surplusY = y; v.migCd = 0; } }
  _migrations() { return this.migrations; }
  _counters7() { return { migrations: this.migrations, villages: this.villages.length, bands: this.bands.length, live: this._live(), atCapacity: this.atCapacity }; }
  // 保存則: chopped == Σ(建築中サイトの丸太) + logsUsed(完成した家が消費した丸太)
  _woodStats() {
    let siteLogs = 0; for (const v of this.villages) if (v.site) siteLogs += v.site.logs;
    return {
      chopped: Math.round(this.totalChopped * 1000) / 1000,
      treesConsumed: Math.round(this.totalTreesConsumed * 1000) / 1000,
      hutsCompleted: this.hutsCompleted, logsUsed: Math.round(this.totalLogsUsed * 1000) / 1000,
      siteLogs: Math.round(siteLogs * 1000) / 1000,
    };
  }
  _bestSuit(): number { let best = 0; const N = 40; for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) { const s = this.suitability((i + 0.5) / N, (j + 0.5) / N); if (s > best) best = s; } return best; }
}
