// terra-touch geo — 実衛星画像 × 実DEM を指で変形するサンドボックス (PC/スマホ両対応)。
//   - 衛星画像(ESRI/Maxar)=色、実DEM(AWS Terrain)=標高 (どちらもトークン不要/CORS)。
//   - 実DEM は sim の bedrock に注入 = 実地形は不動。押した所だけ soil化して安息角で崩れる。
//   - 描画メッシュ = 高精細DEM + sim差分。水は既定OFF、「雨」で実谷に集まる。
//   - スマホ: 1本指=見る(回転)/2本指=ズーム。「彫る」ボタンで1本指を変形に切替。
//     静止時はシム停止で軽量。

import * as THREE from 'three/webgpu';
import {
  texture, uv, vec2, vec3, float, mix, smoothstep,
  normalize as nrm, positionLocal, transformNormalToView, uniform, sin,
} from 'three/tsl';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { SkyMesh } from 'three/addons/objects/SkyMesh.js';
import { TerrainSim } from './sim';
import { GpuSim } from './sim-gpu';
import { VillageSystem, type WorldSensor } from './village';
import { LOCATIONS, pickLocation, computeZoom } from './locations';

const GRID = 4, TILE = 256;
const DEM_SIZE = GRID * TILE;
const EARTH_C = 40075016.686;

// URL の ?loc= で場所を選択 (既定=富士)
const LOC = pickLocation(new URLSearchParams(location.search).get('loc'));
const LAT = LOC.lat, LON = LOC.lon;
const ZOOM = computeZoom(LAT, LOC.extentKm, GRID);
const VERT_EXAG = LOC.exag;

// 端末に応じて負荷を調整
const IS_TOUCH = matchMedia('(pointer: coarse)').matches || 'ontouchstart' in window;
const SIM_N = IS_TOUCH ? 256 : 512; // 1024 を割り切る値 (シムは雨/彫る時のみ稼働=待機は60fps維持)      // 1024 を割り切る値
const MESH_N = IS_TOUCH ? 512 : 768;
const WATER_MESH = IS_TOUCH ? 384 : 512;
const PIX_CAP = IS_TOUCH ? 1.5 : 2;

const status = document.getElementById('status')!;
const lon2tile = (lon: number, z: number) => ((lon + 180) / 360) * 2 ** z;
const lat2tile = (lat: number, z: number) => {
  const r = (lat * Math.PI) / 180;
  return ((1 - Math.asinh(Math.tan(r)) / Math.PI) / 2) * 2 ** z;
};

async function fetchBitmap(url: string): Promise<ImageBitmap> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${res.status} ${url}`);
  return createImageBitmap(await res.blob());
}

async function fetchMosaic(kind: 'img' | 'dem', xMin: number, yMin: number) {
  const canvas = document.createElement('canvas');
  canvas.width = DEM_SIZE; canvas.height = DEM_SIZE;
  const ctx = canvas.getContext('2d', { willReadFrequently: kind === 'dem' })!;
  const jobs: Promise<void>[] = [];
  let done = 0;
  const total = GRID * GRID;
  for (let dy = 0; dy < GRID; dy++) {
    for (let dx = 0; dx < GRID; dx++) {
      const x = xMin + dx, y = yMin + dy;
      const url = kind === 'img'
        ? `https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/${ZOOM}/${y}/${x}`
        : `https://s3.amazonaws.com/elevation-tiles-prod/terrarium/${ZOOM}/${x}/${y}.png`;
      jobs.push(fetchBitmap(url).then((bmp) => {
        ctx.drawImage(bmp, dx * TILE, dy * TILE); bmp.close();
        done++; status.textContent = `${kind === 'img' ? '衛星画像' : '標高'} ${done}/${total}…`;
      }));
    }
  }
  await Promise.all(jobs);
  return { canvas, ctx };
}

async function main() {
  const renderer = new THREE.WebGPURenderer({ antialias: !IS_TOUCH });
  renderer.setPixelRatio(Math.min(devicePixelRatio, PIX_CAP));
  renderer.setSize(innerWidth, innerHeight);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.0;
  // リアルタイム影は無効。衛星画像に既に影が焼き込まれており二重影になる上に重い。
  // 起伏は法線ベースの陰影(shade)で表現する。
  renderer.shadowMap.enabled = false;
  document.body.appendChild(renderer.domElement);
  await renderer.init();

  const cx = lon2tile(LON, ZOOM), cy = lat2tile(LAT, ZOOM);
  const xMin = Math.floor(cx - GRID / 2), yMin = Math.floor(cy - GRID / 2);
  const [img, dem] = await Promise.all([fetchMosaic('img', xMin, yMin), fetchMosaic('dem', xMin, yMin)]);
  status.textContent = '地形を構築中…';

  const px = dem.ctx.getImageData(0, 0, DEM_SIZE, DEM_SIZE).data;
  const demH = new Float32Array(DEM_SIZE * DEM_SIZE);
  let hMin = Infinity, hMax = -Infinity;
  for (let k = 0, p = 0; k < demH.length; k++, p += 4) {
    const h = px[p] * 256 + px[p + 1] + px[p + 2] / 256 - 32768;
    demH[k] = h; if (h < hMin) hMin = h; if (h > hMax) hMax = h;
  }

  const latR = (LAT * Math.PI) / 180;
  const worldW = (EARTH_C / 2 ** ZOOM) * GRID * Math.cos(latR);

  // ── シム: GPU(?gpu=1 かつ WebGPU)= 1024²高解像度 / それ以外 = CPU 512² ──
  const isWebGPU = !!(renderer.backend as { isWebGPUBackend?: boolean }).isWebGPUBackend;
  // 既定: デスクトップ WebGPU は GPU(1024²)。モバイルは安全のため CPU 既定 (?gpu=1 で試行可)。
  const gpuParam = new URLSearchParams(location.search).get('gpu');
  const USE_GPU = isWebGPU && gpuParam !== '0' && (!IS_TOUCH || gpuParam === '1');

  let gpuSim: GpuSim | null = null;
  let cpuSim: TerrainSim | null = null;
  let cpuInitBed: Float32Array | null = null;
  let baseDemTex: THREE.DataTexture | null = null;
  let simBaseTex: THREE.DataTexture | null = null;
  let simTex: THREE.DataTexture | null = null;

  if (USE_GPU) {
    gpuSim = new GpuSim(renderer, { n: DEM_SIZE, world: worldW, bedrock: demH });
    gpuSim.syncDisp();
  } else {
    const factor = DEM_SIZE / SIM_N;
    const simBed = new Float32Array(SIM_N * SIM_N);
    for (let j = 0; j < SIM_N; j++) for (let i = 0; i < SIM_N; i++) {
      let acc = 0;
      for (let bj = 0; bj < factor; bj++) { const sj = j * factor + bj; for (let bi = 0; bi < factor; bi++) acc += demH[sj * DEM_SIZE + (i * factor + bi)]; }
      simBed[j * SIM_N + i] = acc / (factor * factor);
    }
    cpuSim = new TerrainSim({ n: SIM_N, world: worldW, bedrock: simBed });
    cpuInitBed = simBed.slice();
    const toHalf = THREE.DataUtils.toHalfFloat;
    const baseHalf = new Uint16Array(demH.length);
    for (let k = 0; k < demH.length; k++) baseHalf[k] = toHalf(demH[k]);
    baseDemTex = new THREE.DataTexture(baseHalf, DEM_SIZE, DEM_SIZE, THREE.RedFormat, THREE.HalfFloatType);
    baseDemTex.minFilter = baseDemTex.magFilter = THREE.LinearFilter;
    baseDemTex.wrapS = baseDemTex.wrapT = THREE.ClampToEdgeWrapping; baseDemTex.needsUpdate = true;
    const simBaseHalf = new Uint16Array(simBed.length);
    for (let k = 0; k < simBed.length; k++) simBaseHalf[k] = toHalf(simBed[k]);
    simBaseTex = new THREE.DataTexture(simBaseHalf, SIM_N, SIM_N, THREE.RedFormat, THREE.HalfFloatType);
    simBaseTex.minFilter = simBaseTex.magFilter = THREE.LinearFilter;
    simBaseTex.wrapS = simBaseTex.wrapT = THREE.ClampToEdgeWrapping; simBaseTex.needsUpdate = true;
    simTex = cpuSim.tex;
  }

  const imgTex = new THREE.CanvasTexture(img.canvas);
  imgTex.colorSpace = THREE.SRGBColorSpace;
  imgTex.minFilter = THREE.LinearMipmapLinearFilter; imgTex.magFilter = THREE.LinearFilter;
  imgTex.generateMipmaps = true;
  imgTex.anisotropy = renderer.getMaxAnisotropy?.() ?? 8;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x9fb8d4);
  scene.fog = new THREE.Fog(0xaec4dd, worldW * 0.7, worldW * 2.4);
  const uTime = uniform(0);

  const camera = new THREE.PerspectiveCamera(52, innerWidth / innerHeight, 1, worldW * 6);
  camera.position.set(worldW * 0.36, (hMax - hMin) * VERT_EXAG + worldW * 0.28, worldW * 0.5);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, hMax * 0.35 * VERT_EXAG, 0);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;       // 慣性を少し強めて手触りを滑らかに
  controls.zoomToCursor = true;        // ホイール/ピンチはカーソル(2本指中点)へズーム
  controls.screenSpacePanning = false; // パンは地面(水平)沿い=神ゲーの俯瞰移動に自然
  controls.maxPolarAngle = Math.PI * 0.495;
  controls.minDistance = worldW * 0.02; controls.maxDistance = worldW * 2.5;
  const CAM_MIN_D = controls.minDistance, CAM_MAX_D = controls.maxDistance;

  const sun = new THREE.Vector3().setFromSphericalCoords(1, Math.PI / 2 - THREE.MathUtils.degToRad(34), THREE.MathUtils.degToRad(150));
  const sky = new SkyMesh();
  sky.scale.setScalar(worldW * 4);
  sky.turbidity.value = 3.5; sky.rayleigh.value = 1.1;
  sky.mieCoefficient.value = 0.004; sky.mieDirectionalG.value = 0.85;
  sky.sunPosition.value.copy(sun); scene.add(sky);

  const dir = new THREE.DirectionalLight(0xfff4e6, 3.0);
  dir.position.copy(sun).multiplyScalar(worldW);
  scene.add(dir);
  scene.add(new THREE.HemisphereLight(0xbcd2ee, 0x4a4238, 0.55));

  const demTexel = 1 / DEM_SIZE;
  const cell = worldW / DEM_SIZE;
  /* eslint-disable @typescript-eslint/no-explicit-any */
  let groundAt: (ox: number, oy: number) => any;
  let depthNode: any, wetNode: any;
  let waterAt: (ox: number, oy: number) => any;
  if (USE_GPU) {
    const d = gpuSim!.dispTex; // [bed+soil, water, wet, 1]・Nearest
    const dAt = (ox: number, oy: number) => texture(d, uv().add(vec2(ox * demTexel, oy * demTexel)));
    groundAt = (ox, oy) => dAt(ox, oy).r.mul(VERT_EXAG);
    depthNode = dAt(0, 0).g;
    wetNode = dAt(0, 0).b;
    waterAt = (ox, oy) => dAt(ox, oy).g;
  } else {
    const baseAt = (ox: number, oy: number) => texture(baseDemTex!, uv().add(vec2(ox * demTexel, oy * demTexel))).r;
    const simSolidAt = (ox: number, oy: number) => { const s = texture(simTex!, uv().add(vec2(ox * demTexel, oy * demTexel))); return s.r.add(s.g); };
    const simBaseAt = (ox: number, oy: number) => texture(simBaseTex!, uv().add(vec2(ox * demTexel, oy * demTexel))).r;
    groundAt = (ox, oy) => baseAt(ox, oy).add(simSolidAt(ox, oy)).sub(simBaseAt(ox, oy)).mul(VERT_EXAG);
    depthNode = texture(simTex!, uv()).b;
    wetNode = texture(simTex!, uv()).a;
    waterAt = (ox, oy) => texture(simTex!, uv().add(vec2(ox * demTexel, oy * demTexel))).b;
  }

  // GPU は 1024² シムと 1:1 でメッシュも高精細に
  const meshN = USE_GPU ? (IS_TOUCH ? 768 : 1024) : MESH_N;
  const waterMeshN = USE_GPU ? (IS_TOUCH ? 512 : 768) : WATER_MESH;
  const geo = new THREE.PlaneGeometry(worldW, worldW, meshN - 1, meshN - 1);
  geo.rotateX(-Math.PI / 2);
  const mat = new THREE.MeshStandardNodeMaterial({ roughness: 0.95, metalness: 0 });
  mat.positionNode = positionLocal.add(vec3(0, groundAt(0, 0), 0));
  const nLocal = nrm(vec3(groundAt(-1, 0).sub(groundAt(1, 0)), cell * 2, groundAt(0, 1).sub(groundAt(0, -1))));
  mat.normalNode = transformNormalToView(nLocal);
  const shade = mix(float(0.72), float(1.0), smoothstep(0.3, 0.95, nLocal.y));
  const wetA = wetNode;
  const photo = texture(imgTex, uv()).mul(shade);
  mat.colorNode = mix(photo, photo.mul(0.7), smoothstep(0.2, 0.9, wetA)); // 濡れ暗転は控えめに
  const terrain = new THREE.Mesh(geo, mat);
  scene.add(terrain);

  const wgeo = new THREE.PlaneGeometry(worldW, worldW, waterMeshN - 1, waterMeshN - 1);
  wgeo.rotateX(-Math.PI / 2);
  const wmat = new THREE.MeshStandardNodeMaterial({ metalness: 0, roughness: 0.24 });
  wmat.transparent = true; wmat.depthWrite = false;
  wmat.polygonOffset = true; wmat.polygonOffsetFactor = -2; wmat.polygonOffsetUnits = -2; // z-fight緩和
  const depth = depthNode;
  const surfAt = (ox: number, oy: number) => groundAt(ox, oy).add(waterAt(ox, oy).mul(VERT_EXAG));
  wmat.positionNode = positionLocal.add(vec3(0, surfAt(0, 0), 0));

  // ── 水面を「流れて」見せる ──
  // CPU: sim の流速テクスチャ / GPU: 地形勾配から下り方向を推定
  let spd: any, flowDir: any;
  if (USE_GPU) {
    // 下り方向 = -勾配。勾配は地表高の近傍差から。
    const grad = vec2(groundAt(1, 0).sub(groundAt(-1, 0)), groundAt(0, 1).sub(groundAt(0, -1)));
    spd = grad.length().mul(0.3);
    flowDir = nrm(grad.mul(-1).add(vec2(1e-4, 1e-4)));
  } else {
    const vel = texture(cpuSim!.velTex, uv()).xy;
    spd = vel.length();
    flowDir = nrm(vel.add(vec2(1e-4, 1e-4)));
  }
  const proj = uv().x.mul(flowDir.x).add(uv().y.mul(flowDir.y));
  const flowScroll = proj.mul(700).sub(uTime.mul(spd.mul(1.6).add(0.5)));
  const flowWave = sin(flowScroll).mul(smoothstep(0.1, 1.2, spd)).mul(0.08);
  // 静水の微さざ波
  const calm = sin(uv().x.mul(140).add(uTime.mul(0.8))).add(sin(uv().y.mul(120).sub(uTime.mul(0.7)))).mul(0.03);
  const perturb = flowWave.add(calm);
  const wN = nrm(vec3(surfAt(-1, 0).sub(surfAt(1, 0)).add(perturb), cell * 2, surfAt(0, 1).sub(surfAt(0, -1)).add(perturb)));
  wmat.normalNode = transformNormalToView(wN);

  // 深さで色 + フレネルで空 + 流れの速い所に白泡
  // 深い水は光を吸収=暗い。強い直射光(×3)で白飛びしないよう低アルベドに。
  const baseWater = mix(vec3(0.035, 0.10, 0.13), vec3(0.008, 0.028, 0.06), smoothstep(0.4, 8.0, depth));
  const fresnel = float(1).sub(smoothstep(0.2, 0.85, wN.y));
  const skyMix = mix(baseWater, vec3(0.24, 0.34, 0.46), fresnel.mul(0.18)); // 山岳谷の反射は落ち着いた青灰
  const foam = smoothstep(6.0, 8.0, spd).mul(0.12); // 最速部だけごく薄く白波
  wmat.colorNode = mix(skyMix, vec3(0.82, 0.88, 0.93), foam);

  // ゲート: 深さ主体で川筋も見せる。ほぼ垂直な崖だけ膜を隠す。
  const gTerr = nrm(vec3(groundAt(-1, 0).sub(groundAt(1, 0)), cell * 2, groundAt(0, 1).sub(groundAt(0, -1))));
  const notCliff = smoothstep(0.26, 0.5, gTerr.y); // 崖(斜度~75°+)のみ0
  wmat.opacityNode = smoothstep(0.18, 1.1, depth).mul(notCliff).mul(0.94);
  const waterMesh = new THREE.Mesh(wgeo, wmat);
  waterMesh.renderOrder = 1; scene.add(waterMesh);

  // ── 操作モード: 見る(回転) / 彫る ──
  let sculptOn = !IS_TOUCH; // PCは既定で彫る、スマホは既定で見る
  function applyControlMode() {
    if (sculptOn) {
      controls.mouseButtons = { LEFT: null, MIDDLE: THREE.MOUSE.PAN, RIGHT: THREE.MOUSE.ROTATE };
      controls.touches = { ONE: null, TWO: THREE.TOUCH.DOLLY_PAN };
    } else {
      controls.mouseButtons = { LEFT: THREE.MOUSE.ROTATE, MIDDLE: THREE.MOUSE.PAN, RIGHT: THREE.MOUSE.ROTATE };
      controls.touches = { ONE: THREE.TOUCH.ROTATE, TWO: THREE.TOUCH.DOLLY_PAN };
    }
  }
  applyControlMode();

  // ── 統一シムAPI (GPU/CPU 共通) ──
  const simRain = (r: number) => { if (USE_GPU) gpuSim!.rainRate = r; else cpuSim!.rainRate = r; };
  const simBrush = (u: number, v: number, dtb: number, m: 'dig' | 'raise') => {
    if (USE_GPU) gpuSim!.brush(u, v, dtb, m); else cpuSim!.brush(u, v, dtb, m);
  };
  // GPU レイキャスト用: 静的DEMのバイリニア標高 (変形は無視・ピック位置に十分)
  const demHeightUV = (u: number, v: number) => {
    const N = DEM_SIZE;
    const fx = Math.min(Math.max(u, 0), 1) * (N - 1), fy = Math.min(Math.max(v, 0), 1) * (N - 1);
    const i0 = Math.floor(fx), j0 = Math.floor(fy), i1 = Math.min(i0 + 1, N - 1), j1 = Math.min(j0 + 1, N - 1);
    const tx = fx - i0, ty = fy - j0;
    const hh = (i: number, j: number) => demH[j * N + i];
    const a = hh(i0, j0) * (1 - tx) + hh(i1, j0) * tx, b = hh(i0, j1) * (1 - tx) + hh(i1, j1) * tx;
    return a * (1 - ty) + b * ty;
  };
  // 変形追従: 彫った/崩れた後の「実際の地表高」を返す。
  //   GPU: coarse(縮約 ch0=bed+soil)を読戻し済なら使用・未読戻しは静的DEMへフォールバック。
  //   CPU: sim の変形込み地表高。
  // これで pickWorld・カメラめり込み・村の適地判定が彫削後の地形に追従する。
  let coarseReady = false;
  const simHeightUV = (u: number, v: number) =>
    (USE_GPU ? (coarseReady ? gpuSim!.sampleCoarse(u, v, 0) : demHeightUV(u, v)) : cpuSim!.surfaceHeightUV(u, v));

  const raycaster = new THREE.Raycaster();
  const ndc = new THREE.Vector2();
  let sculpting = false;
  let mode: 'dig' | 'raise' = 'dig';
  const pointer = { x: 0, y: 0, has: false };
  const sampleH = (x: number, z: number) => simHeightUV((x + worldW / 2) / worldW, 0.5 - z / worldW) * VERT_EXAG;
  function pickWorld(pxp: number, pyp: number): { x: number; z: number } | null {
    ndc.set((pxp / innerWidth) * 2 - 1, -(pyp / innerHeight) * 2 + 1);
    raycaster.setFromCamera(ndc, camera);
    const o = raycaster.ray.origin, d = raycaster.ray.direction;
    const step = worldW / 700;
    let t = 0, prev = o.y - sampleH(o.x, o.z);
    for (let k = 0; k < 900; k++) {
      t += step; if (t > worldW * 3) break;
      const x = o.x + d.x * t, z = o.z + d.z * t;
      const gap = o.y + d.y * t - sampleH(x, z);
      if (gap <= 0 && prev > 0) {
        let lo = t - step, hi = t;
        for (let b = 0; b < 10; b++) { const m = (lo + hi) * 0.5; const g = o.y + d.y * m - sampleH(o.x + d.x * m, o.z + d.z * m); if (g <= 0) hi = m; else lo = m; }
        const tm = (lo + hi) * 0.5;
        return { x: o.x + d.x * tm, z: o.z + d.z * tm };
      }
      prev = gap;
    }
    return null;
  }

  const canvas = renderer.domElement;
  canvas.addEventListener('pointerdown', (e) => {
    if (!sculptOn) return;
    if (e.button !== 0 && e.pointerType === 'mouse') return;
    sculpting = true; pointer.x = e.offsetX; pointer.y = e.offsetY; pointer.has = true;
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener('pointermove', (e) => { pointer.x = e.offsetX; pointer.y = e.offsetY; });
  const endS = () => { sculpting = false; };
  canvas.addEventListener('pointerup', endS);
  canvas.addEventListener('pointercancel', endS);

  // ── カメラ補助: ダブルタップ/クリックで注目点へ寄る (フォーカス) ──
  // camGoal がセットされると loop 内で target を指数補間しつつ距離を60%へ寄せる。
  let camGoal: { x: number; y: number; z: number; dist: number } | null = null;
  const focusWorld = (x: number, z: number) => {
    const curDist = camera.position.distanceTo(controls.target);
    camGoal = { x, y: sampleH(x, z), z, dist: Math.max(CAM_MIN_D, curDist * 0.6) };
  };
  let lastTap = { t: -1e9, x: 0, y: 0 };
  canvas.addEventListener('pointerdown', (e) => {
    if (e.button !== 0 && e.pointerType === 'mouse') return;
    const now = performance.now();
    if (now - lastTap.t < 300 && Math.hypot(e.offsetX - lastTap.x, e.offsetY - lastTap.y) < 12) {
      const hit = pickWorld(e.offsetX, e.offsetY);
      if (hit) focusWorld(hit.x, hit.z);
      lastTap.t = -1e9; // 消費
    } else {
      lastTap = { t: now, x: e.offsetX, y: e.offsetY };
    }
  });
  // ユーザーがカメラ操作を始めたらフォーカス/チルト介入を止める(手動を尊重)
  let tiltSuspend = 0; // >0 の間は自動チルトを止める(秒)
  controls.addEventListener('start', () => { camGoal = null; tiltSuspend = 2; });

  // ── キーボード: WASD/矢印でパン, Q/E で回転 (地面平行・速度∝ズーム距離) ──
  const keys = new Set<string>();
  addEventListener('keydown', (e) => { keys.add(e.key.toLowerCase()); });
  addEventListener('keyup', (e) => { keys.delete(e.key.toLowerCase()); });
  const _panFwd = new THREE.Vector3(), _panRight = new THREE.Vector3(), _panUp = new THREE.Vector3(0, 1, 0);
  function applyKeys(dt: number) {
    const k = keys;
    const panning = k.has('w') || k.has('a') || k.has('s') || k.has('d') ||
      k.has('arrowup') || k.has('arrowdown') || k.has('arrowleft') || k.has('arrowright');
    const rot = (k.has('q') ? 1 : 0) - (k.has('e') ? 1 : 0);
    if (!panning && rot === 0) return;
    const dist = camera.position.distanceTo(controls.target);
    if (panning) {
      // カメラ前方を地面へ射影した水平ベクトルを基準に移動
      camera.getWorldDirection(_panFwd); _panFwd.y = 0;
      if (_panFwd.lengthSq() < 1e-8) _panFwd.set(0, 0, -1);
      _panFwd.normalize();
      _panRight.crossVectors(_panFwd, _panUp).normalize().multiplyScalar(-1);
      const spd = dist * 0.8 * dt;
      const mv = new THREE.Vector3();
      if (k.has('w') || k.has('arrowup')) mv.add(_panFwd);
      if (k.has('s') || k.has('arrowdown')) mv.sub(_panFwd);
      if (k.has('d') || k.has('arrowright')) mv.add(_panRight);
      if (k.has('a') || k.has('arrowleft')) mv.sub(_panRight);
      if (mv.lengthSq() > 0) { mv.normalize().multiplyScalar(spd); camera.position.add(mv); controls.target.add(mv); }
    }
    if (rot !== 0) {
      // target 周りに水平回転(azimuth)
      const off = camera.position.clone().sub(controls.target);
      const ang = rot * 1.2 * dt;
      const cs = Math.cos(ang), sn = Math.sin(ang);
      const nx = off.x * cs - off.z * sn, nz = off.x * sn + off.z * cs;
      off.x = nx; off.z = nz;
      camera.position.copy(controls.target).add(off);
    }
    camGoal = null; tiltSuspend = 2;
  }

  // ── シム稼働の間引き (静止時は止めて軽量) ──
  let settle = 0;         // >0 の間はシムを回す
  let everRained = false; // 一度でも雨→水を回し続ける

  function applyBrush(dt: number) {
    if (!sculpting || !pointer.has) return;
    const hit = pickWorld(pointer.x, pointer.y);
    if (!hit) return;
    simBrush((hit.x + worldW / 2) / worldW, 0.5 - hit.z / worldW, Math.min(dt, 1 / 30), mode);
    settle = 60;
  }

  let raining = false;
  const btnLook = document.getElementById('look')!;
  const btnDig = document.getElementById('mode-dig')!;
  const btnRaise = document.getElementById('mode-raise')!;
  const btnRain = document.getElementById('rain')!;
  const btnReset = document.getElementById('reset')!;
  const setMode = (m: 'dig' | 'raise') => { mode = m; btnDig.classList.toggle('on', m === 'dig'); btnRaise.classList.toggle('on', m === 'raise'); };
  const refreshLookBtn = () => { btnLook.textContent = sculptOn ? '✏️ 彫るモード' : '🖐 見るモード'; btnLook.classList.toggle('on', sculptOn); };
  btnLook.addEventListener('click', () => { sculptOn = !sculptOn; applyControlMode(); refreshLookBtn(); });
  btnDig.addEventListener('click', () => { if (!sculptOn) { sculptOn = true; applyControlMode(); refreshLookBtn(); } setMode('dig'); });
  btnRaise.addEventListener('click', () => { if (!sculptOn) { sculptOn = true; applyControlMode(); refreshLookBtn(); } setMode('raise'); });
  btnRain.addEventListener('click', () => {
    raining = !raining; simRain(raining ? 0.1 : 0);
    if (raining) { everRained = true; } else { settle = 1800; }
    btnRain.textContent = `🌧 雨 ${raining ? 'ON' : 'OFF'}`; btnRain.classList.toggle('on', raining);
  });
  btnReset.addEventListener('click', () => {
    if (USE_GPU) { gpuSim!.reset(); } else {
      cpuSim!.soil.fill(0); cpuSim!.water.fill(0); cpuSim!.wet.fill(0);
      if (cpuInitBed) cpuSim!.bedrock.set(cpuInitBed);
      (cpuSim as unknown as { terrainDirty: boolean }).terrainDirty = true;
      cpuSim!.drained = 0; cpuSim!.injected = 0;
    }
    raining = false; simRain(0); everRained = false; settle = 2;
    btnRain.textContent = '🌧 雨 OFF'; btnRain.classList.remove('on');
  });
  refreshLookBtn();

  // ── 場所セレクタ ──
  const locSel = document.getElementById('loc-select') as HTMLSelectElement | null;
  if (locSel) {
    locSel.innerHTML = '';
    for (const l of LOCATIONS) {
      const opt = document.createElement('option');
      opt.value = l.key; opt.textContent = l.name_ja;
      if (l.key === LOC.key) opt.selected = true;
      locSel.appendChild(opt);
    }
    locSel.addEventListener('change', () => {
      const p = new URLSearchParams(location.search);
      p.set('loc', locSel.value);
      location.search = p.toString(); // 再読み込みで新しい場所をロード
    });
  }

  // ── 創世モード (村システム) ──
  const sampleCpuField = (arr: Float32Array, n: number, u: number, v: number) => {
    const fx = Math.min(Math.max(u, 0), 1) * (n - 1), fy = Math.min(Math.max(v, 0), 1) * (n - 1);
    const i0 = Math.floor(fx), j0 = Math.floor(fy), i1 = Math.min(i0 + 1, n - 1), j1 = Math.min(j0 + 1, n - 1);
    const tx = fx - i0, ty = fy - j0;
    const gg = (i: number, j: number) => arr[j * n + i];
    const a = gg(i0, j0) * (1 - tx) + gg(i1, j0) * tx, b = gg(i0, j1) * (1 - tx) + gg(i1, j1) * tx;
    return a * (1 - ty) + b * ty;
  };
  const worldSensor: WorldSensor = {
    worldW, vertExag: VERT_EXAG, elevMin: hMin, elevMax: hMax,
    heightUV: (u, v) => simHeightUV(u, v), // 変形追従(彫った地形に村が追従)

    waterUV: USE_GPU ? (u, v) => gpuSim!.sampleCoarse(u, v, 1) : (u, v) => sampleCpuField(cpuSim!.water, SIM_N, u, v),
    wetUV: USE_GPU ? (u, v) => gpuSim!.sampleCoarse(u, v, 2) : (u, v) => sampleCpuField(cpuSim!.wet, SIM_N, u, v),
  };
  let village: VillageSystem | null = null;
  let gameMode = false;
  let coarseTimer = 0;
  const btnGame = document.getElementById('game')!;
  const btnSeed = document.getElementById('seed')!;
  const civEl = document.getElementById('civ')!;
  const ensureVillage = () => { if (!village) { village = new VillageSystem(worldSensor, { maxPeople: IS_TOUCH ? 18 : 44 }); scene.add(village.group); } };
  btnGame.addEventListener('click', () => {
    gameMode = !gameMode;
    btnGame.classList.toggle('on', gameMode);
    btnSeed.style.display = gameMode ? '' : 'none';
    civEl.style.display = gameMode ? '' : 'none';
    if (gameMode) { ensureVillage(); if (USE_GPU) gpuSim!.readCoarse().then(() => { coarseReady = true; }); }
  });
  const seedAtScreen = (px: number, py: number) => {
    ensureVillage();
    const hit = pickWorld(px, py);
    if (hit) village!.seed((hit.x + worldW / 2) / worldW, 0.5 - hit.z / worldW);
  };
  btnSeed.addEventListener('click', () => seedAtScreen(innerWidth / 2, innerHeight / 2));

  const simN = USE_GPU ? DEM_SIZE : SIM_N;
  (window as unknown as { __geo: unknown }).__geo = {
    gpuSim, cpuSim, useGpu: USE_GPU,
    seed: (u: number, v: number) => { ensureVillage(); gameMode = true; village!.seed(u, v); },
    village: () => village,
    // ヘッドレス検証用: 村ロジックを手動で進める (rAF停止中でも動く)
    async tick(seconds = 6, steps = 360) {
      ensureVillage(); gameMode = true;
      if (USE_GPU) { await gpuSim!.readCoarse(); coarseReady = true; }
      const dt = seconds / steps;
      for (let k = 0; k < steps; k++) village!.update(dt);
      await renderer.renderAsync(scene, camera);
      return village!.stats();
    },
    async render() { await renderer.renderAsync(scene, camera); },
    async brushAt(x: number, z: number, sec: number, m: 'dig' | 'raise' = 'dig') {
      const u = (x + worldW / 2) / worldW, v = 0.5 - z / worldW, f = Math.round(sec * 60);
      for (let k = 0; k < f; k++) { simBrush(u, v, 1 / 60, m); if (USE_GPU) gpuSim!.step(0, true); else { cpuSim!.step(1 / 60, 0); cpuSim!.sync(); } }
      await renderer.renderAsync(scene, camera);
    },
    async rainFor(sec: number, rate = 0.1) {
      simRain(rate); const f = Math.round(sec * 60);
      for (let k = 0; k < f; k++) { if (USE_GPU) gpuSim!.step(1, true); else { cpuSim!.step(1 / 60, 1); cpuSim!.sync(); } }
      simRain(0); await renderer.renderAsync(scene, camera);
    },
    async totals() { return USE_GPU ? await gpuSim!.totals() : cpuSim!.totals(); },
    // 決定論再シード(A/B比較・回帰用)。以降の村乱数列が固定される。
    seedRng(n: number) { ensureVillage(); village!.reseed(n); },
    // cap非依存検証用: 独立した村システム(sceneに追加しない)を生成
    _mkSim(maxPeople: number) { return new VillageSystem(worldSensor, { maxPeople }); },
    // カメラ検証用
    camera: {
      focusOn(u: number, v: number) { focusWorld(u * worldW - worldW / 2, worldW / 2 - v * worldW); },
      step(dt = 1 / 60, n = 1) { for (let k = 0; k < n; k++) stepCamera(dt); },
      setPos(x: number, y: number, z: number) { camera.position.set(x, y, z); },
      pressKey(key: string) { keys.add(key.toLowerCase()); },
      releaseKeys() { keys.clear(); },
      targetPos() { return { x: controls.target.x, y: controls.target.y, z: controls.target.z }; },
      camPos() { return { x: camera.position.x, y: camera.position.y, z: camera.position.z }; },
      dist() { return camera.position.distanceTo(controls.target); },
      groundAt(x: number, z: number) { return sampleH(x, z); },
    },
    info: {
      worldW, hMin, hMax, cell: worldW / simN, isTouch: IS_TOUCH, simN, useGpu: USE_GPU,
      get calls() { return renderer.info.render.calls; },
    },
  };

  document.getElementById('hud')!.querySelector('b')!.textContent = `terra-touch — ${LOC.name_ja}`;
  const fpsEl = document.getElementById('fps')!;
  const drift = document.getElementById('drift')!;
  status.textContent = `${(worldW / 1000).toFixed(0)}km四方 · 標高${hMin.toFixed(0)}〜${hMax.toFixed(0)}m · ${IS_TOUCH ? 'スマホ' : 'PC'}`;

  // カメラ更新1フレーム分: キーボード → フォーカス補間 → ズーム連動チルト → controls.update → めり込みクランプ。
  // loop と __geo.stepCamera(ヘッドレス検証) の両方から呼ぶ。
  function stepCamera(dt: number) {
    applyKeys(dt);
    if (camGoal) {
      const a = 1 - Math.exp(-6 * dt);
      controls.target.x += (camGoal.x - controls.target.x) * a;
      controls.target.y += (camGoal.y - controls.target.y) * a;
      controls.target.z += (camGoal.z - controls.target.z) * a;
      const off = camera.position.clone().sub(controls.target);
      const d = off.length();
      off.setLength(d + (camGoal.dist - d) * a);
      camera.position.copy(controls.target).add(off);
      if (Math.hypot(camGoal.x - controls.target.x, camGoal.y - controls.target.y, camGoal.z - controls.target.z) < worldW * 0.001) camGoal = null;
    }
    if (tiltSuspend > 0) { tiltSuspend -= dt; }
    else {
      const off = camera.position.clone().sub(controls.target);
      const r = off.length();
      const f = Math.min(1, Math.max(0, (r - CAM_MIN_D) / (CAM_MAX_D * 0.5)));
      const goalPolar = THREE.MathUtils.degToRad(45 + 25 * f);
      const curPolar = Math.acos(Math.min(1, Math.max(-1, off.y / r)));
      const np = curPolar + (goalPolar - curPolar) * (1 - Math.exp(-1.5 * dt));
      const azim = Math.atan2(off.x, off.z), sinP = Math.sin(np);
      off.set(r * sinP * Math.sin(azim), r * Math.cos(np), r * sinP * Math.cos(azim));
      camera.position.copy(controls.target).add(off);
    }
    controls.update();
    const g = sampleH(camera.position.x, camera.position.z) + worldW * 0.006;
    if (camera.position.y < g) camera.position.y += (g - camera.position.y) * 0.3;
  }

  let frames = 0, last = performance.now(), statLast = last;
  renderer.setAnimationLoop(() => {
    const now = performance.now();
    const dt = Math.min((now - last) / 1000, 1 / 20);
    last = now; uTime.value += dt;

    const active = sculpting || raining || settle > 0;
    if (active) {
      applyBrush(dt);
      if (USE_GPU) {
        gpuSim!.step((raining || everRained) ? 1 : 0, true);
      } else {
        cpuSim!.step(dt, (raining || everRained) ? 1 : 0);
        cpuSim!.sync();
      }
      if (!sculpting && !raining && settle > 0) {
        settle--;
        // 沈静化した瞬間に coarse を1発読み戻す=彫った地形に pick/カメラ/村が追従
        if (settle === 0 && USE_GPU) gpuSim!.readCoarse().then(() => { coarseReady = true; });
      }
    }

    // ── 創世モード: 村ロジック更新 ──
    if (gameMode && village) {
      village.update(Math.min(dt, 1 / 20));
      if (USE_GPU) { coarseTimer += dt; if (coarseTimer > 0.5) { coarseTimer = 0; gpuSim!.readCoarse().then(() => { coarseReady = true; }); } }
    }

    stepCamera(dt);
    renderer.render(scene, camera);
    frames++;
    if (now - statLast >= 500) {
      fpsEl.textContent = `${Math.round((frames * 1000) / (now - statLast))} fps`;
      if (USE_GPU) {
        drift.textContent = `GPU ${simN}²`;
      } else {
        const t = cpuSim!.totals();
        drift.textContent = `土drift ${t.driftPct >= 0 ? '+' : ''}${t.driftPct.toFixed(2)}%${t.nan ? ' ⚠NaN' : ''}`;
      }
      if (gameMode && village) {
        const s = village.stats();
        civEl.textContent = `⏳${Math.floor(s.year)}y 🏘${s.villages} 👥${s.pop}${s.bands ? ` (移動中${s.bands})` : ''} 👶${s.kids} 🧑${s.adults} 👴${s.elders}`;
      }
      frames = 0; statLast = now;
    }
  });

  addEventListener('resize', () => {
    camera.aspect = innerWidth / innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(innerWidth, innerHeight);
  });
}

main().catch((e) => { status.textContent = `失敗: ${e?.message ?? e}`; console.error(e); });
