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

  const factor = DEM_SIZE / SIM_N;
  const simBed = new Float32Array(SIM_N * SIM_N);
  for (let j = 0; j < SIM_N; j++) for (let i = 0; i < SIM_N; i++) {
    let acc = 0;
    for (let bj = 0; bj < factor; bj++) { const sj = j * factor + bj; for (let bi = 0; bi < factor; bi++) acc += demH[sj * DEM_SIZE + (i * factor + bi)]; }
    simBed[j * SIM_N + i] = acc / (factor * factor);
  }

  const sim = new TerrainSim({ n: SIM_N, world: worldW, bedrock: simBed });

  const toHalf = THREE.DataUtils.toHalfFloat;
  const baseHalf = new Uint16Array(demH.length);
  for (let k = 0; k < demH.length; k++) baseHalf[k] = toHalf(demH[k]);
  const baseDemTex = new THREE.DataTexture(baseHalf, DEM_SIZE, DEM_SIZE, THREE.RedFormat, THREE.HalfFloatType);
  baseDemTex.minFilter = baseDemTex.magFilter = THREE.LinearFilter;
  baseDemTex.wrapS = baseDemTex.wrapT = THREE.ClampToEdgeWrapping; baseDemTex.needsUpdate = true;

  const simBaseHalf = new Uint16Array(simBed.length);
  for (let k = 0; k < simBed.length; k++) simBaseHalf[k] = toHalf(simBed[k]);
  const simBaseTex = new THREE.DataTexture(simBaseHalf, SIM_N, SIM_N, THREE.RedFormat, THREE.HalfFloatType);
  simBaseTex.minFilter = simBaseTex.magFilter = THREE.LinearFilter;
  simBaseTex.wrapS = simBaseTex.wrapT = THREE.ClampToEdgeWrapping; simBaseTex.needsUpdate = true;

  const simTex = sim.tex;

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
  controls.maxPolarAngle = Math.PI * 0.495;
  controls.minDistance = worldW * 0.02; controls.maxDistance = worldW * 2.5;

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
  const baseAt = (ox: number, oy: number) => texture(baseDemTex, uv().add(vec2(ox * demTexel, oy * demTexel))).r;
  const simSolidAt = (ox: number, oy: number) => { const s = texture(simTex, uv().add(vec2(ox * demTexel, oy * demTexel))); return s.r.add(s.g); };
  const simBaseAt = (ox: number, oy: number) => texture(simBaseTex, uv().add(vec2(ox * demTexel, oy * demTexel))).r;
  const groundAt = (ox: number, oy: number) => baseAt(ox, oy).add(simSolidAt(ox, oy)).sub(simBaseAt(ox, oy)).mul(VERT_EXAG);

  const geo = new THREE.PlaneGeometry(worldW, worldW, MESH_N - 1, MESH_N - 1);
  geo.rotateX(-Math.PI / 2);
  const mat = new THREE.MeshStandardNodeMaterial({ roughness: 0.95, metalness: 0 });
  mat.positionNode = positionLocal.add(vec3(0, groundAt(0, 0), 0));
  const nLocal = nrm(vec3(groundAt(-1, 0).sub(groundAt(1, 0)), cell * 2, groundAt(0, 1).sub(groundAt(0, -1))));
  mat.normalNode = transformNormalToView(nLocal);
  const shade = mix(float(0.72), float(1.0), smoothstep(0.3, 0.95, nLocal.y));
  const wetA = texture(simTex, uv()).a;
  const photo = texture(imgTex, uv()).mul(shade);
  mat.colorNode = mix(photo, photo.mul(0.7), smoothstep(0.2, 0.9, wetA)); // 濡れ暗転は控えめに
  const terrain = new THREE.Mesh(geo, mat);
  scene.add(terrain);

  const wgeo = new THREE.PlaneGeometry(worldW, worldW, WATER_MESH - 1, WATER_MESH - 1);
  wgeo.rotateX(-Math.PI / 2);
  const wmat = new THREE.MeshStandardNodeMaterial({ metalness: 0, roughness: 0.24 });
  wmat.transparent = true; wmat.depthWrite = false;
  wmat.polygonOffset = true; wmat.polygonOffsetFactor = -2; wmat.polygonOffsetUnits = -2; // z-fight緩和
  const depth = texture(simTex, uv()).b;
  const surfAt = (ox: number, oy: number) => groundAt(ox, oy).add(texture(simTex, uv().add(vec2(ox * demTexel, oy * demTexel))).b.mul(VERT_EXAG));
  wmat.positionNode = positionLocal.add(vec3(0, surfAt(0, 0), 0));

  // ── 流速(m/s)で水面を「流れて」見せる ──
  const vel = texture(sim.velTex, uv()).xy;
  const spd = vel.length();
  const flowDir = nrm(vel.add(vec2(1e-4, 1e-4)));
  // 流れ方向へ進む筋 (速い所ほど筋が立ち速く流れる)
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

  const raycaster = new THREE.Raycaster();
  const ndc = new THREE.Vector2();
  let sculpting = false;
  let mode: 'dig' | 'raise' = 'dig';
  const pointer = { x: 0, y: 0, has: false };
  const sampleH = (x: number, z: number) => sim.surfaceHeightUV((x + worldW / 2) / worldW, 0.5 - z / worldW) * VERT_EXAG;
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

  // ── シム稼働の間引き (静止時は止めて軽量) ──
  let settle = 0;         // >0 の間はシムを回す
  let everRained = false; // 一度でも雨→水を回し続ける

  function applyBrush(dt: number) {
    if (!sculpting || !pointer.has) return;
    const hit = pickWorld(pointer.x, pointer.y);
    if (!hit) return;
    sim.brush((hit.x + worldW / 2) / worldW, 0.5 - hit.z / worldW, Math.min(dt, 1 / 30), mode);
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
    raining = !raining; sim.rainRate = raining ? 0.1 : 0;
    if (raining) { everRained = true; } else { settle = 1800; }
    btnRain.textContent = `🌧 雨 ${raining ? 'ON' : 'OFF'}`; btnRain.classList.toggle('on', raining);
  });
  btnReset.addEventListener('click', () => {
    sim.soil.fill(0); sim.water.fill(0); sim.wet.fill(0); sim.bedrock.set(simBed);
    (sim as unknown as { terrainDirty: boolean }).terrainDirty = true;
    sim.drained = 0; sim.injected = 0; raining = false; sim.rainRate = 0; everRained = false; settle = 2;
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

  (window as unknown as { __geo: unknown }).__geo = {
    sim, async render() { await renderer.renderAsync(scene, camera); },
    async brushAt(x: number, z: number, sec: number, m: 'dig' | 'raise' = 'dig') {
      const u = (x + worldW / 2) / worldW, v = 0.5 - z / worldW, f = Math.round(sec * 60);
      for (let k = 0; k < f; k++) { sim.brush(u, v, 1 / 60, m); sim.step(1 / 60, 0); }
      sim.sync(); await renderer.renderAsync(scene, camera);
    },
    async rainFor(sec: number, rate = 0.12) {
      sim.rainRate = rate; const f = Math.round(sec * 60);
      for (let k = 0; k < f; k++) sim.step(1 / 60, 1);
      sim.rainRate = 0; sim.sync(); await renderer.renderAsync(scene, camera);
    },
    totals: () => sim.totals(),
    info: { worldW, hMin, hMax, cell: worldW / SIM_N, isTouch: IS_TOUCH, simN: SIM_N },
  };

  document.getElementById('hud')!.querySelector('b')!.textContent = `terra-touch — ${LOC.name_ja}`;
  const fpsEl = document.getElementById('fps')!;
  const drift = document.getElementById('drift')!;
  status.textContent = `${(worldW / 1000).toFixed(0)}km四方 · 標高${hMin.toFixed(0)}〜${hMax.toFixed(0)}m · ${IS_TOUCH ? 'スマホ' : 'PC'}`;

  let frames = 0, last = performance.now(), statLast = last;
  renderer.setAnimationLoop(() => {
    const now = performance.now();
    const dt = Math.min((now - last) / 1000, 1 / 20);
    last = now; uTime.value += dt;

    const active = sculpting || raining || settle > 0;
    if (active) {
      applyBrush(dt);
      sim.step(dt, (raining || everRained) ? 1 : 0);
      sim.sync();
      if (!sculpting && !raining && settle > 0) settle--;
    }

    controls.update();
    renderer.render(scene, camera);
    frames++;
    if (now - statLast >= 500) {
      fpsEl.textContent = `${Math.round((frames * 1000) / (now - statLast))} fps`;
      const t = sim.totals();
      drift.textContent = `土drift ${t.driftPct >= 0 ? '+' : ''}${t.driftPct.toFixed(2)}%${t.nan ? ' ⚠NaN' : ''}`;
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
