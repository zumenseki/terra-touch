import * as THREE from 'three/webgpu';
import {
  texture, uv, vec2, vec3, float, mix, smoothstep,
  normalize as nrm, positionLocal, transformNormalToView, uniform, sin,
} from 'three/tsl';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { SkyMesh } from 'three/addons/objects/SkyMesh.js';
import { TerrainSim, SIM_N, WORLD, HEIGHT_M } from './sim';

const MESH_N = 1024;       // 地形描画メッシュ解像度
const WATER_MESH_N = 512;  // 水面メッシュ解像度
const WATER_ITERS = 1;     // 水サブステップ/フレーム (192²で安定・~40fps 実測)

async function main() {
  const renderer = new THREE.WebGPURenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setSize(innerWidth, innerHeight);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 0.85;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  document.body.appendChild(renderer.domElement);
  await renderer.init();

  const backendEl = document.getElementById('backend')!;
  backendEl.textContent = (renderer.backend as { isWebGPUBackend?: boolean }).isWebGPUBackend
    ? 'WebGPU' : 'WebGL2 fallback';

  const scene = new THREE.Scene();
  scene.fog = new THREE.Fog(0xbcd0e5, 400, 1700);

  const camera = new THREE.PerspectiveCamera(55, innerWidth / innerHeight, 0.5, 4000);
  camera.position.set(230, 150, 230);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 25, 0);
  controls.enableDamping = true;
  controls.maxPolarAngle = Math.PI * 0.49;
  controls.minDistance = 30;
  controls.maxDistance = 1000;
  // 左ドラッグ / 1本指 を彫刻用にアプリへ譲る (three r185: null は switch default で安全に無視)
  controls.mouseButtons = { LEFT: null, MIDDLE: THREE.MOUSE.PAN, RIGHT: THREE.MOUSE.ROTATE };
  controls.touches = { ONE: null, TWO: THREE.TOUCH.DOLLY_PAN };

  // ── 空と太陽 ──
  const sun = new THREE.Vector3();
  const elevation = THREE.MathUtils.degToRad(32);
  const azimuth = THREE.MathUtils.degToRad(135);
  sun.setFromSphericalCoords(1, Math.PI / 2 - elevation, azimuth);

  const sky = new SkyMesh();
  sky.scale.setScalar(3000);
  sky.turbidity.value = 4;
  sky.rayleigh.value = 1.3;
  sky.mieCoefficient.value = 0.004;
  sky.mieDirectionalG.value = 0.85;
  sky.sunPosition.value.copy(sun);
  scene.add(sky);

  const dirLight = new THREE.DirectionalLight(0xfff2df, 3.2);
  dirLight.position.copy(sun).multiplyScalar(800);
  dirLight.castShadow = true;
  dirLight.shadow.mapSize.set(2048, 2048);
  const sc = dirLight.shadow.camera;
  sc.left = -400; sc.right = 400; sc.top = 400; sc.bottom = -400;
  sc.near = 100; sc.far = 2200;
  dirLight.shadow.bias = -0.0004;
  dirLight.shadow.normalBias = 1.5;
  scene.add(dirLight);
  scene.add(new THREE.HemisphereLight(0x9db8d9, 0x5b4a33, 0.6));

  // ── シミュレーション ──
  const sim = new TerrainSim();
  const packedTex = sim.tex;
  const texel = 1 / SIM_N;
  const cell = WORLD / SIM_N;
  const uTime = uniform(0);

  // packedTex: r=bedrock, g=soil, b=water, a=wetness (メートル)
  const pAt = (ox: number, oy: number) => texture(packedTex, uv().add(vec2(ox * texel, oy * texel)));
  // 地表高 (r+g)
  const terrH = (ox: number, oy: number) => { const s = pAt(ox, oy); return s.r.add(s.g); };

  // ── 地形メッシュ ──
  const geo = new THREE.PlaneGeometry(WORLD, WORLD, MESH_N - 1, MESH_N - 1);
  geo.rotateX(-Math.PI / 2);

  const mat = new THREE.MeshStandardNodeMaterial({ roughness: 1.0, metalness: 0 });
  const s0 = pAt(0, 0);
  const hC = s0.r.add(s0.g);
  mat.positionNode = positionLocal.add(vec3(0, hC, 0));

  const nLocal = nrm(vec3(
    terrH(-1, 0).sub(terrH(1, 0)),
    cell * 2,
    terrH(0, 1).sub(terrH(0, -1)),
  ));
  mat.normalNode = transformNormalToView(nLocal);

  const slope = nLocal.y;
  const detail = pAt(24, 24).r.add(pAt(7, 7).r).mul(0.5).div(HEIGHT_M);
  const grass = mix(vec3(0.14, 0.20, 0.07), vec3(0.19, 0.25, 0.10), smoothstep(0.25, 0.75, detail));
  const soilRock = mix(vec3(0.33, 0.30, 0.27), vec3(0.31, 0.24, 0.16), smoothstep(0.55, 0.75, slope));
  const heightN = hC.div(HEIGHT_M);
  const snowAmt = smoothstep(0.72, 0.85, heightN).mul(smoothstep(0.6, 0.9, slope));
  const baseCol = mix(soilRock, grass, smoothstep(0.72, 0.88, slope));
  const dryCol = mix(baseCol, vec3(0.85, 0.87, 0.90), snowAmt);
  // 濡れ: wetness で暗く+彩度落ち
  const wetAmt = smoothstep(0.05, 0.8, s0.a);
  const col = mix(dryCol, dryCol.mul(0.45), wetAmt);
  mat.colorNode = col;
  mat.roughnessNode = mix(mix(float(1.0), float(0.55), snowAmt), float(0.35), wetAmt);

  const terrain = new THREE.Mesh(geo, mat);
  terrain.castShadow = true;
  terrain.receiveShadow = true;
  scene.add(terrain);

  // ── 水メッシュ ──
  const wgeo = new THREE.PlaneGeometry(WORLD, WORLD, WATER_MESH_N - 1, WATER_MESH_N - 1);
  wgeo.rotateX(-Math.PI / 2);
  const wmat = new THREE.MeshStandardNodeMaterial({ metalness: 0, roughness: 0.08 });
  wmat.transparent = true;
  wmat.depthWrite = false;

  const surfH = (ox: number, oy: number) => { const s = pAt(ox, oy); return s.r.add(s.g).add(s.b); };
  const ws = pAt(0, 0);
  const depth = ws.b;
  wmat.positionNode = positionLocal.add(vec3(0, ws.r.add(ws.g).add(depth), 0));

  // さざ波を法線に加える (水らしさ・M3 で SSR/泡へ発展)
  const rip = sin(uv().x.mul(220).add(uTime.mul(1.6))).add(sin(uv().y.mul(190).sub(uTime.mul(1.3)))).mul(0.06);
  const wN = nrm(vec3(
    surfH(-1, 0).sub(surfH(1, 0)).add(rip),
    cell * 2,
    surfH(0, 1).sub(surfH(0, -1)).add(rip),
  ));
  wmat.normalNode = transformNormalToView(wN);

  const shallow = vec3(0.10, 0.30, 0.32);
  const deep = vec3(0.02, 0.09, 0.20);
  wmat.colorNode = mix(shallow, deep, smoothstep(0.1, 2.5, depth));
  wmat.opacityNode = smoothstep(0.015, 0.25, depth).mul(0.92);

  const waterMesh = new THREE.Mesh(wgeo, wmat);
  waterMesh.renderOrder = 1;
  scene.add(waterMesh);

  // ── 入力: 左ドラッグ / 1本指で彫刻 ──
  const raycaster = new THREE.Raycaster();
  const ndc = new THREE.Vector2();
  let sculpting = false;
  const pointer = { x: 0, y: 0, has: false };

  // カメラレイを CPU ハイトフィールドへマーチ → world(x,z) を得る
  function pickWorld(px: number, py: number): { x: number; z: number } | null {
    ndc.set((px / innerWidth) * 2 - 1, -(py / innerHeight) * 2 + 1);
    raycaster.setFromCamera(ndc, camera);
    const o = raycaster.ray.origin;
    const d = raycaster.ray.direction;
    let t = 0;
    const step = 1.2;
    let prevGap = o.y - sampleH(o.x, o.z);
    for (let k = 0; k < 700; k++) {
      t += step;
      const x = o.x + d.x * t;
      const z = o.z + d.z * t;
      if (t > 1400) break;
      const gap = o.y + d.y * t - sampleH(x, z);
      if (gap <= 0 && prevGap > 0) {
        // 交差区間を二分
        let lo = t - step, hi = t;
        for (let b = 0; b < 8; b++) {
          const m = (lo + hi) * 0.5;
          const g = o.y + d.y * m - sampleH(o.x + d.x * m, o.z + d.z * m);
          if (g <= 0) hi = m; else lo = m;
        }
        const tm = (lo + hi) * 0.5;
        return { x: o.x + d.x * tm, z: o.z + d.z * tm };
      }
      prevGap = gap;
    }
    return null;
  }
  function sampleH(x: number, z: number): number {
    const u = (x + WORLD / 2) / WORLD;
    const v = 0.5 - z / WORLD;
    return sim.surfaceHeightUV(u, v);
  }

  let mode: 'dig' | 'raise' = 'dig';
  const canvas = renderer.domElement;
  canvas.addEventListener('pointerdown', (e) => {
    if (e.button !== 0 && e.pointerType === 'mouse') return; // 左ボタン/タッチのみ彫刻
    sculpting = true;
    pointer.x = e.offsetX; pointer.y = e.offsetY; pointer.has = true;
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener('pointermove', (e) => {
    pointer.x = e.offsetX; pointer.y = e.offsetY;
  });
  const endSculpt = () => { sculpting = false; };
  canvas.addEventListener('pointerup', endSculpt);
  canvas.addEventListener('pointercancel', endSculpt);

  function applyBrush(dt: number) {
    if (!sculpting || !pointer.has) return;
    const hit = pickWorld(pointer.x, pointer.y);
    if (!hit) return;
    const u = (hit.x + WORLD / 2) / WORLD;
    const v = 0.5 - hit.z / WORLD;
    sim.brush(u, v, Math.min(dt, 1 / 30), mode);
  }

  // ── ツールバー ──
  const btnDig = document.getElementById('mode-dig')!;
  const btnRaise = document.getElementById('mode-raise')!;
  const btnSrc = document.getElementById('src-toggle')!;
  const setMode = (m: 'dig' | 'raise') => {
    mode = m;
    btnDig.classList.toggle('on', m === 'dig');
    btnRaise.classList.toggle('on', m === 'raise');
  };
  btnDig.addEventListener('click', () => setMode('dig'));
  btnRaise.addEventListener('click', () => setMode('raise'));
  btnSrc.addEventListener('click', () => {
    sim.sourceOn = !sim.sourceOn;
    btnSrc.textContent = `💧 湧水 ${sim.sourceOn ? 'ON' : 'OFF'}`;
    btnSrc.classList.toggle('on', sim.sourceOn);
  });
  addEventListener('keydown', (e) => {
    if (e.key === 'd' || e.key === 'D') setMode('dig');
    if (e.key === 'r' || e.key === 'R') setMode('raise');
  });

  // ── デバッグ/検証 API (rAF 停止中も動く) ──
  (window as unknown as { __tt: unknown }).__tt = {
    sim,
    async step(nn = 1) {
      for (let k = 0; k < nn; k++) sim.step(1 / 60);
      sim.sync();
      await renderer.renderAsync(scene, camera);
    },
    async brushAt(x: number, z: number, seconds: number, m: 'dig' | 'raise' = 'dig') {
      const u = (x + WORLD / 2) / WORLD;
      const v = 0.5 - z / WORLD;
      const frames = Math.round(seconds * 60);
      for (let k = 0; k < frames; k++) { sim.brush(u, v, 1 / 60, m); sim.step(1 / 60); }
      sim.sync();
      await renderer.renderAsync(scene, camera);
    },
    // 川を横断する土手を築く (検証用): x0..x1 に沿って raise を敷く
    async damAcross(zLine: number, x0: number, x1: number, seconds: number) {
      const frames = Math.round(seconds * 60);
      const steps = 7;
      for (let k = 0; k < frames; k++) {
        for (let s = 0; s <= steps; s++) {
          const x = x0 + ((x1 - x0) * s) / steps;
          sim.brush((x + WORLD / 2) / WORLD, 0.5 - zLine / WORLD, 1 / 60 / (steps + 1), 'raise');
        }
        sim.step(1 / 60);
      }
      sim.sync();
      await renderer.renderAsync(scene, camera);
    },
    totals: () => sim.totals(),
    heightAt: (x: number, z: number) => sampleH(x, z),
    waterAt: (x: number, z: number) => {
      const u = Math.min(Math.max((x + WORLD / 2) / WORLD, 0), 1);
      const v = Math.min(Math.max(0.5 - z / WORLD, 0), 1);
      const i = Math.round(u * (SIM_N - 1));
      const j = Math.round(v * (SIM_N - 1));
      return sim.water[j * SIM_N + i];
    },
    setSource: (on: boolean) => { sim.sourceOn = on; },
    async render() { await renderer.renderAsync(scene, camera); },
  };

  // ── ループ ──
  const fpsEl = document.getElementById('fps')!;
  const soilEl = document.getElementById('soil')!;
  const waterEl = document.getElementById('water')!;
  let frames = 0;
  let last = performance.now();
  let statLast = last;

  renderer.setAnimationLoop(() => {
    const now = performance.now();
    const dt = Math.min((now - last) / 1000, 1 / 20);
    last = now;
    uTime.value += dt;

    applyBrush(dt);
    sim.step(dt, WATER_ITERS);
    sim.sync();

    controls.update();
    renderer.render(scene, camera);

    frames++;
    if (now - statLast >= 500) {
      fpsEl.textContent = `${Math.round((frames * 1000) / (now - statLast))} fps`;
      const t = sim.totals();
      soilEl.textContent = `${t.driftPct >= 0 ? '+' : ''}${t.driftPct.toFixed(2)}%`;
      soilEl.style.color = Math.abs(t.driftPct) > 5 ? '#ff6b6b' : '#8fe';
      waterEl.textContent = `${Math.round(t.water).toLocaleString()} m³`;
      if (t.nan) waterEl.textContent += ' ⚠NaN';
      frames = 0;
      statLast = now;
    }
  });

  addEventListener('resize', () => {
    camera.aspect = innerWidth / innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(innerWidth, innerHeight);
  });
}

main().catch((e) => {
  document.getElementById('backend')!.textContent = `初期化失敗: ${e?.message ?? e}`;
  console.error(e);
});
