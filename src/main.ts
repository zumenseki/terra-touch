import * as THREE from 'three/webgpu';
import {
  texture, uv, vec2, vec3, float, mix, smoothstep,
  normalize as nrm, positionLocal, transformNormalToView,
} from 'three/tsl';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { SkyMesh } from 'three/addons/objects/SkyMesh.js';
import { fbm, ridged } from './noise';

// ── 世界定数 (SPEC.md 準拠) ──────────────────────────────
const N = 1024;      // ハイトフィールド解像度
const WORLD = 600;   // 一辺 (m)
const HEIGHT = 90;   // 最大標高 (m)

// ── 地形生成: fbm 丘陵 + ridged 尾根 + 蛇行谷 (M2 の川床) ──
function generateHeights(): Float32Array {
  const h = new Float32Array(N * N);
  for (let j = 0; j < N; j++) {
    const z = (j / N) * WORLD;
    // 谷の中心線: 蛇行カーブ (M2 でここに川を流す)
    const riverX = WORLD * 0.5 + Math.sin(z * 0.012) * 70 + Math.sin(z * 0.031) * 22;
    for (let i = 0; i < N; i++) {
      const x = (i / N) * WORLD;
      const base = fbm(x * 0.004, z * 0.004);
      const mount = ridged(x * 0.008, z * 0.008);
      let v = base * 0.45 + mount * 0.55 * base;
      const d = Math.abs(x - riverX);
      const valley = 1 - Math.exp(-(d * d) / (2 * 45 * 45)); // 0=谷底 1=谷外
      v *= 0.22 + 0.78 * valley;
      h[j * N + i] = v;
    }
  }
  return h;
}

function makeHeightTexture(heights: Float32Array): THREE.DataTexture {
  // r16float は WebGPU 標準でフィルタ可 (r32float は要拡張のため fp16 を採用)
  const half = new Uint16Array(heights.length);
  for (let k = 0; k < heights.length; k++) half[k] = THREE.DataUtils.toHalfFloat(heights[k]);
  const tex = new THREE.DataTexture(half, N, N, THREE.RedFormat, THREE.HalfFloatType);
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.wrapS = THREE.RepeatWrapping; // uv*24 の色ムラ流用のため
  tex.wrapT = THREE.RepeatWrapping;
  tex.needsUpdate = true;
  return tex;
}

async function main() {
  const renderer = new THREE.WebGPURenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setSize(innerWidth, innerHeight);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 0.8;
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

  // ── 地形: ハイトフィールドテクスチャ + TSL 頂点変位 ──
  // (M1 でこのテクスチャを compute で書き換える前提の構成)
  const heights = generateHeights();
  const heightTex = makeHeightTexture(heights);

  const geo = new THREE.PlaneGeometry(WORLD, WORLD, N - 1, N - 1);
  geo.rotateX(-Math.PI / 2); // XZ 平面・+Y 上

  const texel = 1 / N;
  const cell = WORLD / N;
  const sampleH = (ox: number, oy: number) =>
    texture(heightTex, uv().add(vec2(ox * texel, oy * texel))).r;

  const hC = sampleH(0, 0);
  const hL = sampleH(-1, 0);
  const hR = sampleH(1, 0);
  const hD = sampleH(0, -1); // v-texel = ワールド +z 側
  const hU = sampleH(0, 1);

  const mat = new THREE.MeshStandardNodeMaterial({ roughness: 1.0, metalness: 0 });
  mat.positionNode = positionLocal.add(vec3(0, hC.mul(HEIGHT), 0));

  // 法線: 隣接テクセルの高低差から (rotateX 済のため v は -z 方向に増加)
  const nLocal = nrm(vec3(
    hL.sub(hR).mul(HEIGHT),
    cell * 2,
    hU.sub(hD).mul(HEIGHT),
  ));
  mat.normalNode = transformNormalToView(nLocal);

  // 色: 勾配と高度で 草/土/岩/雪 をスプラット (M3 で実写テクスチャに差し替え)
  const slope = nLocal.y; // 1=平坦
  // 高周波の色ムラ: 2スケール平均で等高線状の縞を抑える
  const detail = texture(heightTex, uv().mul(24)).r.add(texture(heightTex, uv().mul(7.3)).r).mul(0.5);
  const grass = mix(vec3(0.14, 0.20, 0.07), vec3(0.19, 0.25, 0.10), smoothstep(0.25, 0.75, detail));
  const soilRock = mix(vec3(0.33, 0.30, 0.27), vec3(0.31, 0.24, 0.16), smoothstep(0.55, 0.75, slope));
  const snowAmt = smoothstep(0.72, 0.85, hC).mul(smoothstep(0.6, 0.9, slope));
  const baseCol = mix(soilRock, grass, smoothstep(0.72, 0.88, slope));
  mat.colorNode = mix(baseCol, vec3(0.85, 0.87, 0.90), snowAmt);
  mat.roughnessNode = mix(float(1.0), float(0.55), snowAmt);

  const terrain = new THREE.Mesh(geo, mat);
  terrain.castShadow = true;
  terrain.receiveShadow = true;
  scene.add(terrain);

  // ── ループ + fps 計測 ──
  const fpsEl = document.getElementById('fps')!;
  let frames = 0;
  let last = performance.now();
  renderer.setAnimationLoop(() => {
    controls.update();
    renderer.render(scene, camera);
    frames++;
    const now = performance.now();
    if (now - last >= 500) {
      fpsEl.textContent = `${Math.round((frames * 1000) / (now - last))} fps`;
      frames = 0;
      last = now;
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
