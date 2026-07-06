// terra-touch geo POC — 実衛星画像 × 実標高DEM を読み込んで表示する。
// これが「Google Earth を指でいじる」の土台: 本物の写真を本物の地形に貼る。
// 変形シム (brush/water) は次段でこの実 DEM ハイトフィールド上に載せる。
//
// データ源 (どちらも API キー不要 / CORS 対応):
//   衛星画像 = ESRI World Imagery (© Esri, Maxar)   z/y/x
//   標高DEM  = AWS Terrain Tiles / terrarium PNG      z/x/y  (h = R*256 + G + B/256 - 32768 m)

import * as THREE from 'three/webgpu';
import {
  texture, uv, vec2, vec3, float, mix, smoothstep,
  normalize as nrm, positionLocal, transformNormalToView,
} from 'three/tsl';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { SkyMesh } from 'three/addons/objects/SkyMesh.js';

// ── 表示する場所 (富士山) ──
const LAT = 35.3606;
const LON = 138.7274;
const ZOOM = 12;
const GRID = 6;        // GRID×GRID タイル (z12 で約48km四方)
const TILE = 256;
const VERT_EXAG = 1.35; // 立体感を少し強調 (1.0=実スケール)

const EARTH_C = 40075016.686; // 赤道円周 m
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

/** GRID×GRID のタイルを取得し 1枚のキャンバスに合成 */
async function fetchMosaic(kind: 'img' | 'dem', xMin: number, yMin: number) {
  const size = GRID * TILE;
  const canvas = document.createElement('canvas');
  canvas.width = size; canvas.height = size;
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
      jobs.push(
        fetchBitmap(url).then((bmp) => {
          ctx.drawImage(bmp, dx * TILE, dy * TILE);
          bmp.close();
          done++;
          status.textContent = `${kind === 'img' ? '衛星画像' : '標高'} ${done}/${total} タイル…`;
        }),
      );
    }
  }
  await Promise.all(jobs);
  return { canvas, ctx, size };
}

async function main() {
  const renderer = new THREE.WebGPURenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setSize(innerWidth, innerHeight);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.0;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  document.body.appendChild(renderer.domElement);
  await renderer.init();

  // タイル範囲 (中心を GRID の中央へ)
  const cx = lon2tile(LON, ZOOM);
  const cy = lat2tile(LAT, ZOOM);
  const xMin = Math.floor(cx - GRID / 2);
  const yMin = Math.floor(cy - GRID / 2);

  // 衛星画像と DEM を並行取得
  const [img, dem] = await Promise.all([fetchMosaic('img', xMin, yMin), fetchMosaic('dem', xMin, yMin)]);
  status.textContent = '地形を構築中…';

  // DEM 復号 (terrarium) → メートル標高
  const { ctx: dctx, size } = dem;
  const px = dctx.getImageData(0, 0, size, size).data;
  const heights = new Float32Array(size * size);
  let hMin = Infinity, hMax = -Infinity;
  for (let k = 0, p = 0; k < size * size; k++, p += 4) {
    const h = px[p] * 256 + px[p + 1] + px[p + 2] / 256 - 32768;
    heights[k] = h;
    if (h < hMin) hMin = h;
    if (h > hMax) hMax = h;
  }

  // ワールド寸法 (web mercator メートル → cos(lat) で地上メートルへ)
  const latR = (LAT * Math.PI) / 180;
  const mercTile = EARTH_C / 2 ** ZOOM;
  const worldW = mercTile * GRID * Math.cos(latR); // 一辺 m

  // 標高テクスチャ (fp16・メートル)
  const half = new Uint16Array(size * size);
  for (let k = 0; k < heights.length; k++) half[k] = THREE.DataUtils.toHalfFloat(heights[k]);
  const heightTex = new THREE.DataTexture(half, size, size, THREE.RedFormat, THREE.HalfFloatType);
  heightTex.minFilter = THREE.LinearFilter;
  heightTex.magFilter = THREE.LinearFilter;
  heightTex.needsUpdate = true;

  // 衛星画像テクスチャ
  const imgTex = new THREE.CanvasTexture(img.canvas);
  imgTex.colorSpace = THREE.SRGBColorSpace;
  imgTex.minFilter = THREE.LinearMipmapLinearFilter;
  imgTex.magFilter = THREE.LinearFilter;
  imgTex.generateMipmaps = true;
  imgTex.anisotropy = renderer.getMaxAnisotropy?.() ?? 8;

  // ── シーン ──
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x9fb8d4);
  scene.fog = new THREE.Fog(0xaec4dd, worldW * 0.6, worldW * 2.2);

  const camera = new THREE.PerspectiveCamera(52, innerWidth / innerHeight, 1, worldW * 6);
  camera.position.set(worldW * 0.42, (hMax - hMin) * VERT_EXAG + worldW * 0.3, worldW * 0.55);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, (hMax * 0.4 - hMin) * VERT_EXAG, 0);
  controls.enableDamping = true;
  controls.maxPolarAngle = Math.PI * 0.495;
  controls.minDistance = worldW * 0.03;
  controls.maxDistance = worldW * 2.5;

  // 太陽 (南寄り・朝方の斜光で陰影を立てる)
  const sun = new THREE.Vector3().setFromSphericalCoords(
    1, Math.PI / 2 - THREE.MathUtils.degToRad(34), THREE.MathUtils.degToRad(150),
  );
  const sky = new SkyMesh();
  sky.scale.setScalar(worldW * 4);
  sky.turbidity.value = 3.5;
  sky.rayleigh.value = 1.1;
  sky.mieCoefficient.value = 0.004;
  sky.mieDirectionalG.value = 0.85;
  sky.sunPosition.value.copy(sun);
  scene.add(sky);

  const dir = new THREE.DirectionalLight(0xfff4e6, 3.0);
  dir.position.copy(sun).multiplyScalar(worldW);
  dir.castShadow = true;
  dir.shadow.mapSize.set(4096, 4096);
  const s = dir.shadow.camera;
  const r = worldW * 0.75;
  s.left = -r; s.right = r; s.top = r; s.bottom = -r;
  s.near = worldW * 0.1; s.far = worldW * 3;
  dir.shadow.bias = -0.0003;
  dir.shadow.normalBias = worldW * 0.002;
  scene.add(dir);
  scene.add(new THREE.HemisphereLight(0xbcd2ee, 0x4a4238, 0.55));

  // ── 地形メッシュ (実DEM変位 + 実画像ドレープ) ──
  const MESH = 1024;
  const geo = new THREE.PlaneGeometry(worldW, worldW, MESH - 1, MESH - 1);
  geo.rotateX(-Math.PI / 2);

  const texel = 1 / size;
  const cell = worldW / size;
  const hAt = (ox: number, oy: number) => texture(heightTex, uv().add(vec2(ox * texel, oy * texel))).r.mul(VERT_EXAG);

  const mat = new THREE.MeshStandardNodeMaterial({ roughness: 0.95, metalness: 0 });
  mat.positionNode = positionLocal.add(vec3(0, hAt(0, 0), 0));
  const nLocal = nrm(vec3(
    hAt(-1, 0).sub(hAt(1, 0)),
    cell * 2,
    hAt(0, 1).sub(hAt(0, -1)),
  ));
  mat.normalNode = transformNormalToView(nLocal);
  // 実衛星画像に斜面陰影を軽く掛けて立体感を補強
  const shade = mix(float(0.7), float(1.0), smoothstep(0.35, 0.95, nLocal.y));
  mat.colorNode = texture(imgTex, uv()).mul(shade);

  const terrain = new THREE.Mesh(geo, mat);
  terrain.castShadow = true;
  terrain.receiveShadow = true;
  scene.add(terrain);

  status.textContent = `${(worldW / 1000).toFixed(0)}km四方 · 標高 ${hMin.toFixed(0)}〜${hMax.toFixed(0)}m · 実衛星画像`;
  (document.getElementById('hud') as HTMLElement).querySelector('b')!.textContent =
    'terra-touch geo — 富士山 (実写)';

  (window as unknown as { __geo: unknown }).__geo = {
    async render() { await renderer.renderAsync(scene, camera); },
    info: { worldW, hMin, hMax, size },
  };

  renderer.setAnimationLoop(() => {
    controls.update();
    renderer.render(scene, camera);
  });

  addEventListener('resize', () => {
    camera.aspect = innerWidth / innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(innerWidth, innerHeight);
  });
}

main().catch((e) => {
  status.textContent = `失敗: ${e?.message ?? e}`;
  console.error(e);
});
