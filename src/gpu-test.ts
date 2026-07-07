// GpuSim 単体テスト: 合成地形(片下がり+谷)に雨→水が低所へ流れるか + NaN無 + 保存を検証。
import * as THREE from 'three/webgpu';
import { GpuSim } from './sim-gpu';

const out = document.getElementById('out')!;
const log = (s: string) => { out.textContent += '\n' + s; };

async function main() {
  const renderer = new THREE.WebGPURenderer({ antialias: false });
  renderer.setSize(4, 4);
  document.body.appendChild(renderer.domElement);
  await renderer.init();
  log('backend: ' + ((renderer.backend as { isWebGPUBackend?: boolean }).isWebGPUBackend ? 'WebGPU' : 'WebGL2'));

  const N = 256;
  const world = 2000; // m
  // 合成地形: x方向に片下がり + 中央に谷
  const bed = new Float32Array(N * N);
  for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
    const xf = i / (N - 1), zf = j / (N - 1);
    const tilt = 200 * (1 - xf);           // 左が高い
    const valley = -60 * Math.exp(-((zf - 0.5) ** 2) / (2 * 0.05 ** 2)); // 中央横谷
    bed[j * N + i] = 50 + tilt + valley;
  }
  const sim = new GpuSim(renderer, { n: N, world, bedrock: bed });
  (window as unknown as { __gpu: unknown }).__gpu = { sim, renderer };

  sim.step(1, false); // ウォームアップ(GPUバッファ確保)
  const t0 = await sim.totals();
  log(`初期 solid=${Math.round(t0.solid)} water=${Math.round(t0.water)} nan=${t0.nan}`);

  // 雨を降らせる
  sim.rainRate = 0.5;
  for (let f = 0; f < 200; f++) sim.step(1, false); // slumpなしで水のみ
  const t1 = await sim.totals();
  log(`雨200step後 solid=${Math.round(t1.solid)} water=${Math.round(t1.water)} nan=${t1.nan}`);

  // 低所(右 xf=1付近)と高所(左 xf=0付近)の水深を比較
  const buf = new Float32Array(await renderer.getArrayBufferAsync(sim.state.value as never));
  const depthAt = (i: number, j: number) => buf[(j * N + i) * 4 + 2];
  let loW = 0, hiW = 0, loN = 0, hiN = 0, maxD = 0;
  for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
    const d = depthAt(i, j); if (d > maxD) maxD = d;
    if (i > N * 0.8) { loW += d; loN++; } else if (i < N * 0.2) { hiW += d; hiN++; }
  }
  log(`低所(右)平均水深=${(loW / loN).toFixed(3)}m 高所(左)平均=${(hiW / hiN).toFixed(3)}m maxDepth=${maxD.toFixed(2)}m`);
  log(`判定: 低所>高所 なら水は下流へ流れてる → ${(loW / loN) > (hiW / hiN) ? 'PASS ✅' : 'FAIL ❌'}`);
  log(`NaN無 → ${t1.nan ? 'FAIL ❌' : 'PASS ✅'}`);

  // ブラシ保存テスト
  const before = (await sim.totals()).solid;
  for (let f = 0; f < 30; f++) sim.brush(0.5, 0.5, 1 / 60, 'dig');
  const after = (await sim.totals()).solid;
  log(`ブラシ dig30回: solid ${Math.round(before)}→${Math.round(after)} drift=${(((after - before) / before) * 100).toFixed(3)}%`);
}
main().catch((e) => { out.textContent += '\n❌ ' + (e?.stack || e); console.error(e); });
