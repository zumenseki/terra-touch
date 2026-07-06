// 2D value noise + fBm — 地形生成用 (決定的・依存ゼロ)

function hash2(ix: number, iy: number): number {
  let n = (ix * 374761393 + iy * 668265263) | 0;
  n = (((n ^ (n >>> 13)) | 0) * 1274126177) | 0;
  return ((n ^ (n >>> 16)) >>> 0) / 4294967296;
}

function valueNoise(x: number, y: number): number {
  const ix = Math.floor(x);
  const iy = Math.floor(y);
  const fx = x - ix;
  const fy = y - iy;
  const ux = fx * fx * fx * (fx * (fx * 6 - 15) + 10);
  const uy = fy * fy * fy * (fy * (fy * 6 - 15) + 10);
  const a = hash2(ix, iy);
  const b = hash2(ix + 1, iy);
  const c = hash2(ix, iy + 1);
  const d = hash2(ix + 1, iy + 1);
  return a + (b - a) * ux + (c - a) * uy + (a - b - c + d) * ux * uy;
}

// オクターブ毎に座標を回転して格子の軸方向アーティファクトを消す
const RC = Math.cos(0.6);
const RS = Math.sin(0.6);

/** 0..1 のフラクタルノイズ (なだらかな起伏) */
export function fbm(x: number, y: number, octaves = 6): number {
  let sum = 0;
  let amp = 0.5;
  let norm = 0;
  let px = x;
  let py = y;
  for (let i = 0; i < octaves; i++) {
    sum += amp * valueNoise(px + i * 17.13, py - i * 9.71);
    norm += amp;
    amp *= 0.5;
    const nx = (px * RC - py * RS) * 2;
    const ny = (px * RS + py * RC) * 2;
    px = nx;
    py = ny;
  }
  return sum / norm;
}

/** 0..1 のリッジノイズ (尖った尾根) */
export function ridged(x: number, y: number, octaves = 5): number {
  let sum = 0;
  let amp = 0.5;
  let norm = 0;
  let px = x + 31.7;
  let py = y + 7.3;
  for (let i = 0; i < octaves; i++) {
    const n = 1 - Math.abs(2 * valueNoise(px + i * 13.7, py + i * 5.1) - 1);
    sum += amp * n * n;
    norm += amp;
    amp *= 0.5;
    const nx = (px * RC - py * RS) * 2.1;
    const ny = (px * RS + py * RC) * 2.1;
    px = nx;
    py = ny;
  }
  return sum / norm;
}
