// ブラシの離散正規化 (DESIGN-m1m2.md「ブラシ」)
// GPU 側と同一の式でテクセル格子上の Σw, Σg を事前計算し、
// depositScale = Σw/Σg を uniform で渡す → 掘削体積と堆積体積が離散的に一致する。

export interface BrushNorm {
  /** 掘りプロファイル w の離散和 (テクセル単位) */
  sumW: number;
  /** 堆積プロファイル g の離散和 */
  sumG: number;
  /** g に掛けると Σ(掘り) = Σ(堆積) になる係数 */
  depositScale: number;
  /** 影響半径 (m) = 1.6R */
  outerRadius: number;
}

/** r<R: w(r) = (1-(r/R)^2)^2 */
export function digProfile(r: number, R: number): number {
  if (r >= R) return 0;
  const t = 1 - (r / R) * (r / R);
  return t * t;
}

/** R<r<1.6R: g(r) = sin^2(pi*(r-R)/(0.6R)) */
export function depositProfile(r: number, R: number): number {
  const outer = 1.6 * R;
  if (r <= R || r >= outer) return 0;
  const s = Math.sin((Math.PI * (r - R)) / (0.6 * R));
  return s * s;
}

/**
 * テクセル格子上の離散和。中心はテクセル中央に置いた近似
 * (サブテクセル位置による変動は <0.5% で ±5% 予算内)。
 */
export function computeBrushNorm(radiusMeters: number, cellMeters: number): BrushNorm {
  const R = radiusMeters;
  const outer = 1.6 * R;
  const extent = Math.ceil(outer / cellMeters) + 1;
  let sumW = 0;
  let sumG = 0;
  for (let dj = -extent; dj <= extent; dj++) {
    for (let di = -extent; di <= extent; di++) {
      const r = Math.hypot(di, dj) * cellMeters;
      sumW += digProfile(r, R);
      sumG += depositProfile(r, R);
    }
  }
  return {
    sumW,
    sumG,
    depositScale: sumG > 0 ? sumW / sumG : 0,
    outerRadius: outer,
  };
}
