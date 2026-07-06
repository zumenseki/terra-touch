# M1/M2 設計 (2026-07-07)

SPEC.md の M1 (押す触感)・M2 (水) の実装設計。レビュー時はこの文書と実装の一致を照合する。

## 実装メモ (方針転換: GPU compute → CPU シム)

当初は TSL compute + storage texture で GPU 上に状態を持つ設計だったが、
installed three r185 の compute/storage API を実ソースで確定できなかった
(調査エージェントがクレジット切れで中断) ため、**確実性を優先して CPU シミュレーションに転換**。

- 状態は CPU の Float32Array (bedrock/soil/water/wet + 水フラックス4本)。
- 毎フレーム RGBA16F の DataTexture [r=bedrock, g=soil, b=water, a=wet] (メートル) に詰めて
  GPU 描画へ渡す ([sim.ts](../src/sim.ts) `sync()`)。地形(r,g)は変形時のみ再パック。
- 描画メッシュは 1024²、シム格子は **192²** (60fps 目標での実測最適点)。
  bilinear で滑らかに補間されるため幾何解像度低下は目立たない。
- 利点: 体積保存が CPU 総和で厳密検証可 / レイキャストが配列参照で即時正確 / API リスク 0。
- **M4 で Web Worker + 256² 化して 60fps 常時達成予定** (現状 ~40fps・SPEC 下限30 は充足)。

以下の状態表・アルゴリズムは CPU 実装でもそのまま成立している。

## 状態テクスチャ (すべて r32float storage・単位メートル・1024²)

| 名 | 内容 | 更新 |
|---|---|---|
| `bedrock` (B) | 岩盤高。初期地形。ブラシでのみ減る | brush |
| `soil` (S) | 砂・土層。初期 0。ブラシ堆積とスランプで動く | brush / slump |
| `soilFlux` | スランプ4方向流出量 (rgba32float) | slump pass1 |
| `water` (d) | 水深 | water pass2 |
| `waterFlux` | 水4方向流出量 (rgba32float・L,R,B,T) | water pass1 |

地表面高 H = B + S。水面高 W = H + d。
2層構造の理由: スランプを S だけに適用 → 初期の急峻な山が溶けない。
M4 の浸食 (岩盤→土砂化) への布石。

## ブラシ (体積保存・押すと縁が盛り上がる)

- 中心 c (ワールド xz)、半径 R (初期 15m)、掘り速度 A (m/s)、dt 毎に適用。
- 掘りプロファイル: r<R で w(r) = (1-(r/R)²)²。
- 堆積プロファイル: R<r<1.6R で g(r) = sin²(π(r-R)/(0.6R))。
- 正規化: JS 側でテクセル格子上の離散和 Σw, Σg を事前計算 (R 変更時のみ)、
  uniform `depositScale = Σw/Σg` を渡す → 掘った体積 = 盛った体積 (離散的に一致)。
- テクセル毎: dig = w·A·dt → S から先に取り、足りない分は B から。堆積は必ず S へ。
  クランプ: B≥0, S≥0。
- 端の体積漏れ防止: ブラシ中心を [1.6R, WORLD−1.6R] にクランプ。
- ブラシ位置は GPU レイキャスト (下記) の結果バッファを **GPU 上で直接** 読む
  (CPU 往復なし・遅延ゼロ)。

## GPU レイキャスト

- 1スレッド compute。カメラレイ (uniform origin/dir) を最大 600m、固定歩幅 0.75m で
  マーチ → H=B+S と比較、符号反転区間を 8 回二分探索で精緻化。
- 出力: storage buffer [x, y, z, hitFlag]。ブラシ pass と HUD リング表示 (非同期読戻し・
  1フレーム遅れ可) が共用。

## スランプ (安息角・S のみ移動)

パイプモデル型 2 パス (レース回避・完全保存):

1. **flux**: 各セル、4近傍 n について excess_n = max(0, (H_c − H_n) − maxDiff)。
   maxDiff = tan(34°)·cell。out_n = k·excess_n/4 (k=0.5)。
   総流出 Σout > S_c なら S_c/Σout 倍に縮小 (土が無い分は出せない)。
2. **apply**: S += Σ(隣の自分向き flux) − Σ(自分の flux)。

保存性: 出た分は必ずどこかに入る → Σ(B+S) はブラシ・スランプで不変 (機械精度)。

## 水 (浅水・パイプモデル / Mei–Decaudin–Hu)

1. **flux**: f_n ← max(0, f_n·damp + dt·g·(W_c − W_n)·cell)。damp=0.998, g=9.81。
   スケール K = min(1, d_c·cell² / (Σf·dt)) で負深度防止。
2. **depth**: d += dt·(Σ流入 − Σ流出)/cell²。
- 境界: 最外周は d=0 固定 (排水・マップ外へ消える)。流出量は総量ログで別掲。
- 湧き水: 谷上流の1点 (半径3m) に Q≈12 m³/s を注入。トグル可。蒸発なし。
- dt = 1/120s × 2 substep/frame。CFL 相当は damp とスケールで担保 (実測で調整)。

## パス順 (毎フレーム)

raycast → (押下中) brush → slump×1 → water flux → water depth (×2 substep) → render。

## 描画

- 地形: 頂点は texel 1:1 の `.load` (フィルタ不要)。フラグメント法線は 4近傍 load の
  手動バイリニア。色スプラットは M0 踏襲 + 濡れ (d>0 と湿り) で暗く。
- 水: 同トポロジ別メッシュ。y = H + d。d の手動バイリニアで
  opacity = smoothstep(0.02, 0.5, d)·0.92、深度で色 (浅=青緑透明・深=濃紺)。
  roughness 0.06。transparent・depthWrite false。泡・SSR は M3。

## 総量モニタ (受入条件1・drift 検知)

- 4×4 縮約ピラミッド compute (vec4: B,S,d 同時) 1024→1 → storage buffer →
  非同期読戻し (1s 毎)。
- HUD: 土 Σ(B+S) の初期比 drift% (|drift|>5% で赤)、水 Σd、fps、backend。
- 水は「湧き総量 − 端排水」があるため drift 判定は土のみ。水は注入停止時の安定性を
  デバッグ API で確認。

## デバッグ API (自動検証用・rAF 停止中も動く)

`window.__tt = { step(n), brushAt(wx, wz, seconds), totals(), heightAt(wx, wz), setSource(on), render() }`
- step: シムを n ステップ進め renderAsync 1 回 (hidden タブでも検証可能)。
- totals: 縮約を即時実行し {bedrock, soil, water} を返す (await)。
- heightAt: GPU レイキャストを真上からのレイで実行し H を返す (await)。

## 受入テスト対応

| 受入 | 検証手段 |
|---|---|
| 1. 体積保存±5% | totals() を brushAt 前後で比較 + HUD drift% |
| 2. せき止め→迂回 | brushAt で川を横断する土手を作り step(600) → 上流 heightAt(水面) 上昇・下流水深減を確認 |
| 4. 60fps/err0/NaN0 | fps HUD・console・totals() の NaN チェック |
