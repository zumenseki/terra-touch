# SPEC — N8「生活と狩り」(実機FB5起点・親方承認 2026-07-17)

親方の実機FB(2026-07-17): ①足の関節が逆(→即修正済 f05f717) ②人は家に入って寝る/食事する
③動物も動く(→N6実装済・deploy済) ④動物の数が多すぎ(→即修正 f05f717) ⑤マンモス狩り
⑥戦闘で人も死亡 ⑦槍投げて戦う ⑧木を伐採(可視化) ⑨家が最初から建つのは変(→7eb16e4修正済)

## 親方承認の設計判断(2026-07-17)
- 戦闘範囲: 村同士の争いも含める(=N8.5 村レイドを追加)。
- 狩り死亡リスク: 高(期待1.5〜3人/成功狩り)。
- 実装順: 木→家生活→マンモス→狩り→村レイド。都度deployは親方確認。

## 進捗
- ✅ N8.1 木の実体化 = 実装+検証(commit予定)。treeMesh追従/伐採で木減再生で戻る(実測: 8→6→4本→再生8)/木材保存則/cap非依存28==64/NaN0。
- ⬜ N8.2 家生活 / ⬜ N8.3 マンモス / ⬜ N8.4 狩り / ⬜ N8.5 村レイド

## 0. 大原則(不変条件)
- SIM(人口/経済/死・決定論・cap非依存)と VIEW(見た目・rngView/hash・体レンタル)を分離。
- 決定論: SIM は rngSim のみ。VIEW は rngView/ハッシュ。Math.random/Date.now 禁止。
- cap非依存(28==64): SIM状態は台帳(LifeAgent)と村集約のみ。マンモス数・狩り人数・木本数は台帳整数かマップ定数から導く。
- 保存則: 食料会計(除去==供給)・木材保存則(chopped==siteLogs+logsUsed)。
- 検証: _mkSim(28)==_mkSim(64) bit一致 / 決定論 / NaN無 / tsc0 / スクショ。

### 接続点
- simTick順: eco.tick→村{ageAndDie→demo→economy→buildStep→demo→birthsIn→demo→age++}→band→消滅村除去→migrateAndFission→respawnIfEmpty→reconcileBodies→eco.retarget。DT_Y=0.025。
- clock()=gameYear。昼夜は無い(YEAR_SEC=20実秒/年)→②にVIEW昼夜クロック新設。
- 体: reconcileBodies/assignBody/releaseBody/killBody(死フェード2秒・rng不使用)。
- 木: buildStep, vg.trees(数値), LOGS_BY_STAGE[0,4,8]。生態: land密度(ECO_N=32)。
- 家メッシュ流儀: paint(geo,hex)+MeshStandardNodeMaterial(vertexColors)+mergeGeometries、renderStaticでダーティフラグ配置。

## N8.1 木の実体化(⑧・VIEW最小) ✅実装済
村ごと木InstancedMesh(MAX_TREES=800)・本数=min(TREE_SHOW_CAP=30,round(vg.trees))・配置=決定論ハッシュ(村u,v+k・家の外側リングU*3〜7)。treeSigダーティフラグ(round(trees)変化で再構築)。伐採で末尾から消え再生で戻る。幹(円柱茶)+葉(円錐緑)merge・hutMat流用(draw+1)。SIM不変。

## N8.2 昼夜クロック+家生活(②・VIEW専用)
dayPhase(0..1・DAY_SEC≈8実秒・SIM独立)。空/光を変える。夜=最寄りの家へ→scale0、満杯は焚火周りで就寝。食事(≈0.5短窓)=集まりpause。昼=従来。rngView/ハッシュのみ(rngSim不消費)。数人夜警。検証: 体が家へ→朝再出現/SIM stats昼夜不変/cap非依存/NaN無。

## N8.3 マンモス個体(⑤土台・SIM個体レイヤー)
mammoths[]{id,u,v,hp,state,tu,tv,target}。個数=マップ定数(3〜5)。配置=地形ハッシュ(rngSim)。徘徊+respawn。専用ジオメトリ(胴+足4+鼻+牙)(draw+1)。体レンタル不使用。検証: 決定論/cap非依存28==64/respawn個体数保存/NaN無/draw+1。

## N8.4 狩り・槍・戦闘死(⑤⑥⑦・SIM解決+VIEW槍)
村近く(半径R)にマンモス&狩人(適齢男性)→hunt解決(rngSim): hunters比例でhp削る/反撃で狩人死亡(killBody)/撃破→大量食料windfall(食料会計の供給)。死亡=期待1.5〜3人/成功(親方=高)。totalDeaths計上。VIEW=体がマンモスへ槍(細円柱)放物線(draw+1)。cap非依存: hunters=台帳整数。

## N8.5 村レイド(⑥拡張・承認で追加)
飢餓村が近隣村を襲う。両村戦力(適齢成人数)で勝敗・双方死者・勝者が食料奪取(rngSim)。VIEW=槍投げ流用。戦力=台帳整数。検証: 決定論/cap非依存/死者・食料移動の保存/NaN無。

## 6. 非目標(N8外)
マンモス以外の大型獣 / 個体詳細AI / 食料以外のマンモス副産物(毛皮等)。

## 7. 検証テンプレ
_mkSim(28)と(64)同seed同地点で stats/wood/demo/mammoth/hunt bit一致 / 決定論 / NaN無 / 保存則 / draw実測 / tsc0 / スクショ。
