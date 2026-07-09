# 創世 生命シミュレーション裏設定 (SPEC-life-sim)

2026-07-09 確定。17エージェントWF(コード精読4→サブ設計7→統合→バランス敵対検証4→最終確定)で策定。
要望: 時間経過 / 寿命 / 伐採→運搬→建築 / 川ツール / 魚湧き→誘引 / 獲物持ち帰り /
食料・家・男女比・集落規模に応じた出産 / 子の成長 / 放置繁栄と食料難の自律移住。

検証4レンズが暴いた5大破綻(①採集係数*0.5で乾燥地即絶滅 ②surplusY発火不能で分村ゼロ
③母系家系ヒートデス ④GPU/CPU水サンプリング非対称 ⑤NaN文鎮化)を修正済みの最終値。

---

## 0. 最重要決着: 個体 vs 村抽象

**「全員が実在する離散個体・体(3Dモデル)だけが希少なレンタル資源」**

- `LifeAgent` を村ごと `agents: LifeAgent[]`(AoS)で保持。1体≈48B・MAX_AGENTS=1024≈50KB。
  フィールド: `id`(単調連番=同一性) / `age`(gameYear, float) / `sex`(0=F,1=M) /
  `span`(誕生時確定寿命 45+rngSim*25) / `preg`(受胎CD残) / `line`(母系家系ID) / `body`(pool添字, -1=非表示)。
- **cap(PC44/スマホ18・N3後64/28)は描画専用の体プール上限のみ**。want枠は優先度
  (ジョブ実行者 > 子供最大40% > 大人slot昇順)で配る。子の見た目 = 借りた体の `scale=0.4+0.6*min(age,15)/15`。
- 「背景人口」という別の数値は存在しない。背景 = `body==-1` の実在個体。経済/人口動態は体の有無に一切依存しない。
- **cap非依存の構造保証(2段)**: (a) 一方向依存 = 台帳はpoolを読まない・poolが台帳を読むだけ。
  (b) rngSim(生命/経済/生態/移住)と rngView(表示のみ)の分離 =
  `reseed(s): rngSim=mulberry32(s), rngView=mulberry32(s^0x9e3779b9)`。
  → maxPeople=18と44で同seed→全人口動態statsがbit一致(検証⑧)。

## 1. 時間・ゲームクロック

- 時間単位 = **gameYear 一本**(別単位全廃)。**YEAR_SEC=20実秒/gameYear**(確定・単定数で変更可)。
  寿命≈19分・成人まで5分・4→40人繁栄=10-17分・1時間放置≈3世代。
- `village.update(dt)` 冒頭で低頻度tick: `simAcc+=dt; while(simAcc>=SIM_TICK){simAcc-=SIM_TICK; simTick(dtY)}`。
  SIM_TICK=0.5実秒(coarse readbackと同cadence)・dtY=0.025年/tick。
- clock導出 = `ticksDone*dtY + simAcc/YEAR_SEC`(単純加算しない=可変dtで乖離しない)。
- rAFも__geo.tickも同一update経路。geo.ts側にクロックを置かない。タブ非表示中は停止(オフライン進行なし・確定)。
- simTick固定順: 1.生態regrow 2.villages配列順 3.bands 4.表示照合(rngViewのみ)。

## 2. 加齢・寿命・死

- ADULT_AGE=15 / ELDER_AGE=50 / FERTILE_F=15-40。自然死 = age≥span(45-70・平均57.5)。
- **軽hazard付き(確定・e0≈38-40)**: 誕生日(floor(age)跨ぎ)に `rngSim()<hazardLUT[年齢]` で運死。
  hazardLUT = q0:0.12 / q1-4:0.03 / q5-14:0.008 / q15+:0.006(flat・整数年LUT・Math.pow禁止)。
  乳児死は数字のみ(演出なし)。
- 死: `agents.splice`(順序保存=決定論) → demoバケット-- → 体持ちなら死亡フェード2秒(rng不使用)→release。
- demoキャッシュ(kids/adultsF/adultsM/elders)はイベント時のみO(1)増減。

## 3. 食料経済+労働配分

- 単位 = 1fu(大人1人1gameYear分)。labor = adultsF+adultsM+0.5*elders(子は消費のみ)。
- **適応配分**: builders=min(2,floor(labor*0.35*foodGate)) → fishers(fishPot>0.2, 上限8) →
  hunters(huntPot>0.2, 上限6) → gatherers=残り全部(⚠*0.5禁止=即絶滅バグの元)。
  foodGate=clamp((foodStock/upkeep-0.1)/0.3,0,1)(飢餓時builders=0で全員食料へ)。
- 獲得: intake=takeFish+takeAnimal+GATHER_BASE*effW*s(*dtY)。GATHER_BASE=1.6(損益分岐s=0.625)。
  effW=min(Wg,12)+0.25*max(0,Wg-12)(収穫逓減=分村圧)。
- 消費: upkeep=kids*0.5+adults*1.0+elders*0.7。storageCap=hutsBuilt*4.0。
- ヒステリシス: nutrition<1→hungerY+=dtY*3*(1-nutrition), else -1/年。
  **surplusY=生産定義**: intake≥1.1*upkeep*dtY で+dtY, else 0(在庫定義は数学的に発火不能=禁止)。
- 飢餓死: deathAcc+=pop*0.4*(1-nutrition)*dtY・高齢順・**繁殖適齢F(15-40)は最後まで除外**(回復コア温存)。
- 持ち帰り演出(魚/肉を村へ運ぶ)は純演出・経済値非依存(=cap非依存)。

## 4. 出産式(中核)

```
B = fertBase * eligibleF * E_food * E_house * E_pair * E_size * E_div * dtY
```
- fertBase=0.50。eligibleF=|{F, 15≤age≤40, preg==0}|。
- E_food = clamp((foodStock/upkeep + 0.5*max(0,intakeRate/upkeep-1) - 0.1)/0.3, 0, 1)(在庫+黒字フロー項)。
- E_house = clamp((hutsBuilt*6 - pop)/6, 0, 1) → **均衡pop=6*huts=家が成長の律速(階段状成長)**。
- E_pair = 2*min(M,F)/max(1,M+F)(単一性別→0=真の不妊)。
- E_size = clamp(adults/4, 0, 1)。
- E_div = clamp(0.5+0.166*(lines-1), 0.5, 1.0)。lines=生存distinct家系(世代スタンプ配列でO(pop))。
- **家系再生**: 子lineは母継承だが6%で新line。+fission/誘引/respawnが新line鋳造(ヒートデス解毒)。
- 出産: birthAcc+=B; while(≥1){出産}。母preg=1.75年。sex=50:50だが村内|M-F|>3なら70%少数派バイアス。

## 5. 建築 = 伐採→運搬→建築

- BuildSite{k(黄金角スパイラル添字), logs:0-8, workY, stage:0-2}・村同時1件。
- LOGS_PER_HUT=8 / LOGS_PER_TREE=2 / 伐採+運搬実効=24logs/worker年 / BUILD=0.5worker年。
  STAGE閾値 = logs 0/4/8 & workY 0/0.25/0.5(木待ちの建ちかけが見える)。工期典型≈8-17実秒。
- TREE_REGROW=4本/年・TREES_CAP=24*vegScore(vegScore=flat*(1-en)*(0.3+0.7*min(1,wet*2))=川を掘ると森が濃い)。
- 着工条件: site==null & hutsBuilt<12 & pop>hutsBuilt*6-2 & trees≥4。
- **新村の家credit**: seed/分村/移住着地とも hutsBuilt=clamp(ceil(pop/6)+1, 2, 12)
  (入植4人→2軒=初子1.4年で離陸 / 移住45人→9軒=着地餓死防止)。
- 木材保存則: chopped == logsAtForest + Σsite.logs + hutsBuilt*8 + 返却。

## 6. 神の川ツール

- `Spring{id,u,v,Q}` = geo.ts層所有の唯一の新永続状態。MAX_SPRINGS=PC8/touch4(FIFO+HUD枯渇予告)。
  SPRING_Q=12m³/s・SPRING_RAD=max(2cell, worldW/256)。
- 操作: なぞり=起点Spring+軌跡carve(simBrush dig*0.35)/タップ=泉のみ/泉近傍タップ=削除。
- 水輸送は既存Mei浅水シム(境界排水=有界)に全委譲。CPU点源(sourceOn)を多点化・GPUは新kSourceカーネル。
- 🔴 **coarse water/wet = 8×8 maxプーリングを両経路で統一**(CPUバイリニア廃止・heightは中心点維持)。
  細川(<94m幅)がGPU/CPUで定性的に食い違う罠を封じる。flowUV(coarse ch3=|flux|)追加。
- FISH_WATER_MIN=0.15m / FLOW_MIN=0.2m/s。suitabilityはwaterScore飽和で川→適地化が村側改修ゼロで成立。

## 7. 生態 = 魚+陸獣+誘引

- 新規 `src/ecology.ts`・密度フィールド駆動(個体AIなし)。ECO_N=32(1024セル)。
- 動態: K<1e-3→N*=max(0,1-decay*dtY)(**線形・exp禁止**), else N+=(r*N*(1-N/K)+seedRate*K)*dtY。
  魚: r=1.1/年・seed=0.6K/年・decay=10/年・K=24*smoothstep(0.10,0.60,水深[0.05m量子化])。
  陸獣: r=0.3/年・K=16*flat*green*(1-min(1,水深/0.5))。拡散D=1.0/0.6/年。
  capacityRefresh=セル毎3×3=9点max(細川ブラインド回避)。
- harvest(Holling-II): Cmax=6/3・Nhalf=6/4・maxTake=0.4/年。
  **NaN guard**: tot>1e-6?按分:0.5 / vg.food=max(0,isFinite(x)?x:prev)。
  乱獲→数秒で枯渇→休漁で数十〜百秒回復(非対称)。
- **誘引**: resourceScore=clamp01((fishNear+0.6*landNear)/40)、
  誘引因子=clamp(0.8+0.4*rs, 0.8, **1.2**)(≤1.0だと魚目当て移住が式上死ぬ)。suitability_eff=suitability*因子。
- 描画追加 = fish 1 + 獣統合 1 + 木材の山 1 = **draw call +3のみ**(泉マーカーはfires相乗り)。
  位相ゆらぎは頂点シェーダ・instanceMatrix書込はeco-tick時のみ。

## 8. 移住・分村・気候・リスポーン

- **破綻→移住**: hungerY≥1.5 & migCD≤0 → 村→Band退行(agents/food携行)→適地スキャン。
  移住スコア=suitability*clamp(0.8+0.4*foodPot,0.8,1.2)*congestion(1/(1+0.5*近傍村))*jitter(0.95+0.1*hash)。
  共有セル収穫はpro-rata按分。再定住はuv>0.08離れた地・migCD=3年。
- **繁栄→分村**: pop≥45 & surplusY≥2 → 30%を新Band分離(新line)。単村上限≈72=小集落クラスタ路線(確定)。
- **気候振動(確定ON)**: intake*=1+0.30*sineLUT[256](clock/12+φ_region)。周期12gameYear・村セルhashで地域位相。
  凶作十年→飢餓移住 / 豊作十年→分村が放置でも自律発生。
- **リスポーン(確定ON)**: 総pop<8で最適水辺に4人[F,M,F,M]band(4新line)を約45実秒毎spawn(空世界非吸収)。
- MAX_AGENTS=1024到達: 出生を静かに棄却+HUD「環境収容力到達」(throwしない)。VILLAGE_MIN_POP=2で消滅。

## 9. 決定論スコープ+モバイル予算(全体規律)

- bit一致 = 同一コードパス(CPUヘッドレス固定dt)限定。GPU実機は許容帯(t=600sでpop±15%・fish±20%)。
- **simTick/ecoTick内 Math.pow/exp/sin/log/tan 禁止**(hazard/climate=LUT・decay=線形)。水深0.05m量子化。
- 性能: ≈40-60K float演算/秒(支配項=capacityRefresh+拡散)。毎frameは描画補間のみ(個体走査ゼロ)。
- rngSim/rngView分離が cap非依存の技術核。

## 確定した体感分岐(全て推奨案で確定・単定数で変更可)

| 分岐 | 確定 | 代替 |
|---|---|---|
| YEAR_SEC | **20実秒**(寿命19分) | 30=じっくり / 12=せっかち |
| 死モデル | **軽hazard e0≈40**(乳児死=数字のみ) | hazard無し e0≈57.5+fertBase0.30 |
| リスポーン | **ON**(45秒毎4人・全滅凍結回避) | OFF=神が手動再seed |
| 気候振動 | **ON**(±30%・12年周期・地域位相) | OFF=飽和後は静止画 |
| 集落規模 | **小集落クラスタ**(fission45・上限72) | 大都市(実装増) |
| 幼児(age<3) | **非表示**(3-15歳はscaleで成長可視) | 母の腰に子玉+1mesh |
| オフライン進行 | **なし**(見ている間だけ動く) | visibilitychange加算(決定論破壊) |

## 実装ロードマップ(N1-N7・旧M2-M6を再編)

| # | 内容 | 工数 | 主検証 |
|---|---|---|---|
| N1 | クロック+個体台帳+加齢/寿命/死(出産は簡易版) | 2-3 | 人口保存・cap非依存bit一致・寿命分布e0∈[35,45] |
| N2 | 食料経済+完全出産式(環境ゲート) | 3 | 質量保存・家頭打ちpop=6N±2・単一性別不妊・成長30-50gy |
| N3 | 部位別インスタンス化+性差+子の見た目(旧M2) | 2-3 | callsDelta≦20・sexRatio・子scale単調 |
| N4 | 伐採→運搬→建築パイプライン | 2-3 | 木材保存則・工期8-17s・森枯渇→建築凍結 |
| N5 | 神の川ツール+水センサ整合(8×8max統一) | 2 | 注入恒等式・水収支<0.1%・川→適地化 |
| N6 | 生態(魚+獣)+誘引+持ち帰り | 3 | 生態決定論・乱獲非対称回復・draw≤+3 |
| N7 | 移住/分村/気候/リスポーン(創発完成) | 2-3 | 破綻→移住・繁栄→分村・放置100年で凍結しない |

検証スイート30項目(不変量6+決定論6+均衡エッジ15+性能3)は本文各節参照。
全て `__geo.seedRng(n)→__geo.tick(sec,steps)` の固定dtヘッドレス経路。CPUが正・GPUは許容帯。
