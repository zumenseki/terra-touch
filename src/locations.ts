// terra-touch 名所プリセット (座標検証ワークフロー確定・2026-07-07)。
// lat/lon=中心、extentKm=見どころを収める一辺の目安、exag=垂直誇張。
// zoom は extentKm から自動計算 (小さい地形ほど高ズーム=細かいセル=川が解像する)。

export interface LocationDef {
  key: string;
  name_ja: string;
  lat: number;
  lon: number;
  extentKm: number;
  exag: number;
  hasWater?: boolean;
}

const EARTH_C = 40075016.686;

/** extentKm と緯度から Web Mercator の適正ズームを求める (GRID タイルで枠に収める) */
export function computeZoom(lat: number, extentKm: number, grid: number): number {
  const target = extentKm * 1000 * 1.5;
  const z = Math.round(Math.log2((EARTH_C * grid * Math.cos((lat * Math.PI) / 180)) / target));
  return Math.max(10, Math.min(15, z));
}

export const LOCATIONS: LocationDef[] = [
  { key: 'fuji', name_ja: '富士山', lat: 35.3608, lon: 138.7275, extentKm: 24, exag: 1.1 },
  // 峡谷
  { key: 'grand-canyon', name_ja: 'グランドキャニオン', lat: 36.09, lon: -112.115, extentKm: 40, exag: 1.3, hasWater: true },
  { key: 'yarlung-tsangpo', name_ja: 'ヤルンツァンポ大峡谷', lat: 29.72, lon: 95.05, extentKm: 50, exag: 1.0, hasWater: true },
  { key: 'verdon-gorge', name_ja: 'ヴェルドン峡谷', lat: 43.745, lon: 6.34, extentKm: 20, exag: 1.2, hasWater: true },
  // 一枚岩・奇岩・尖塔
  { key: 'uluru', name_ja: 'ウルル(エアーズロック)', lat: -25.3444, lon: 131.0354, extentKm: 10, exag: 1.4 },
  { key: 'half-dome', name_ja: 'ハーフドーム(ヨセミテ)', lat: 37.746, lon: -119.5329, extentKm: 11, exag: 1.05, hasWater: true },
  { key: 'devils-tower', name_ja: 'デビルスタワー', lat: 44.5903, lon: -104.7152, extentKm: 8, exag: 1.3 },
  { key: 'zhangjiajie', name_ja: '張家界(武陵源)', lat: 29.348, lon: 110.435, extentKm: 12, exag: 1.1 },
  { key: 'mount-roraima', name_ja: 'ロライマ山', lat: 5.18, lon: -60.7581, extentKm: 18, exag: 1.2, hasWater: true },
  // 山・火山
  { key: 'matterhorn', name_ja: 'マッターホルン', lat: 45.97639, lon: 7.65861, extentKm: 14, exag: 1.1 },
  { key: 'everest', name_ja: 'エベレスト', lat: 27.9881, lon: 86.925, extentKm: 30, exag: 1.05, hasWater: true },
  { key: 'denali', name_ja: 'デナリ', lat: 63.0694, lon: -151.0061, extentKm: 40, exag: 1.1, hasWater: true },
  { key: 'fitz-roy', name_ja: 'フィッツロイ', lat: -49.271278, lon: -73.043222, extentKm: 18, exag: 1.1, hasWater: true },
  { key: 'kilimanjaro', name_ja: 'キリマンジャロ', lat: -3.0759, lon: 37.3533, extentKm: 45, exag: 1.0 },
  // 水が主役
  { key: 'milford-sound', name_ja: 'ミルフォード・サウンド', lat: -44.645, lon: 167.885, extentKm: 13, exag: 1.0, hasWater: true },
  { key: 'iguazu-falls', name_ja: 'イグアスの滝', lat: -25.69583, lon: -54.43611, extentKm: 4, exag: 1.5, hasWater: true },
  { key: 'victoria-falls', name_ja: 'ビクトリアの滝', lat: -17.9248, lon: 25.8567, extentKm: 5, exag: 1.5, hasWater: true },
  // 砂漠・特殊地形
  { key: 'salar-de-uyuni', name_ja: 'ウユニ塩湖', lat: -20.242, lon: -67.627, extentKm: 30, exag: 1.6, hasWater: true },
  { key: 'sossusvlei', name_ja: 'ソススフレイ砂丘', lat: -24.7565, lon: 15.2915, extentKm: 7, exag: 1.25 },
  { key: 'grand-prismatic', name_ja: 'グランド・プリズマティック', lat: 44.5252, lon: -110.8383, extentKm: 2, exag: 1.6, hasWater: true },
];

export function pickLocation(key?: string | null): LocationDef {
  if (key) {
    const f = LOCATIONS.find((l) => l.key === key);
    if (f) return f;
  }
  return LOCATIONS[0];
}
