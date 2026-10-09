export type SceneGlyphKind =
  | 'vehicle'
  | 'dining'
  | 'shop'
  | 'lift'
  | 'office'
  | 'home'
  | 'outdoor'
  | 'building'

const GLYPHS: Record<SceneGlyphKind, string[]> = {
  vehicle: [
    'M5 16h14l-1.5-6h-11L5 16Zm2-6 1.5-4h7L17 10M7 16v3M17 16v3M8 13h.01M16 13h.01',
  ],
  dining: [
    'M7 3v7M4.5 3v5.5A2.5 2.5 0 0 0 7 11v10M9.5 3v5.5A2.5 2.5 0 0 1 7 11M16 3v18M16 3c3 2 4 5 4 8h-4',
  ],
  shop: [
    'M5 8h14l-1 13H6L5 8ZM9 8V6a3 3 0 0 1 6 0v2',
  ],
  lift: [
    'M5 3h14v18H5V3ZM9 8l3-3 3 3M12 5v6M15 16l-3 3-3-3M12 13v6',
  ],
  office: [
    'M4 4h16v11H4V4ZM8 20h8M12 15v5M8 11l3-3 2 2 3-4',
  ],
  home: [
    'M3 11 12 3l9 8M5 9.5V21h14V9.5M9 21v-6h6v6',
  ],
  outdoor: [
    'M12 3 7 10h3l-4 6h5v5h2v-5h5l-4-6h3L12 3Z',
  ],
  building: [
    'M5 21V5l7-3 7 3v16M3 21h18M9 7h1M14 7h1M9 11h1M14 11h1M9 15h1M14 15h1M10 21v-3h4v3',
  ],
}

/** The place name is the only stable semantic signal; match the more specific categories first. */
export function sceneGlyphFor(name: string): { kind: SceneGlyphKind; paths: string[] } {
  let kind: SceneGlyphKind = 'building'
  if (/车|驾驶|座舱|车站|机场|car|train|bus|cockpit|station|airport|ship|boat/i.test(name)) kind = 'vehicle'
  else if (/电梯|楼梯|扶梯|电梯厅|elevator|lift|stair|escalator/i.test(name)) kind = 'lift'
  else if (/餐厅|餐馆|饭店|饭馆|食堂|咖啡|包间|后厨|酒吧|茶馆|拉面|饮品|厨房|restaurant|cafe|café|diner|kitchen|bar|tavern|inn|pub|tea/i.test(name)) kind = 'dining'
  else if (/超市|商店|商场|市场|便利店|卖场|shop|store|market|mall/i.test(name)) kind = 'shop'
  else if (/会议|办公室|公司|工作室|工位|写字楼|office|meeting|studio|workshop/i.test(name)) kind = 'office'
  else if (/家|卧室|客厅|公寓|宿舍|住宅|别墅|home|house|bedroom|living room|apartment|dorm|villa/i.test(name)) kind = 'home'
  else if (/街|路|广场|公园|花园|楼下|门外|天台|海边|山|street|road|square|park|garden|rooftop|beach|shore|mountain|forest|field/i.test(name)) kind = 'outdoor'
  return { kind, paths: GLYPHS[kind] }
}

export default function SceneGlyph({ name }: { name: string }) {
  const glyph = sceneGlyphFor(name)
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" data-scene-glyph={glyph.kind}>
      {glyph.paths.map((path) => <path d={path} key={path} />)}
    </svg>
  )
}
