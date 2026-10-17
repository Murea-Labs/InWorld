import { t } from './i18n'
/**
 * Visual style presets (style bible).
 * 
 * Each one must be **a full English paragraph usable directly as a prompt**: it constrains
 * every character image, place image and video of the world as real conditioning.
 * 
 * Each covers only art style, color, materials, lighting and lens feel. No specific people,
 * plot or places: those belong to each beat and would pollute the whole world here.
 */
export interface StylePreset {
  id: string
  label: string
  text: string
}

export const STYLE_PRESETS: StylePreset[] = [
  {
    id: 'photoreal',
    label: t('电影写实'),
    text:
      'cinematic live action, photorealistic, natural daylight with soft falloff, ' +
      'shallow depth of field, subtle film grain, muted neutral color grade, ' +
      'handheld camera with gentle drift, realistic skin and fabric texture',
  },
  {
    id: 'anime',
    label: t('日系动画'),
    text:
      '2D Japanese animation, clean confident line art, cel shading with two-tone shadows, ' +
      'saturated but harmonious palette, painted background art with visible brushwork, ' +
      'warm rim light, gentle bloom on highlights, steady framing',
  },
  {
    id: 'watercolor',
    label: t('童话水彩'),
    text:
      'storybook watercolor illustration, soft wet-on-wet washes, visible paper grain, ' +
      'pastel palette with warm cream base, loose pencil under-drawing left visible, ' +
      'diffuse ambient light, no harsh shadows, gentle static camera',
  },
  {
    id: 'inkwash',
    label: t('东方水墨'),
    text:
      'Chinese ink-wash painting, monochrome black ink on rice paper with a single ' +
      'muted accent tone, generous negative space, soft bleeding edges, dry-brush texture, ' +
      'flat overcast light, slow contemplative camera',
  },
  {
    id: 'oil',
    label: t('史诗奇幻油画'),
    text:
      'epic fantasy oil painting, thick impasto brushwork, deep chiaroscuro, ' +
      'warm amber and deep umber palette with cool shadow accents, golden rim light, ' +
      'varnished canvas texture, slow majestic camera moves',
  },
  {
    id: 'neon',
    label: t('赛博霓虹'),
    text:
      'cyberpunk neon cinematography, high contrast, wet reflective surfaces, ' +
      'teal and magenta light sources cutting through haze, volumetric fog, ' +
      'anamorphic lens flares, heavy bokeh, slight chromatic aberration',
  },
  {
    id: 'steampunk',
    label: t('复古蒸汽朋克'),
    text:
      'retro steampunk, brass and oiled leather materials, riveted iron, ' +
      'sepia and verdigris palette, warm gaslight and ember glow, drifting steam and dust motes, ' +
      'aged photographic vignette, slightly desaturated highlights',
  },
  {
    id: 'gothic',
    label: t('暗黑哥特'),
    text:
      'dark gothic, low-key lighting with deep crushed blacks, cold desaturated palette ' +
      'with a single blood-red accent, damp stone and tarnished silver texture, ' +
      'hard directional key light, heavy shadow, slow oppressive camera',
  },
  {
    id: 'comic',
    label: t('手绘漫画'),
    text:
      'hand-drawn comic art, bold inked outlines of varying weight, flat color fills ' +
      'with halftone shading, limited high-contrast palette, visible paper texture, ' +
      'graphic simplified lighting, punchy static framing',
  },
]

/**
 * The menu sent to the backend to pick from. Include the start of the English text; with
 * only the Chinese label the model could pair "ink wash" with cyberpunk.
 */
export const styleMenu = () =>
  STYLE_PRESETS.map((p) => ({ id: p.id, label: `${p.label} — ${p.text.slice(0, 64)}` }))

/** The backend returns only the id; the text comes from this curated list so there is one copy. */
export const styleById = (id: string) =>
  STYLE_PRESETS.find((p) => p.id === id)?.text ?? ''
