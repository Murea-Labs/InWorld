import * as api from './api'
import { styleById } from './presets'
import type { RandomWorld } from './types'
import { t } from './i18n'

/** In-page retries resume the same world at the first unfinished asset. */
export interface WorldStart {
  idea: string
  owner: string
  concept: RandomWorld
  videoModel: string
  worldId?: string
  completed: number
}

export async function prepareWorld(start: WorldStart, progress: (name: string, i: number, n: number) => void) {
  const w = start.concept
  const assets = w.assets ?? []
  if (start.idea && (!assets.length || assets.filter(a => a.kind === 'player').length !== 1
    || assets.filter(a => a.kind === 'place').length !== 1)) {
    throw new Error(t('开场资产不完整，请重新描述世界'))
  }
  if (!start.worldId) {
    const created = await api.createWorld(w.title, w.setting, w.visual_style || styleById(w.style_id),
      start.videoModel, w.mainline)
    start.worldId = created.id
  }
  for (let i = start.completed; i < assets.length; i++) {
    const asset = assets[i]
    progress(asset.name, i + 1, assets.length)
    const form = new FormData()
    form.append('name', asset.name)
    form.append('image_prompt', asset.prompt)
    if (asset.kind === 'place') {
      form.append('make_current', 'true')
      await api.addPlace(start.worldId, form)
    } else {
      form.append('control', asset.kind === 'player' ? 'user' : 'ai')
      form.append('kind', asset.kind === 'object' ? '物件' : asset.kind === 'creature' ? '生物' : '人物')
      await api.addSubject(start.worldId, form)
    }
    start.completed = i + 1
  }
  return start.worldId
}
