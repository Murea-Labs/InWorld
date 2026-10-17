import { t } from '../i18n'
import { useEffect, useRef, useState } from 'react';
import { mediaUrl } from '../api';
import { markFilePickerOpen } from '../filepicker';
import * as api from './api';
import type { AssetKind, Room } from './api';

function AssetCard({ kind, room, host, working, run }: {
  kind: AssetKind; room: Room; host: boolean; working: boolean;
  run: (work: () => Promise<Room>) => Promise<void>;
}) {
  const label = kind === 'scene' ? t('场景') : 'BOSS';
  const current = kind === 'scene'
    ? { name: room.scene_name, description: room.scene_description || '', ref: room.scene_ref,
        source: room.scene_asset_source }
    : room.monster;
  const recommendation = room.asset_recommendations?.[kind];
  const preview = room.asset_previews?.[kind];
  const initial = preview || (current?.ref ? current : recommendation);
  const [name, setName] = useState(initial?.name || '');
  const [description, setDescription] = useState(initial?.description || '');
  const [file, setFile] = useState<File | null>(null);
  const edited = useRef(false);
  useEffect(() => {
    if (!edited.current && initial) {
      setName(initial.name); setDescription(initial.description);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps -- follow the server value until the player edits
  }, [initial?.name, initial?.description]);
  const rendering = preview?.status === 'generating';
  const disabled = !host || working || room.status === 'generating';
  const tooMany = kind === 'monster' && room.members.length > 3;
  const hasPreview = preview?.status === 'ready';
  const extraction = room.asset_extraction;
  const extractionItem = extraction?.[kind];
  return <section className="coop-shared-asset" aria-label={t("{label}参考图设置", { label })}>
    <div className="coop-shared-asset-head">
      <h3>{t("{label}参考图", { label })}</h3><span className="coop-muted">{current?.ref ? t('已采用') : t('选填')}</span>
    </div>
    <p className="coop-muted">{kind === 'scene'
      ? t('建立故事发生的地点。场景图保持空景，人物和怪兽由各自参考图加入。')
      : t('为共同面对的生物固定外观。是否加入、如何互动，由你们决定。')}</p>
    {extraction?.status === 'generating' && extractionItem?.status === 'pending' &&
      <p role="status" className="coop-muted">{t("序章视频已完成，正在从关键帧提取{label}设定…", { label })}</p>}
    {current?.ref && <figure className="coop-asset-current">
      <img src={mediaUrl('refs', current.ref)} alt={t("已采用的{label}：{name}", { label, name: current.name })} />
      <figcaption>{current.name}{t(" · 已用于后续镜头")}{current.source === 'opening_video' ? t(' · 序章提取') : current.source === 'round_video' ? t(' · 视频自动提取') : ''}</figcaption>
    </figure>}
    {recommendation && <div className="coop-asset-suggestion">
      <strong>{t("背景推荐")} · {recommendation.name}</strong>
      <p>{recommendation.description}</p>
      {host && <button className="ghost sm" disabled={disabled} onClick={() => {
        edited.current = true; setName(recommendation.name); setDescription(recommendation.description);
      }}>{t("填入推荐{label}设定", { label })}</button>}
    </div>}
    {host && <>
      <fieldset disabled={disabled}>
        <label>{t("{label}名称", { label })}<input value={name} maxLength={60} onChange={e => {
          edited.current = true; setName(e.target.value);
        }} placeholder={kind === 'scene' ? t('例如：巨龙的地下宝库') : t('例如：守护宝库的岩甲巨龙')} /></label>
        <label>{t("{label}描述", { label })}<textarea value={description} rows={4} maxLength={500} onChange={e => {
          edited.current = true; setDescription(e.target.value);
        }} placeholder={t("可先等待背景推荐，也可以直接填写自己的设定")} /></label>
        <div className="coop-asset-buttons">
          <button className="primary" disabled={!room.shared_image_backend || rendering || tooMany || !name.trim() || !description.trim()}
            onClick={() => void run(() => api.generateAsset(room.code, kind, { name, description }))}>
            {rendering ? t("正在生成{label}…", { label }) : t("生成{label}参考图", { label })}
          </button>
          {current?.ref && <button className="ghost" onClick={() => void run(() => api.removeAsset(room.code, kind))}>
            {t("移除{label}参考图", { label })}
          </button>}
        </div>
        <details className="coop-asset-upload"><summary>{t("或上传自己的{label}参考图", { label })}</summary>
          <label className="coop-upload"><span>{t("选择{label}图片", { label })}</span><small>{file?.name || 'JPG / PNG / WebP'}</small>
            <input aria-label={t("上传{label}参考图", { label })} type="file" accept="image/jpeg,image/png,image/webp"
              onClick={markFilePickerOpen} onChange={e => setFile(e.target.files?.[0] || null)} />
          </label>
          <button disabled={!file || !name.trim() || !description.trim() || tooMany} onClick={() => {
            const body = new FormData(); body.set('name', name); body.set('description', description); body.set('file', file!);
            void run(() => api.uploadAsset(room.code, kind, body));
          }}>{t("上传并采用{label}图", { label })}</button>
        </details>
      </fieldset>
      {tooMany && <p className="coop-muted">{t("独立怪兽和场景最多可与三位玩家共同出镜；当前队伍已有四位玩家。")}</p>}
    </>}
    {rendering && <p role="status">{t("正在生成{label}预览，完成后可确认采用。", { label })}</p>}
    {preview?.status === 'failed' && <p role="alert" className="coop-asset-error">{preview.error}</p>}
    {extractionItem?.status === 'skipped' && <p className="coop-muted">{t("序章画面中未确认出现{label}，你仍可以手动设置。", { label })}</p>}
    {extractionItem?.status === 'superseded' && <p className="coop-muted">{t("你已修改{label}设定，序章提取结果已保留你的选择。", { label })}</p>}
    {extractionItem?.status === 'failed' && <p role="alert" className="coop-asset-error">{t("序章提取{label}失败，你仍可以手动设置。", { label })}</p>}
    {hasPreview && <figure className="coop-asset-preview">
      <img src={mediaUrl('refs', preview.ref)} alt={t("{label}生成预览：{name}", { label, name: preview.name })} />
      <figcaption><strong>{preview.name}{t(" · 待采用")}</strong><p>{preview.description}</p></figcaption>
      {host && <button className="primary" disabled={disabled || tooMany}
        onClick={() => void run(() => api.useAsset(room.code, kind, preview.request_id))}>{t("采用这张{label}图", { label })}</button>}
    </figure>}
  </section>;
}

export default function WorldAssets({ room, host, apply, kind }: {room: Room; host: boolean; apply: (room: Room) => void; kind: AssetKind}) {
  const [working, setWorking] = useState(false);
  const [recommending, setRecommending] = useState(false);
  const [error, setError] = useState('');
  const requested = useRef(false);
  const run = async (work: () => Promise<Room>) => {
    setWorking(true); setError('');
    try { apply(await work()); } catch (e) { setError((e as Error).message); }
    finally { setWorking(false); }
  };
  const recommend = async (refresh = false) => {
    setRecommending(true); setError('');
    try { apply(await api.recommendAssets(room.code, refresh)); }
    catch (e) { setError((e as Error).message); }
    finally { setRecommending(false); }
  };
  useEffect(() => {
    if (host && !room.asset_recommendations && !requested.current && room.status !== 'generating'
      && room.asset_extraction?.status !== 'generating' && room.asset_extraction?.status !== 'ready') {
      requested.current = true; void recommend();
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps -- recommend once per room and status, not on every room update
  }, [host, room.code, room.status]);
  return <div className="coop-world-assets">
    {kind === 'monster' && <p>{t("BOSS 是可选设定。你可以根据故事需要创建一只共享的敌对生物，也可以留空。")}</p>}
    {host && <button className="ghost" disabled={recommending || working || room.status === 'generating'}
      onClick={() => void recommend(true)}>{recommending ? t('正在根据故事背景推荐…') : t('根据背景重新推荐')}</button>}
    {recommending && <p role="status">{t("正在根据故事背景准备推荐设定…")}</p>}
    {error && <p className="coop-asset-error" role="alert">{error}</p>}
    {!host && <p className="coop-muted">{t("房主负责生成和采用共享参考图，所有队友都能查看。")}</p>}
    {host && !room.shared_image_backend && <p className="coop-muted">{t("尚未配置生图服务，可以先编辑设定并上传参考图。")}</p>}
    <div className="coop-shared-assets">
      <AssetCard kind={kind} room={room} host={host} working={working} run={run} />
    </div>
  </div>;
}
