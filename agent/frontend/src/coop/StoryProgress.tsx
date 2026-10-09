import { t } from '../i18n'
import type { Room } from "./api";

export default function StoryProgress({ room, clip, compact = false, stageOnly = false, onOpenOutline }: {
  room: Room; clip?: string | null; compact?: boolean; stageOnly?: boolean; onOpenOutline?: () => void;
}) {
  const shown = room.rounds.find(round => round.node.clip === clip);
  const latest = room.rounds[room.rounds.length - 1];
  const latestClip = latest?.node.clip || room.opening_node?.clip;
  // History playback must not show progress from a later clip.
  const historical = !!clip && clip !== latestClip && clip !== room.preview_clip;
  const progress = historical ? shown?.story_progress : room.story_progress;
  const goal = historical ? undefined : room.story_outline?.overall_goal;
  const canOpenOutline = !!room.story_outline?.stages.length && !!onOpenOutline;
  if (!progress?.stage_title && !goal && !canOpenOutline) return null;
  if (stageOnly) return progress?.stage_title || goal ? <div className="coop-work-progress" aria-label={t("剧情进度")}>
    {goal && <div className="coop-work-stage"><span>{t("本关目标")}</span><b title={goal}>{goal}</b></div>}
    {progress?.stage_title && <div className="coop-work-stage">
    <span>{historical ? t("片段阶段") : t("当前阶段")}</span>
    <b title={progress.stage_title}>{progress.stage_title}</b>
    </div>}
  </div> : null;
  return <div className={`coop-story-progress${compact ? " compact" : ""}`} aria-label={t("剧情进度")}>
    {goal && <div className="coop-story-content"><b>{t("本关目标")}</b><span className="coop-story-detail" title={goal}>{goal}</span></div>}
    {progress?.stage_title && <div className="coop-story-content">
      <b className="coop-story-stage" title={progress.stage_title}>{t("剧情进度")} · {progress.stage_title}</b>
      {progress.stage_goal && <span className="coop-story-detail" title={progress.stage_goal}>{progress.stage_goal}</span>}
      {progress.summary && <span className="coop-story-detail coop-story-summary" title={progress.summary}>{progress.summary}</span>}
    </div>}
    {canOpenOutline && <button type="button" className="ghost coop-story-outline-button"
      aria-haspopup="dialog" onClick={onOpenOutline}>
      {historical ? t("查看当前大纲") : t("查看大纲")}
    </button>}
  </div>;
}
