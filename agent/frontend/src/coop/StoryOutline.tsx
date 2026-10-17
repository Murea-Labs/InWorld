import { t as tr } from '../i18n'
import type { Room } from "./api";
import { useStoryDisplay } from "./useStoryDisplay";

const STATUS_LABELS = {
  pending: tr("待推进"), active: tr("未完成"), completed: tr("已完成"), blocked: tr("受阻"), abandoned: tr("已放弃"),
};
const PURPOSE_LABELS = {
  approach: tr("接近目标"), discover: tr("发现线索"), opportunity: tr("创造机会"), breakthrough: tr("突破障碍"),
  transition: tr("转场"), closure: tr("收尾"), observe: tr("确认状态"),
};

export default function StoryOutline({ room }: { room: Room }) {
  const display = useStoryDisplay(room);
  const t = display.text;
  const outline = room.story_outline;
  if (!outline?.stages.length) return <p className="coop-muted">{tr("大纲正在准备中。")}</p>;
  const progress = room.story_progress;
  const active = progress?.active_stage_id ?? outline.active_stage_id;
  const activeIndex = outline.stages.findIndex(stage => stage.id === active);
  const activeStage = outline.stages[activeIndex];
  const shortPlan = room.intro?.short_term_plan;
  const plannedStage = outline.stages.find(stage => stage.id === shortPlan?.stage_id);
  return <div className="coop-outline">
    <div className="coop-outline-goal">
      <span className="eyebrow">{tr("第 {n} 章 · 总体目标", { n: outline.chapter })}</span>
      <h3>{t(outline.overall_goal)}</h3>
      <p className="coop-muted">{tr("当前世界的最新规划，会随实际冒险过程调整。")}</p>
      <div className="coop-outline-meta">
        <span>{tr("{n} 个阶段", { n: outline.stages.length })}</span>
        <span>{outline.reviewed_round ? tr("已结合第 {n} 轮及之前的过程修订", { n: outline.reviewed_round }) : tr("初始大纲")}</span>
        {progress?.last_round != null && <span>{tr("进度已核对至第 {n} 轮", { n: progress.last_round })}</span>}
      </div>
    </div>
    {display.pending && <p className="coop-outline-language" role="status">
      {display.failed ? tr("部分中文说明暂未就绪。") : tr("正在整理旧记录的中文说明…")}
      {display.failed && <button onClick={display.retry}>{tr("重试")}</button>}
    </p>}
    <div className="coop-outline-current">
      <b>{tr("当前阶段")}</b>
      <span>{activeStage ? `${tr("第 {n} 阶段", { n: activeIndex + 1 })} · ${t(activeStage.title)}` : tr("本章目标已完成")}</span>
    </div>
    {(progress?.summary || progress?.next_gap) && <div className="coop-outline-progress">
      {progress.summary && <p><b>{tr("最新进度")}</b>{t(progress.summary)}</p>}
      {progress.next_gap && <p><b>{tr("尚待完成")}</b>{t(progress.next_gap)}</p>}
    </div>}
    {shortPlan && <section className="coop-outline-near-term" aria-label={tr("行动安排")}>
      <div className="coop-outline-section-head">
        <h3>{tr("行动安排")}</h3><span>{PURPOSE_LABELS[shortPlan.purpose]}</span>
      </div>
      {plannedStage && <p className="coop-muted">{tr("服务于第 {n} 阶段", { n: outline.stages.indexOf(plannedStage) + 1 })} · {t(plannedStage.title)}</p>}
      <p className="coop-outline-plan-note">{tr("本轮只尝试第一步，后续安排待画面确认条件后调整。")}</p>
      <ul className="coop-outline-actions">
        {shortPlan.steps.map((step, index) => <li key={step.id} className={`coop-outline-action ${index === 0 ? "is-now" : "is-later"}`}>
          <span className="coop-outline-action-label">{index === 0 ? tr("本轮尝试") : tr("后续待确认")}</span>
          <h4>{t(step.goal)}</h4>
          <p>{t(step.cooperation)}</p>
          <details>
            <summary>{tr("条件与备选方案")}</summary>
            <dl>
              <dt>{tr("所需条件")}</dt><dd>{t(step.prerequisite)}</dd>
              <dt>{tr("观察目标")}</dt><dd>{t(step.completion_evidence)}</dd>
              <dt>{tr("受阻时")}</dt><dd>{t(step.if_blocked)}</dd>
            </dl>
          </details>
        </li>)}
      </ul>
      <details className="coop-outline-plan-reason">
        <summary>{tr("安排依据")}</summary><p>{t(shortPlan.change_reason)}</p>
      </details>
    </section>}
    <section className="coop-outline-global" aria-label={tr("全局大纲")}>
      <div className="coop-outline-section-head"><h3>{tr("全局大纲")}</h3><span>{tr("{n} 个剧情阶段", { n: outline.stages.length })}</span></div>
      <p className="coop-muted">{tr("从开始到完成总体目标的阶段路线。")}</p>
      <ol className="coop-outline-stages">
        {outline.stages.map((stage, index) => {
          const state = progress?.stages?.[stage.id];
          const status = state?.status ?? (stage.id === active ? "active" : "pending");
          const current = stage.id === active && status !== "completed" && status !== "abandoned";
          return <li key={stage.id} className={`coop-outline-stage is-${status}`}
            aria-current={current ? "step" : undefined}>
            <div className="coop-outline-stage-head">
              <span className="coop-outline-number">{index + 1}</span>
              <h3>{t(stage.title)}</h3>
              <span className="coop-outline-status">{current
                ? (status === "blocked" ? tr("当前 · 受阻") : tr("当前 · 进行中"))
                : STATUS_LABELS[status]}</span>
            </div>
            <p className="coop-outline-stage-goal">{t(stage.goal)}</p>
            <dl>
              <dt>{tr("完成条件")}</dt><dd>{t(stage.completion_evidence)}</dd>
              {stage.alternative && <><dt>{tr("备选方向")}</dt><dd>{t(stage.alternative)}</dd></>}
              {state?.evidence && <><dt>{tr("观察证据")}</dt><dd>{t(state.evidence)}</dd></>}
              {state?.remaining && status !== "completed" && status !== "abandoned" &&
                <><dt>{current ? tr("尚待完成") : tr("观察备注")}</dt><dd>{t(state.remaining)}</dd></>}
              {state?.reason && <><dt>{tr("调整原因")}</dt><dd>{t(state.reason)}</dd></>}
            </dl>
          </li>;
        })}
      </ol>
    </section>
  </div>;
}
