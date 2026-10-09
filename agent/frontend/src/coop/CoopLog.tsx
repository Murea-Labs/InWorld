import { t } from '../i18n'
import { useEffect, useMemo, useState } from "react";
import { getLog, RoomRequestError, type LogRow } from "./api";

export type LogStage = "all" | "agent" | "generation";
const LABELS: Record<string, string> = {
  player_action: t("玩家动作"), round_start: t("回合启动"), video_observation_context: t("准备画面审核上下文"),
  agent_request: t("Agent 请求"), agent_result: t("Agent 返回"), action_translation_rejected: t("动作翻译校验失败"),
  action_translation_model_retry: t("重试动作翻译"), llm: t("模型调用"), gemini: t("模型调用"),
  opening_ready: t("开场建议完成"), opening_failed: t("开场建议失败"), opening_discarded: t("开场建议已过期"),
  coop_ai: t("AI 队友动作"), video_request: t("提交视频 Prompt"), video_submit: t("视频服务提交"),
  video_status: t("视频生成进度"), video_ready: t("视频文件就绪"), video_result: t("视频返回"), coop_done: t("回合完成"),
  round_failed: t("回合失败"), portrait_request: t("角色生图请求"), coop_portrait: t("角色生图返回"),
  shared_reference_request: t("共享参考图请求"), shared_reference_failed: t("共享参考图失败"),
  opening_asset_observation: t("序章关键帧读取"), opening_asset_request: t("序章资产生图"),
  opening_asset_extraction: t("序章资产完成"), opening_asset_request_failed: t("序章资产生图失败"),
  opening_asset_extraction_failed: t("序章资产提取失败"),
  opening_setup_ready: t("序章资产准备完成"),
  round_boss_observation: t("读取本轮画面"), round_boss_asset_request: t("新 BOSS 生图"),
  round_boss_asset_request_failed: t("新 BOSS 生图失败"), round_boss_detection: t("本轮 BOSS 检测完成"),
  round_boss_detection_failed: t("本轮 BOSS 检测失败"), round_boss_resume_failed: t("恢复本轮生成失败"),
  coop_action_prefetch_ready: t("角色动作预翻译完成"), coop_action_prefetch_failed: t("角色动作预翻译失败"),
  coop_action_prefetch_discarded: t("角色动作预翻译已过期"),
  coop_recommendation_prompt_ready: t("推荐联合 Prompt 已预编排"),
  coop_speculative_video_ready: t("推荐视频已预生成"),
  coop_recommendation_prefetch_hit: t("命中推荐预生成"), coop_speculative_video_consumed: t("已采用推荐视频"),
  coop_recommendation_prefetch_failed: t("推荐预生成失败"), coop_recommendation_prefetch_fallback: t("回退正常生成"),
  coop_generation_narration: t("生成等待旁白"),
  coop_forecast_ready: t("BOSS/环境预告已确定"),
  coop_story_outline_ready: t("剧情大纲已更新"), coop_story_outline_failed: t("剧情大纲稍后重试"),
  coop_story_progress_ready: t("本轮剧情进度已核对"), coop_story_progress_invalid: t("剧情进度证据待确认"),
  coop_story_guidance_refreshed: t("根据最新剧情重新规划未发布建议"),
  coop_story_guidance_stale: t("剧情目标已变化，重新准备建议"),
  coop_guidance_review_ready: t("已核对最新状态对建议的影响"),
  coop_guidance_refreshed: t("已根据最新进展校正建议"),
  coop_guidance_refresh_failed: t("建议校正暂未完成，保留现有方案"),
  coop_tactic_review_ready: t("已核对战术重复与推进效果"),
  coop_tactic_review_failed: t("战术核验暂未完成，保留现有方案"),
  coop_tactics_repeated: t("部分战术仍重复，已记录待调整"),
  coop_state_updates_invalid: t("部分状态更新证据不足"),
  coop_forecast_reactions_ready: t("针对预告的行动建议完成"),
  coop_duration_selected: t("已根据动作路径与先后关系确定视频时长"),
};
const EXPLANATIONS: Record<string, string> = {
  player_action: t("记录玩家本轮提交的角色动作；只有所有角色都准备好后才会进入联合镜头。"),
  round_start: t("锁定本轮的队伍、动作和参考图，开始编排并生成一段共同视频。"),
  video_observation_context: t("为独立画面审核准备关键帧和文字上下文；普通回合的历史预告不会读取这些关键帧。"),
  agent_request: t("把世界设定、角色资料和当前事实边界发送给 Agent，请它生成建议或编排数据。"),
  agent_result: t("Agent 返回结构化结果；下方可以查看原始返回和校验结果。"),
  action_translation_rejected: t("校验发现动作翻译不是英文或字段格式不符，尚未提交给视频模型。"),
  action_translation_model_retry: t("上一次动作翻译没有通过校验，系统要求 Agent 按错误原因重试一次。"),
  opening_ready: t("序章或新一轮行动建议已经生成，前端现在可以显示给玩家。"),
  opening_failed: t("行动建议生成失败；视频生成不会使用这次不完整的建议。"),
  opening_discarded: t("建议生成期间角色、回合或参考图发生变化，这个过期结果被丢弃。"),
  coop_ai: t("为房间中的 AI 队友生成并保存一个与真人配合的动作。"),
  video_request: t("已完成动作翻译和镜头编排，下面的 prompt 就是实际送入视频模型的完整文本。"),
  video_submit: t("把最终 Prompt 和参考图提交到视频服务。"),
  video_status: t("记录视频服务排队、运行和完成状态。"),
  video_ready: t("视频文件已经返回；下一轮历史预告可能已在视频生成期间并行准备。"),
  video_result: t("视频模型返回了任务结果、视频文件和生成参数。"),
  coop_done: t("视频、关键帧和世界节点都已保存，本轮正式完成。"),
  round_failed: t("本轮在编排、视频生成或保存阶段失败；玩家动作仍会保留以便重试。"),
  portrait_request: t("提交角色参考图生成请求，完成后需要玩家确认采用。"),
  coop_portrait: t("角色参考图生成结果已返回。"),
  shared_reference_request: t("提交场景或怪兽共享参考图生成请求。"),
  shared_reference_failed: t("共享参考图生成失败，已有采用的参考图不会被替换。"),
  opening_asset_observation: t("读取序幕视频关键帧，供 Agent 判断画面中实际出现的场景和 BOSS。"),
  opening_asset_request: t("根据 Agent 选中的序幕关键帧生成空场景或隔离的 BOSS 参考图。"),
  opening_asset_extraction: t("序幕场景和 BOSS 资产已按画面证据写入世界；玩家修改过的资产保持不变。"),
  opening_asset_request_failed: t("单个序章资产生图失败，另一项资产仍会继续处理。"),
  opening_asset_extraction_failed: t("序幕资产提取失败；可以继续手动设置场景和 BOSS。"),
  opening_setup_ready: t("角色参考资产已经就绪；场景和 BOSS 可在序幕后按画面提取，此阶段没有调用行动建议 Agent。"),
  round_boss_observation: t("读取本轮视频关键帧，交给 Agent 判断是否真的出现新的非人类 BOSS。"),
  round_boss_asset_request: t("根据本轮视频中选定的真实关键帧生成纯白背景的 BOSS 固定参考图。"),
  round_boss_asset_request_failed: t("新 BOSS 已被画面识别，但固定参考图生成失败，不影响本轮视频。"),
  round_boss_detection: t("本轮 BOSS 画面审核完成；如果确认出现新 BOSS，资产已绑定到后续联合镜头。"),
  round_boss_detection_failed: t("本轮 BOSS 画面审核失败，继续冒险不会受影响，可在后续视频再次检查。"),
  round_boss_resume_failed: t("BOSS 审核完成后恢复待生成动作失败，需要玩家重新提交或手动生成。"),
  coop_action_prefetch_ready: t("该角色点击准备后，英文动作已在等待队友期间翻译完成。"),
  coop_action_prefetch_failed: t("动作预翻译失败；全员提交后会自动回到完整联合编排。"),
  coop_action_prefetch_discarded: t("玩家动作或世界状态已变化，未使用这份旧翻译。"),
  coop_recommendation_prompt_ready: t("新视频播放时，系统已为全部推荐动作准备好完整联合 Prompt。"),
  coop_speculative_video_ready: t("系统已用推荐动作在后台生成下一段视频，尚未写入世界。"),
  coop_recommendation_prefetch_hit: t("全部玩家原样采用了系统建议，开始接管预编排或预生成结果。"),
  coop_speculative_video_consumed: t("预生成视频的签名与玩家输入完全一致，已正式写入世界。"),
  coop_recommendation_prefetch_failed: t("推荐预处理失败，不会影响玩家的真实动作。"),
  coop_recommendation_prefetch_fallback: t("预生成不可用，已使用同一份玩家输入回到正常联合生成。"),
  coop_generation_narration: t("根据本轮局势变化和按提交顺序排列的角色动作生成短旁白，只在等待视频时播放。"),
  coop_forecast_ready: t("预告与共同目标已通过校验，接下来据此设计每位玩家的动作；完整建议完成后一起发布。"),
  coop_forecast_reactions_ready: t("各玩家建议已绑定到本次预告；下方记录动作对应的事件短语和应对理由。"),
  coop_duration_selected: t("按实际动作步骤及依赖关系估时：并行取最长，先后执行累加，并留出结尾观察时间。记录包含所选时长、估计依据及是否达到配置上限；异常时使用本地估算。"),
};
const stageOf = (row: LogRow) => row.stage ||
  (/^(video|image|portrait|coop_portrait|coop_done|coop_speculative_video|opening_asset_(?:request(?:_failed)?|extraction(?:_failed)?)|round_boss_)/.test(row.kind) ? "generation" : "agent");
const idOf = (row: LogRow) => `${row.t}:${row.kind}:${row.request_id || ""}`;
const roundName = (round?: number) => round === 0 ? t("开场与角色") :
  round == null ? t("早期记录") : t("第 {n} 轮", { n: round });
const pretty = (value: unknown) => typeof value === "string" ? value : JSON.stringify(value, null, 2);
const FIELDS: Record<string, string> = {request:t("完整请求"), response:t("完整返回"), prompt:t("完整 Prompt"),
  error:t("错误"), seconds:t("耗时 / 秒"), usage:t("Token 用量"), params:t("生成参数"), players:t("角色与动作"),
  request_id:t("本次任务 ID"), refs:t("参考图"), payload:t("实际提交参数"), model:t("使用模型"),
  node:t("关联节点"), frames:t("视频关键帧"), clip:t("视频文件"), video_id:t("视频任务 ID"),
  frame:t("选中的关键帧"),
  duration:t("视频时长 / 秒"), duration_policy:t("时长选择依据"), continue_from:t("续写父任务"), require_continuation:t("是否强制续写"),
  queue_seconds:t("排队耗时 / 秒"), player:t("玩家 ID"), character:t("角色"), action:t("动作"),
  tag:t("调用用途"), stage:t("处理阶段"), status:t("状态"), started_at:t("开始时间"), completed_at:t("完成时间"),
  inference_time_s:t("推理耗时 / 秒"), artifact_id:t("续写句柄"), mode:t("游玩模式"), reason:t("丢弃原因")};

export default function CoopLog({ code, initialRound, initialStage }: {
  code: string; initialRound: number | null; initialStage: LogStage;
}) {
  const [rows, setRows] = useState<LogRow[]>([]);
  const [error, setError] = useState("");
  const [round, setRound] = useState<number | null>(initialRound);
  const [stage, setStage] = useState(initialStage);
  const [selected, setSelected] = useState("");
  const [refresh, setRefresh] = useState(0);
  const [loaded, setLoaded] = useState(false);
  useEffect(() => {
    let active = true;
    let timer: ReturnType<typeof setTimeout>;
    let failures = 0;
    const poll = async () => {
      try {
        const next = await getLog(code);
        if (active) {
          setRows(next);
          setError("");
          setLoaded(true);
          failures = 0;
        }
      } catch (e) {
        if (active) {
          failures += 1;
          const transient = e instanceof RoomRequestError && [502, 503, 504].includes(e.status);
          // When the server or a proxy in front of it is briefly unavailable, keep the loaded logs and back off,
          // instead of interrupting the reader with raw status-code errors.
          setError(transient
            ? t("日志服务暂时不可用，正在自动重试…")
            : (e as Error).message);
        }
      } finally {
        if (active) {
          const delay = failures ? Math.min(10000, 1500 * 2 ** Math.min(failures - 1, 3)) : 3000;
          timer = setTimeout(() => void poll(), delay);
        }
      }
    };
    void poll();
    return () => { active = false; clearTimeout(timer); };
  }, [code, refresh]);
  const rounds = useMemo(() => [...new Set(rows.flatMap(r => r.round == null ? [] : [r.round]))]
    .sort((a, b) => b - a), [rows]);
  const visible = rows.filter(r => (round == null || r.round === round) &&
    (stage === "all" || stageOf(r) === stage));
  const entry = visible.find(r => idOf(r) === selected) || visible[0];
  const groups = [...new Set(visible.map(r => r.round))];
  return <div className="coop-log">
    <div className="coop-log-filters">
      <select aria-label={t("日志回合")} value={round ?? "all"}
        onChange={e => { setRound(e.target.value === "all" ? null : Number(e.target.value)); setSelected(""); }}>
        <option value="all">{t("所有回合")}</option>
        {[...new Set([...rounds, ...(round == null ? [] : [round])])].sort((a,b)=>b-a)
          .map(r => <option key={r} value={r}>{roundName(r)}</option>)}
      </select>
      <div className="coop-log-tabs" aria-label={t("日志阶段")}>
        {([ ["all", t("全部")], ["agent", "Agent"], ["generation", t("生成")] ] as const).map(([key, name]) =>
          <button key={key} className={`sm ${stage === key ? "on" : "ghost"}`} aria-pressed={stage === key}
            onClick={() => { setStage(key); setSelected(""); }}>{name}</button>)}
      </div>
      <span className="spacer" />
      <small>{loaded ? t("{n} 条 · 实时更新", { n: visible.length }) : t("读取中…")}</small>
      <button className="sm ghost" onClick={() => setRefresh(v => v + 1)}>{t("刷新")}</button>
    </div>
    {error && <p className="log-err" role="alert">{error}</p>}
    <div className="log-split">
      <nav className="log-nav" aria-label={t("本轮日志记录")}>
        {groups.map(g => <div className="log-group" key={g ?? "legacy"}>
          <div className="log-group-head"><b>{roundName(g)}</b></div>
          {visible.filter(r => r.round === g).map(r => <button key={idOf(r)}
            className={`log-item ${entry === r ? "on" : ""} ${r.ok === false ? "is-bad" : ""}`}
            title={EXPLANATIONS[r.kind] || r.kind}
            onClick={() => setSelected(idOf(r))}>
            <span className="log-kind">{stageOf(r) === "agent" ? "Agent" : t("生成")}</span>
            <span className="log-line">{LABELS[r.kind] || r.kind}</span>
            <span className="log-time">{new Date(r.t * 1000).toLocaleTimeString("zh-CN", {hour12:false})}</span>
          </button>)}
        </div>)}
        {loaded && !visible.length && <p className="log-empty">{t("这个阶段还没有记录。开场编排、发送动作或生成时会自动记录。")}</p>}
      </nav>
      <section className="log-body" aria-label={t("完整日志详情")}>
        {entry ? <>
          <div className="log-detail-head"><b>{LABELS[entry.kind] || entry.kind}</b>
            <span className="log-model">{entry.tag || ""}</span></div>
          <p className="log-step-explanation">{EXPLANATIONS[entry.kind] || t("这是一条系统流水记录，可展开下方字段查看原始数据。")}</p>
          {Object.entries(entry).filter(([k]) => !["t", "kind", "stage", "round", "tag"].includes(k))
            .map(([key, value]) => <div key={key}>
              <div className="log-label">{FIELDS[key] || key}</div>
              <pre className="log-pre">{pretty(value)}</pre>
            </div>)}
        </> : <p className="log-empty">{t("选择一条记录查看完整请求、返回和生成参数。")}</p>}
      </section>
    </div>
    <p className="coop-log-footnote">{t("显示最近 300 条。新版调用正文完整记录；旧版已截断的内容无法补回。")}</p>
  </div>;
}
