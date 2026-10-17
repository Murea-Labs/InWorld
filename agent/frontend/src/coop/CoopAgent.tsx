import { t } from '../i18n'
import { useState } from "react";
import AgentTrace from "../components/AgentTrace";
import type { Room } from "./api";

/** The agent steps of the round being made (live) or of any finished clip. */
export default function CoopAgent({ room, busy }: { room: Room; busy: boolean }) {
  const clips = [
    ...(room.opening_node ? [{ label: t("序章"), node: room.opening_node }] : []),
    ...room.rounds.map((round) => ({ label: t("第 {n} 轮", { n: round.round }), node: round.node })),
  ];
  const live = busy && (room.agent_trace?.length ?? 0) > 0;
  const [picked, setPicked] = useState(clips.length - 1);
  const index = Math.min(Math.max(picked, 0), clips.length - 1);
  if (live) {
    return (
      <div className="coop-agent">
        <p className="coop-agent-head">{t("正在生成第 {n} 轮", { n: room.generation_round ?? room.round })}</p>
        <AgentTrace steps={room.agent_trace ?? []} live />
      </div>
    );
  }
  if (!clips.length) return <p className="trace-empty">{t("还没有生成过镜头。")}</p>;
  return (
    <div className="coop-agent">
      <label className="coop-agent-head">
        {t("查看")}
        <select value={index} onChange={(event) => setPicked(Number(event.target.value))}>
          {clips.map((clip, i) => <option key={clip.node.id} value={i}>{clip.label}</option>)}
        </select>
      </label>
      <AgentTrace steps={clips[index].node.trace ?? []} />
    </div>
  );
}
