import { t } from '../i18n'
import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type FormEvent,
} from "react";
import { getHealth, mediaUrl } from "../api";
import type { User } from "../types";
import { loadTts, saveTts, speak, stopSpeaking } from "../tts";
import * as api from "./api";
import type { Player, Room, RoomSummary } from "./api";
import { MUREA_THEMES, type MureaTheme } from "../theme";
import { markFilePickerOpen } from "../filepicker";
import CoopLog, { type LogStage } from "./CoopLog";
import CoopAgent from "./CoopAgent";
import { PHASES, PHASE_LABEL } from "../components/AgentTrace";
import type { TracePhase, TraceStep } from "../types";
import SharedVideo from "./SharedVideo";
import StoryProgress from "./StoryProgress";
import StoryOutline from "./StoryOutline";
import useImmersive from "./useImmersive";
import WorldAssets from "./WorldAssets";
import CoopLobby, { DuoIcon } from "./CoopLobby";
import { ActionComposer, CoopDialog, PortraitPreview, letter } from "./RoomParts";
import Dialog from "../dialogs/Dialog";
import "./coop.css";


function orderedActionPlayers(players: Player[], order: string[] = []) {
  const rank = new Map(order.map((userId, index) => [userId, index]));
  return [...players].sort((a, b) => {
    const left = rank.get(a.user_id) ?? Number.MAX_SAFE_INTEGER;
    const right = rank.get(b.user_id) ?? Number.MAX_SAFE_INTEGER;
    return left - right;
  });
}





export default function CoopRoom({
  user,
  code,
  onExit,
  onSwitchUser,
  theme,
  onThemeChange,
}: {
  theme: MureaTheme;
  onThemeChange: (theme: MureaTheme) => void;
  user: User;
  code: string;
  onExit: () => void;
  onSwitchUser: () => void;
}) {
  const [room, setRoom] = useState<Room | null>(null);
  const [rooms, setRooms] = useState<RoomSummary[]>([]);
  const [aiProfile, setAIProfile] = useState("axe");
  const [dirtySeats, setDirtySeats] = useState<Record<string, boolean>>({});
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [working, setWorking] = useState("");
  const [deleting, setDeleting] = useState<Pick<RoomSummary, "code" | "title" | "mode"> | null>(null);
  const [deleteNote, setDeleteNote] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(!!code);
  const [invited, setInvited] = useState(false);
  const [copied, setCopied] = useState(false);
  const [panel, setPanel] = useState<
    "" | "character" | "scene" | "scene-edit" | "monster" | "prompt" | "history" | "invite" | "log" | "outline" | "agent"
  >("");
  const [logRound, setLogRound] = useState<number | null>(null);
  const [logStage, setLogStage] = useState<LogStage>("all");
  const [portraitTarget, setPortraitTarget] = useState("");
  const [mobileParty, setMobileParty] = useState(false);
  const [mobileJournal, setMobileJournal] = useState(false);
  const [journalTab, setJournalTab] = useState<"actions" | "outline">("actions");
  const [playingClip, setPlayingClip] = useState<string | null>(null);
  const [selectedClip, setSelectedClip] = useState<string | null>(null);
  const [undoNode, setUndoNode] = useState<string | null>(null);
  const [activeActionUserId, setActiveActionUserId] = useState<string | null>(null);
  const [tts, setTts] = useState<boolean | null>(null);
  const [ttsUp, setTtsUp] = useState(true);
  const [speaking, setSpeaking] = useState(false);
  const spokenNarration = useRef("");
  const alive = useRef(true);
  const silenceNarration = useCallback(() => {
    stopSpeaking();
    setSpeaking(false);
  }, []);
  useEffect(() => {
    alive.current = true;
    stopSpeaking();
    void getHealth().then((health) => {
      if (!alive.current) return;
      setTts(health.tts ? loadTts() : null);
      setTtsUp(health.tts_up !== false);
    }).catch(() => {
      if (alive.current) setTts(null);
    });
    return () => {
      alive.current = false;
      stopSpeaking();
    };
  }, []);

  const apply = useCallback(
    (next: Room) =>
      setRoom((prev) =>
        prev?.code === next.code && prev.revision >= next.revision ? prev : next,
      ),
    [],
  );
  const { immersive, exiting, start: startImmersive, stop: stopImmersive } =
    useImmersive(code, room, playingClip, apply, setError);
  const clipChanged = useCallback((src: string) => {
    setPlayingClip(decodeURIComponent(src.split("/").pop() || ""));
  }, []);
  useEffect(() => {
    let active = true;
    let timer: ReturnType<typeof setTimeout>;
    let stopWatching: (() => void) | undefined;
    let streamHealthy = false;
    let pollError = "";
    setRoom(null);
    setRooms([]);
    setPanel("");
    setMobileParty(false);
    setMobileJournal(false);
    setJournalTab("actions");
    setPlayingClip(null);
    setSelectedClip(null);
    setError("");
    setInvited(false);
    setLoading(!!code);
    const poll = async () => {
      try {
        if (code) {
          const next = await api.getRoom(code);
          if (active) {
            apply(next);
            if (pollError) setError((current) => current === pollError ? "" : current);
            if (!stopWatching) stopWatching = api.watchRoom(code,
              (update) => { if (active) apply(update); },
              () => { if (active) { setRoom(null); window.location.hash = "coop"; } },
              (connected) => { streamHealthy = connected;
                if (connected && active && pollError) setError((current) => current === pollError ? "" : current);
                if (!connected && active) {
                clearTimeout(timer); timer = setTimeout(() => void poll(), 1000);
              } });
          }
        } else {
          const next = await api.listRooms();
          if (active) setRooms(next);
        }
      } catch (e) {
        if (active) {
          if (code && e instanceof api.RoomRequestError && [404, 410].includes(e.status)) {
            setRoom(null);
            window.location.hash = "coop";
          } else if (code && e instanceof api.RoomRequestError && e.status === 403) {
            setInvited(true);
          } else {
            pollError = (e as Error).message;
            setError(pollError);
          }
        }
      } finally {
        if (active) {
          setLoading(false);
          if (code && (!stopWatching || !streamHealthy)) timer = setTimeout(() => void poll(), 3000);
        }
      }
    };
    void poll();
    return () => {
      active = false;
      clearTimeout(timer);
      stopWatching?.();
    };
  }, [code, user.id, apply]);

  const me = room?.members.find((p) => p.user_id === user.id);
  const local = room?.mode === "local";
  const actionPlayers = room?.members.filter((p) => local || p.user_id === user.id) || [];
  useEffect(() => {
    const ids = actionPlayers.map((p) => p.user_id);
    setActiveActionUserId((current) =>
      current && ids.includes(current) ? current : ids[0] || null,
    );
  // eslint-disable-next-line react-hooks/exhaustive-deps -- actionPlayers is derived from the listed values
  }, [room?.code, room?.members, local, user.id]);
  useEffect(() => {
    setDirtySeats({});
  }, [room?.code, room?.round]);
  const hasDraftChanges = Object.values(dirtySeats).some(Boolean);
  const selectedPlayer = portraitTarget
    ? room?.members.find((p) => p.user_id === portraitTarget)
    : me;
  useEffect(() => {
    setName(selectedPlayer?.character || "");
    setDescription(selectedPlayer?.description || "");
    setFile(null);
  }, [
    room?.code,
    selectedPlayer?.user_id,
    selectedPlayer?.character,
    selectedPlayer?.description,
    selectedPlayer?.ref,
  ]);
  const busy = room?.status === "generating";
  const host = room?.host_id === user.id;
  const ready = room?.members.filter((p) => p.ready).length || 0;
  const allReady =
    !!room && room.members.length >= 2 && ready === room.members.length;
  const latest = room?.rounds[room.rounds.length - 1];
  const currentVideo = latest?.node || room?.opening_node;
  const playbackClip = immersive ? currentVideo?.clip
    : selectedClip || room?.preview_clip || currentVideo?.clip;
  useEffect(() => {
    // A newly completed clip becomes the default view. Keep an explicitly
    // selected older clip until the room advances or a new clip arrives.
    setSelectedClip(null);
  }, [room?.code, currentVideo?.clip]);
  useEffect(() => {
    if (!room || room.status !== "generating" || immersive || room.generation_cancelling) {
      if (spokenNarration.current || speaking) silenceNarration();
      spokenNarration.current = "";
      return;
    }
    // preview_clip is published as soon as the video model has returned the playable file,
    // before frame extraction and room finalization. Stop here so narration
    // cannot overlap the video's own soundtrack.
    if (room.preview_clip) {
      if (spokenNarration.current || speaking) silenceNarration();
      return;
    }
    const narration = room.generation_narration?.trim();
    if (!narration || tts !== true || !ttsUp) return;
    const key = `${room.world_id}:${room.generation_round}:${room.opening_generation ? "opening" : "round"}:${narration}`;
    if (spokenNarration.current === key) return;
    spokenNarration.current = key;
    void speak(narration, setSpeaking).catch((failure: Error) => {
      if (!alive.current) return;
      if (failure?.message === "autoplay-blocked") {
        setError(t("浏览器拦下了自动旁白，请先在页面上点击一下；下一轮会继续尝试。"));
        return;
      }
      setTtsUp(false);
      setError(t("语音服务暂时无法连接，本轮继续生成视频。"));
    });
  // eslint-disable-next-line react-hooks/exhaustive-deps -- speak once per narration key, not on every room update
  }, [room?.world_id, room?.status, room?.generation_round,
      room?.opening_generation, room?.generation_narration, room?.preview_clip,
      speaking, silenceNarration, tts, ttsUp, immersive, room?.generation_cancelling]);
  const fragmentNodes = useMemo(() => {
    if (!room) return [];
    const nodes = [
      room.opening_node?.clip
        ? { key: `opening:${room.opening_node.clip}`, label: t("序章"), clip: room.opening_node.clip }
        : null,
      ...[...room.rounds]
        .sort((a, b) => a.round - b.round)
        .filter((round) => !!round.node.clip)
        .map((round) => ({
          key: `round:${round.round}:${round.node.clip}`,
          label: t("第 {n} 轮", { n: round.round }),
          clip: round.node.clip as string,
        })),
    ];
    return nodes.filter((node): node is { key: string; label: string; clip: string } => !!node);
  // eslint-disable-next-line react-hooks/exhaustive-deps -- the reel changes only with its clips
  }, [room?.opening_node?.clip, room?.rounds]);
  const videoPlaylist = useMemo(() => {
    if (!room) return [];
    const clips = [
      room.opening_node?.clip,
      ...[...room.rounds]
        .sort((a, b) => a.round - b.round)
        .map((round) => round.node.clip),
    ];
    return clips.filter((clip): clip is string => !!clip).map((clip) => mediaUrl("clips", clip));
  // eslint-disable-next-line react-hooks/exhaustive-deps -- the reel changes only with its clips
  }, [room?.opening_node?.clip, room?.rounds]);
  const guidance = (!busy || room?.generation_cancelling) && room?.intro?.status === "ready" ? room.intro : null;
  const opening = !currentVideo ? guidance : null;
  const retryingOpening = room?.status === "failed" && !!room.opening_generation;
  // Before the opening clip there is no observed scene for a next-turn
  // forecast. Keep the initial narration/objective visible, but hide the
  // post-video world event and per-character suggestions so they cannot be
  // mistaken for actions that the prologue will perform.
  const roundGuidance = currentVideo ? guidance : null;
  const immersiveGuidance = room?.intro?.status === "ready" ? room.intro : null;
  const immersivePrefetchStatus = room?.recommendation_prefetch?.status;
  const immersiveStatus = busy
    ? (room?.progress_stage === "generation"
      ? t("视频生成中：{s}", { s: room.progress || t("正在渲染共同镜头") })
      : t("正在编排下一段：{s}", { s: room?.progress || t("准备联合指令") }))
    : room?.intro?.status === "generating"
      ? t("正在根据上一段画面准备下一步建议…")
      : immersivePrefetchStatus === "planning"
        ? t("正在预编排推荐动作的联合 Prompt…")
        : immersivePrefetchStatus === "prompt_ready"
          ? t("推荐动作 Prompt 已准备，正在预生成视频…")
          : immersivePrefetchStatus === "rendering"
            ? t("正在预生成下一段视频…")
            : immersivePrefetchStatus === "ready"
              ? t("下一段视频已准备好，正在自动接续…")
              : !room?.intro
                ? t("正在同步下一步建议…")
                : t("沉浸模式已开启 · 等待下一步建议");

  const openLogs = (round: number | null = null, stage: LogStage = "all") => {
    setLogRound(round); setLogStage(stage); setPanel("log");
  };
  const go = (next: Room) => {
    apply(next);
    window.location.hash = `coop/${next.code}`;
  };
  const lobby = () => {
    silenceNarration();
    setRoom(null);
    window.location.hash = "coop";
  };
  const exitToLobby = () => {
    silenceNarration();
    onExit();
  };
  const act = async (label: string, fn: () => Promise<void>) => {
    setWorking(label);
    setError("");
    try {
      await fn();
    } catch (e) {
      if (room && e instanceof api.RoomRequestError && e.code === "generation_inputs_changed") {
        try {
          const next = await api.getRoom(room.code);
          if (alive.current) {
            apply(next);
            setError(t("本轮生成内容已变化，已同步最新房间，请核对后再生成。"));
          }
        } catch {
          if (alive.current) setError(t("本轮生成内容已变化，暂未同步到最新房间，请刷新后核对再生成。"));
        }
        return;
      }
      if (alive.current) setError((e as Error).message);
    } finally {
      if (alive.current) setWorking("");
    }
  };
  const askDelete = (target: RoomSummary) => {
    setPanel("");
    setDeleteNote("");
    setDeleting(target);
  };
  const deleteWorld = async () => {
    if (!deleting || working) return;
    setWorking("delete");
    setDeleteNote("");
    try {
      await api.deleteRoom(deleting.code);
      setRooms((current) => current.filter((r) => r.code !== deleting.code));
      setDeleting(null);
      if (code === deleting.code) lobby();
    } catch (e) {
      setDeleteNote((e as Error).message);
    } finally {
      if (alive.current) setWorking("");
    }
  };
  const saveCharacter = async (event: FormEvent) => {
    event.preventDefault();
    if (!room) return;
    await act("character", async () => {
      const body = new FormData();
      body.set("name", name);
      body.set("description", description);
      if (portraitTarget) body.set("target_uid", portraitTarget);
      if (file) body.set("file", file);
      apply(await api.setReference(room.code, "character", body));
      setPanel("");
      setFile(null);
    });
  };
  const drawPortrait = (randomize: boolean, target_uid?: string) => {
    if (!room) return;
    void act("portrait", async () =>
      apply(
        await api.generatePortrait(room.code, {
          randomize,
          target_uid,
          name: randomize ? undefined : name,
          description: randomize ? undefined : description,
        }),
      ),
    );
  };
  const copyInvite = () =>
    act("copy", async () => {
      if (!room) return;
      const url = new URL(window.location.href);
      url.hash = `coop/${room.code}`;
      if (!navigator.clipboard) {
        setError(t("邀请码：{code}。请选中上方邀请码复制。", { code: room.code }));
        return;
      }
      await navigator.clipboard.writeText(room.invite_url || url.toString());
      setCopied(true);
      setTimeout(() => setCopied(false), 2500);
    });

  const openCharacter = (target = "") => {
    setPortraitTarget(target);
    setPanel("character");
  };
  const statusText = busy
    ? t("全员已发送 · 自动生成中")
    : hasDraftChanges
      ? t("发送动作后，等待队友一起出发")
      : !allReady
        ? t("全员发送后自动生成并播放")
        : !room?.generation_enabled
          ? t("联合指令已就绪 · 视频暂未开启")
        : room.status === "failed" ? t("生成未完成 · 动作已保留") : t("全员已发送 · 即将生成");
  const sharedAssetItems = room ? [
    { kind: "scene" as const, label: t("场景"), name: room.scene_name, ref: room.scene_ref },
    { kind: "monster" as const, label: "BOSS", name: room.monster?.name || "", ref: room.monster?.ref || "" },
  ] : [];

  return (
    <div id="app" className={`coop-app${immersive ? " immersive" : ""}`} data-theme={theme}>
      {immersive && <button className="ghost sm coop-immersive-exit"
        title={t("退出沉浸模式（Esc）")} onClick={() => void stopImmersive()}>{t("退出沉浸")}</button>}
      <div className="coop-shell">
        <header className="coop-topbar">
          <button className="brand ghost coop-brand" onClick={exitToLobby}>
            <h1>Murea Agent</h1>
          </button>
          <span className="head-sep" />
          <span className="coop-mode">
            <DuoIcon />
            {t("双人模式")}
          </span>
          {room && (
            <button
              className="ghost sm coop-world-title"
              onClick={() => setPanel("scene")}
              title={room.title}
            >
              {room.title}
              <span>⌄</span>
            </button>
          )}
          <div className="spacer" />
          <div className="coop-top-actions">
            <button
              className={`sm tts-btn${tts ? " on" : " ghost"}${speaking ? " speaking" : ""}`}
              disabled={tts === null || !ttsUp}
              onClick={() => {
                if (tts === null) return;
                const next = !tts;
                saveTts(next);
                if (!next) {
                  spokenNarration.current = "";
                  silenceNarration();
                }
                setTts(next);
              }}
              title={
                tts === null
                  ? t("后端没有配置 TTS，语音不可用")
                  : !ttsUp
                    ? t("语音服务暂时无法连接")
                    : tts
                      ? t("关闭生成等待旁白")
                      : t("视频生成时朗读本轮局势与角色动作")
              }
            >
              <span className="tts-speaker" aria-hidden="true">
                <svg viewBox="0 0 24 24">
                  <path className="tts-speaker-body" d="M3.5 9h3.2l4.6-3.7v13.4L6.7 15H3.5z" />
                  <path className="tts-speaker-wave wave-near" d="M14.2 9.1a4 4 0 0 1 0 5.8" />
                  <path className="tts-speaker-wave wave-far" d="M17.1 6.5a7.5 7.5 0 0 1 0 11" />
                </svg>
              </span>
              {t("语音")}
            </button>
            <select
              className="coop-theme"
              aria-label={t("界面主题")}
              value={theme}
              onChange={(e) => onThemeChange(e.target.value as MureaTheme)}
            >
              {MUREA_THEMES.map((t) => (
                <option key={t.id} value={t.id}>
                  {t.label}
                </option>
              ))}
            </select>
            <button
              className="sm head-user"
              disabled={exiting}
              onClick={async () => {
                silenceNarration();
                await stopImmersive();
                onSwitchUser();
              }}
              title={t("切换身份")}
              aria-label={t("切换身份：{name}", { name: user.name })}
            >
              <span>{user.name.slice(0, 1)}</span>
              <b>{user.name}</b>
            </button>
            {code && (
              <button className="sm" onClick={exitToLobby} title={t("返回多人大厅")}>
                {t("返回大厅")}
              </button>
            )}
          </div>
        </header>

        {error && (
          <div className="coop-error coop-toast" role="alert">
            <span>{error}</span>
            <button className="ghost sm" onClick={() => setError("")}>
              {t("关闭")}
            </button>
          </div>
        )}
        {loading ? (
          <div className="coop-loading">{t("正在寻找你的队伍…")}</div>
        ) : !room ? (
          <CoopLobby user={user} code={code} invited={invited} rooms={rooms} working={working}
            act={act} go={go} askDelete={askDelete} />
        ) : (
          <main className={`coop-workspace ${mobileParty ? "show-party" : ""} ${mobileJournal ? "show-journal" : ""}`}>
            <div className="coop-workbar">
              <button className="ghost sm" onClick={lobby}>
                ← <span className="coop-back-label">{t("我的世界")}</span>
              </button>
              <span className="coop-round">
                <span className="coop-work-mode">{local ? t("本地双人") : t("联机房间")} · </span>{t("第 {n} 轮", { n: room.round })}
              </span>
              <StoryProgress room={room} clip={playingClip || playbackClip} stageOnly />
              <div className="spacer" />
              <button
                className="ghost sm"
                aria-label={t("背景设定")}
                onClick={() => setPanel("scene")}
              >
                {t("背景设定")}
              </button>
              <button className="ghost sm" onClick={() => setPanel("history")}>
                {t("历史")} <span>{room.rounds.length}</span>
              </button>
              <button className="ghost sm" onClick={() => setPanel("agent")}
                title={t("看 agent 每一轮怎么做的：感知、规划、检查、行动、提交、反思")}>
                Agent
              </button>
              {host && (room.rounds.length > 0 || room.opening_node) && <button
                className="ghost sm" disabled={busy || !!working || immersive}
                onClick={() => { setError(""); setUndoNode(currentVideo!.id); }}>
                {t("撤回最新一段")}
              </button>}
              <button className="ghost sm coop-log-entry" onClick={() => openLogs()}>{t("后台日志")}</button>
              {!immersive && <button className="ghost sm coop-immersive-toggle"
                aria-pressed={false}
                disabled={exiting || !!room.generation_cancelling}
                onClick={() => {
                  setMobileJournal(false); setMobileParty(false);
                  setSelectedClip(null); setError(""); startImmersive();
                }}
                title={t("进入沉浸模式")}>
                {t("沉浸模式")}
              </button>}
              {!local && (
                <button className="sm" onClick={() => setPanel("invite")}>
                  {copied ? t("链接已复制 ✓") : t("邀请队友")}
                </button>
              )}
              <button
                className="sm coop-mobile-toggle"
                aria-expanded={mobileJournal}
                onClick={() => { setMobileJournal((v) => !v); setMobileParty(false); }}
              >
                {mobileJournal ? t("返回画面") : t("行动 / 大纲")}
              </button>
              <button
                className="sm coop-mobile-toggle"
                aria-expanded={mobileParty}
                onClick={() => { setMobileParty((v) => !v); setMobileJournal(false); }}
              >
                {mobileParty ? t("返回画面") : t("队伍 {n}", { n: room.members.length })}
              </button>
            </div>
            <section
              className="coop-panel coop-action-chat"
              aria-label={t("世界记录")}
              style={room.scene_ref ? {
                backgroundImage: `linear-gradient(rgba(7, 13, 19, .82), rgba(7, 13, 19, .96)), url("${mediaUrl("refs", room.scene_ref)}")`,
              } : undefined}
            >
              <div className="coop-action-chat-head">
                <div className="coop-journal-tabs" role="tablist" aria-label={t("世界记录")}>
                  {(["actions", "outline"] as const).map((tab, index) => <button
                    key={tab} type="button" role="tab" id={`coop-journal-tab-${tab}`}
                    aria-selected={journalTab === tab} aria-controls={`coop-journal-panel-${tab}`}
                    tabIndex={journalTab === tab ? 0 : -1}
                    onClick={() => setJournalTab(tab)}
                    onKeyDown={(event) => {
                      if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
                      event.preventDefault();
                      const next = event.key === "Home" ? 0 : event.key === "End" ? 1 : 1 - index;
                      setJournalTab(next === 0 ? "actions" : "outline");
                      event.currentTarget.parentElement?.querySelectorAll<HTMLButtonElement>('[role="tab"]')[next]?.focus();
                    }}
                  >{tab === "actions" ? t("行动记录") : t("大纲")}</button>)}
                </div>
                <span>{journalTab === "actions" ? t("所有轮次") : t("当前世界")}</span>
              </div>
              <div className="coop-action-chat-list coop-journal-panel" role="tabpanel"
                id="coop-journal-panel-actions" aria-labelledby="coop-journal-tab-actions"
                hidden={journalTab !== "actions"} tabIndex={0}>
                {!room.rounds.length && !room.members.some((p) => p.action.trim()) && (
                  <p className="coop-action-chat-empty">{t("提交动作后，所有人的行动会显示在这里。")}</p>
                )}
                {[...room.rounds]
                  .sort((a, b) => a.round - b.round)
                  .map((round) => (
                    <div className="coop-action-chat-round" key={`history:${round.round}`}>
                      <div className="coop-action-chat-round-label">{t("第 {n} 轮", { n: round.round })}</div>
                      {round.inputs.map((entry) => (
                        <div className="coop-action-chat-message" key={`${round.round}:${entry.user_id}`}>
                          <div className="coop-action-chat-person">
                            {entry.ref ? (
                              <img src={mediaUrl("refs", entry.ref)} alt="" />
                            ) : (
                              <span className="coop-action-chat-avatar">{entry.character?.slice(0, 1) || "·"}</span>
                            )}
                            <div>
                              <b>{entry.character || t("角色未命名")}</b>
                            </div>
                            <time>{t("第 {n} 轮", { n: round.round })}</time>
                          </div>
                          <p>{entry.action || t("（未填写动作）")}</p>
                        </div>
                      ))}
                    </div>
                  ))}
                {room.members.some((p) => p.action.trim()) && (
                  <div className="coop-action-chat-round current">
                    <div className="coop-action-chat-round-label">{t("本轮")}</div>
                    {orderedActionPlayers(room.members, room.action_order)
                      .filter((p) => p.action.trim())
                      .map((entry) => (
                        <div className="coop-action-chat-message" key={`current:${entry.user_id}`}>
                          <div className="coop-action-chat-person">
                            {entry.ref ? (
                              <img src={mediaUrl("refs", entry.ref)} alt="" />
                            ) : (
                              <span className="coop-action-chat-avatar">{entry.character?.slice(0, 1) || "·"}</span>
                            )}
                            <div>
                              <b>{entry.character || t("角色未命名")}</b>
                            </div>
                            <time>{t("刚刚")}</time>
                          </div>
                          <p>{entry.action}</p>
                        </div>
                      ))}
                  </div>
                )}
              </div>
              <div className="coop-action-chat-list coop-journal-panel coop-journal-outline" role="tabpanel"
                id="coop-journal-panel-outline" aria-labelledby="coop-journal-tab-outline"
                hidden={journalTab !== "outline"} tabIndex={0}>
                <StoryOutline room={room} />
              </div>
            </section>
            <div className="coop-play-main">
              <div className="coop-video-guidance">
                <section className="coop-screen" aria-label={t("共同画面")}>
                {playbackClip ? (
                  <SharedVideo
                    src={mediaUrl("clips", playbackClip)}
                    playlist={videoPlaylist}
                    immersive={immersive}
                    onClipChange={clipChanged}
                    onPlay={silenceNarration}
                  />
                ) : (
                  <div className={`coop-stage-empty ${opening ? "has-opening" : ""}`}>
                    <div className="coop-stage-intro">
                      <div className="coop-stage-title">
                        <span className="eyebrow">{opening ? t("序章准备") : "YOUR SHARED WORLD"}</span>
                        <h2>{room.title}</h2>
                      </div>
                      <p className="coop-setting">{opening?.narration || room.setting}</p>
                    </div>
                    <div className="coop-cast">
                      {room.members.map((p, i) => (
                        <button
                          key={p.user_id}
                          className="coop-cast-card"
                          disabled={
                            !local &&
                            p.user_id !== user.id &&
                            !(host && p.kind === "ai")
                          }
                          onClick={() =>
                            openCharacter(
                              local || p.kind === "ai" ? p.user_id : "",
                            )
                          }
                          aria-label={t("设置角色 {name}", { name: p.character || letter(i) })}
                        >
                          {p.ref ? (
                            <img
                              src={mediaUrl("refs", p.ref)}
                              alt={p.character}
                            />
                          ) : (
                            <div className="coop-cast-placeholder">
                              <DuoIcon />
                              <span>{t("设置形象")}</span>
                            </div>
                          )}
                          <span className="coop-cast-letter">{letter(i)}</span>
                          <strong>{p.character || t("你的角色")}</strong>
                        </button>
                      ))}
                      {room.members.length < 2 && (
                        <button
                          className="coop-cast-card coop-cast-waiting"
                          onClick={() => {
                            setPanel("invite");
                          }}
                        >
                          <DuoIcon />
                          <span>{t("等待你的搭档")}</span>
                        </button>
                      )}
                      <button
                        className="coop-cast-card coop-stage-asset-card"
                        onClick={() => setPanel("scene-edit")}
                        aria-label={t("设置场景图")}
                      >
                        {room.scene_ref ? (
                          <img src={mediaUrl("refs", room.scene_ref)} alt={room.scene_name || t("场景图")} />
                        ) : (
                          <div className="coop-cast-placeholder">
                            <span className="coop-stage-assets-icon">✦</span>
                            <span>{room.asset_previews?.scene?.status === "generating" ? t("正在生成") : t("设置场景图")}</span>
                          </div>
                        )}
                        <span className="coop-cast-letter">{t("场景")}</span>
                        <strong>{room.scene_name || t("场景图")}</strong>
                      </button>
                      <button
                        className="coop-cast-card coop-stage-asset-card"
                        onClick={() => setPanel("monster")}
                        aria-label={t("设置 BOSS 图")}
                      >
                        {room.monster?.ref ? (
                          <img src={mediaUrl("refs", room.monster.ref)} alt={room.monster.name || t("BOSS图")} />
                        ) : (
                          <div className="coop-cast-placeholder">
                            <span className="coop-stage-assets-icon">◈</span>
                            <span>{t("自主创建")}</span>
                          </div>
                        )}
                        <span className="coop-cast-letter">BOSS</span>
                        <strong>{room.monster?.name || t("BOSS图（可选）")}</strong>
                      </button>
                    </div>
                    {opening && host && (
                      <button
                        className="primary coop-start-adventure"
                        disabled={!!working || busy || !room.generation_enabled}
                        onClick={() => void act("start-opening", async () =>
                          apply(await (retryingOpening ? api.generate(room) : api.startOpening(room.code)))
                        )}
                      >
                        {working === "start-opening"
                          ? retryingOpening ? t("正在重试序章…") : t("正在开启冒险…")
                          : retryingOpening ? t("重试序章视频") : t("开启冒险")}
                        <span>↗</span>
                      </button>
                    )}
                    <p className="coop-stage-hint">
                      {room.intro?.status === "generating"
                        ? t("正在准备角色资产…")
                        : opening
                          ? t("未设置的场景和 BOSS 会在序幕后根据画面提取；点击“开启冒险”生成序章视频。")
                        : !me?.ref
                        ? t("点击角色，生成或上传你的形象")
                        : t("全员形象准备完成后，点击“开启冒险”生成序章视频；共享资产可在序幕后提取")}
                    </p>
                    {room.intro?.status === "failed" && <div className="coop-intro-error">
                      <span>{room.intro.error}</span>
                      {host && <button className="sm ghost" disabled={!!working || busy}
                        onClick={() => void act("intro", async () => apply(await api.retryIntro(room.code)))}>{t("重试开场")}</button>}
                    </div>}
                  </div>
                )}
                {busy && !room.generation_cancelling && !room.preview_clip && !immersive && (
                  <div className="coop-generating" role="status">
                    <span className="coop-spinner" />
                    <h3>{t("正在生成共同的镜头")}</h3>
                    {!!room.agent_trace?.length && <AgentPhases steps={room.agent_trace} />}
                    <p>{room.progress}</p>
                  </div>
                )}
                {immersive && (
                  <div className="coop-immersive-hud" role="status" aria-live="polite">
                    <div className="coop-immersive-hud-head">
                      <span className="eyebrow">{t("沉浸进行中")}</span>
                      <b>{room.chapter_transition?.pending ? t("正在准备下一关") : immersiveStatus}</b>
                    </div>
                    <StoryProgress room={room} clip={playingClip} compact onOpenOutline={() => setPanel("outline")} />
                    {room.chapter_transition?.pending && <p>{room.chapter_transition.error || t("新场景与资产就绪后将自动继续。")}</p>}
                    {immersiveGuidance?.world_event && (
                      <div className="coop-immersive-event">
                        <span>{t("BOSS / 场景预告")}</span>
                        <strong>{immersiveGuidance.world_event}</strong>
                      </div>
                    )}
                    {immersiveGuidance?.objective && (
                      <div className="coop-immersive-objective">
                        <span>{t("当前目标")}</span><strong>{immersiveGuidance.objective}</strong>
                      </div>
                    )}
                    {!!immersiveGuidance?.suggestions?.length && (
                      <div className="coop-immersive-suggestions">
                        <span>{t("下一步动作建议")}</span>
                        <div>
                          {room.members.map((member, index) => {
                            const suggestion = immersiveGuidance.suggestions.find((item) => item.user_id === member.user_id)?.action;
                            if (!suggestion) return null;
                            return (
                              <p key={member.user_id}>
                                <b>{member.character || t("玩家 {p}", { p: letter(index) })}</b>
                                <span>{suggestion}</span>
                              </p>
                            );
                          })}
                        </div>
                      </div>
                    )}
                  </div>
                )}
                </section>
                {roundGuidance?.world_event && (
                  <div className="coop-world-event-banner" role="status">
                    <b>{t("BOSS / 环境动态预告")}</b>
                    <span>{roundGuidance.world_event}</span>
                  </div>
                )}
              </div>
              {room.generation_cancelling && <div className="coop-guidance-status" role="status">{t("正在取消下一段视频，已有预告和建议已保留…")}</div>}
              {currentVideo && room.intro?.status === "generating" && !busy && (
                <div className="coop-guidance-status" role="status">{t("正在根据历史局势和本轮行动准备下一步建议…")}</div>
              )}
              {currentVideo && room.intro?.status === "failed" && (
                <div className="coop-guidance-status">{room.intro.error}
                  {host && <button className="sm ghost" disabled={!!working || busy}
                    onClick={() => void act("intro", async () => apply(await api.retryIntro(room.code)))}>{t("重试建议")}</button>}
                </div>
              )}
              {currentVideo && room.intro?.status === "ready" &&
                room.recommendation_prefetch?.status &&
                !["failed", "superseded", "consuming"].includes(room.recommendation_prefetch.status) && (
                  <div className="coop-guidance-status" role="status">
                    {room.recommendation_prefetch.status === "planning"
                      ? t("正在提前编排系统推荐的联合 Prompt…")
                      : room.recommendation_prefetch.status === "prompt_ready"
                        ? t("推荐动作的联合 Prompt 已准备好")
                        : room.recommendation_prefetch.status === "rendering"
                          ? t("正在播放期间提前生成推荐动作视频…")
                          : t("推荐动作视频已提前准备好，原样采用建议即可立即命中")}
                  </div>
                )}
              <div className="coop-action-area">
                {room.chapter_transition?.pending && <section className="coop-guidance-status" role="status" aria-live="polite">
                  <h3>{room.chapter_transition.status === "failed" ? t("下一关准备未完成") : t("正在准备下一关")}</h3>
                  <p>{room.chapter_transition.error || (room.chapter_transition.status === "assets" ? t("正在准备场景与生物参考，完成后继续冒险。") : t("正在根据本关结果规划新的目标与场景。"))}</p>
                  <p>{t("已有视频与动作草稿会保留。")}</p>
                  {room.chapter_transition.status === "failed" && <button disabled={!!working || !host}
                    onClick={() => void act("intro", async () => apply(await api.retryIntro(room.code)))}>{t("重试准备下一关")}</button>}
                </section>}
                <section className="coop-action-editor" aria-label={t("角色行动输入")}>
                  <div className={`coop-composers ${local ? "is-local" : ""}`}>
                    {actionPlayers.map((p) => (
                      <ActionComposer
                        key={`${room.code}:${room.round}:${p.user_id}`}
                        player={p}
                        index={room.members.indexOf(p)}
                        local={local}
                        active={p.user_id === activeActionUserId}
                        tabs={
                          <div className="coop-action-tabs" role="group" aria-label={t("切换玩家")}>
                            {actionPlayers.map((seat) => {
                              const label = letter(room.members.indexOf(seat));
                              return (
                                <button
                                  type="button"
                                  aria-pressed={seat.user_id === activeActionUserId}
                                  aria-label={t("切换到玩家 {p}：{name}", { p: label, name: seat.character || seat.name })}
                                  title={`${t("玩家 {p}", { p: label })} · ${seat.character || seat.name}${seat.ready ? t(" · 已发送") : ""}`}
                                  className={`coop-action-tab ${seat.user_id === activeActionUserId ? "is-active" : ""}`}
                                  key={seat.user_id}
                                  onClick={() => setActiveActionUserId(seat.user_id)}
                                >
                                  {label}
                                </button>
                              );
                            })}
                          </div>
                        }
                        suggestion={roundGuidance?.suggestions.find(s => s.user_id === p.user_id)?.action}
                        onDirty={(dirty) =>
                          setDirtySeats((prev) =>
                            prev[p.user_id] === dirty
                              ? prev
                              : { ...prev, [p.user_id]: dirty },
                          )
                        }
                        busy={busy || !!room.chapter_transition?.pending}
                        working={!!working}
                        onCharacter={() => openCharacter(local ? p.user_id : "")}
                        onSubmit={(text, ready) =>
                          void act("action", async () =>
                            apply(
                              await api.submitAction(
                                room,
                                text,
                                ready,
                                local ? p.user_id : undefined,
                              ),
                            ),
                          )
                        }
                      />
                    ))}
                  </div>
                </section>
              </div>
            </div>
            <aside
              className={`coop-panel coop-party ${room.members.length === 4 ? "is-four" : ""}`}
              aria-label={t("队伍与联合动作")}
            >
              <div className="coop-party-heading">
                <h2>
                  {t("队伍")}{" "}
                  <span>
                    {room.members.length}/{room.capacity}
                  </span>
                </h2>
                <button className="ghost sm" onClick={() => setPanel("prompt")}>
                  {t("联合指令 ↗")}
                </button>
              </div>
              <div className="coop-party-scroll">
                <section className="coop-party-assets" aria-label={t("共享参考图")}>
                  <div className="coop-party-assets-head">
                    <strong>{t("背景设定")}</strong>
                    <span className="coop-muted">{t("场景 / BOSS")}</span>
                  </div>
                  <div className="coop-party-assets-grid">
                    {sharedAssetItems.map((asset) => (
                      <button
                        type="button"
                        className="coop-party-asset"
                        key={asset.kind}
                        onClick={() => setPanel(asset.kind === "scene" ? "scene-edit" : "monster")}
                        aria-label={t("{label}参考图", { label: asset.label })}
                      >
                        {asset.ref ? (
                          <img src={mediaUrl("refs", asset.ref)} alt={asset.name || asset.label} />
                        ) : (
                          <span className="coop-party-asset-empty">✦</span>
                        )}
                        <span>
                          <b>{asset.name || asset.label}</b>
                          <small>{asset.ref ? t("已采用") : t("待设置")}</small>
                        </span>
                      </button>
                    ))}
                  </div>
                </section>
                {room.members.map((p, i) => (
                  <article
                    key={p.user_id}
                    className={`coop-player ${p.ready ? "is-ready" : ""}`}
                  >
                    <div className="coop-player-head">
                      <button
                        className="coop-avatar"
                        aria-label={t("编辑形象 {name}", { name: p.character || p.name })}
                        disabled={
                          !local &&
                          p.user_id !== user.id &&
                          !(host && p.kind === "ai")
                        }
                        onClick={() =>
                          openCharacter(
                            local || p.kind === "ai" ? p.user_id : "",
                          )
                        }
                      >
                        {p.ref ? (
                          <img
                            src={mediaUrl("refs", p.ref)}
                            alt={p.character}
                          />
                        ) : (
                          <span>{letter(i)}</span>
                        )}
                      </button>
                      <div className="coop-player-info">
                        <strong>{p.character || t("等待设置角色")}</strong>
                        <small>
                          {p.kind === "ai"
                            ? `AI · ${(p.ai_source || room.ai_mode) === "offline" ? t("离线演练") : t("自主配合")}`
                            : local
                              ? t("玩家 {p} · 由你控制", { p: letter(i) })
                              : p.user_id === user.id
                                ? t("你")
                                : p.name}
                          {p.user_id === room.host_id ? t(" · 房主") : ""}
                        </small>
                      </div>
                      <span
                        className={`coop-player-status ${p.ready ? "ready" : ""}`}
                      >
                        {p.ready ? t("已发送") : t("待发送")}
                      </span>
                    </div>
                    <p className="coop-player-action" title={p.action}>
                      {p.action ||
                        (p.kind === "ai"
                          ? t("你提交动作后，队友会自动配合。")
                          : p.ref
                            ? t("等待写下本轮动作…")
                            : t("先生成或上传角色形象。"))}
                    </p>
                    {host && p.kind === "ai" && (
                      <div className="coop-player-tools">
                        <button
                          className="ghost sm"
                          aria-label={t("移除 AI 队友 {name}", { name: p.name })}
                          disabled={busy || !!working}
                          onClick={() =>
                            void act("ai-remove", async () =>
                              apply(await api.removeAI(room.code, p.user_id)),
                            )
                          }
                        >
                          {t("移除")}
                        </button>
                      </div>
                    )}
                  </article>
                ))}
                {host && !local && room.members.length < room.capacity && (
                  <div className="coop-ai-controls">
                    <label className="coop-sr-only" htmlFor="coop-ai-profile">
                      {t("队友角色")}
                    </label>
                    <select
                      id="coop-ai-profile"
                      value={aiProfile}
                      onChange={(e) => setAIProfile(e.target.value)}
                      disabled={busy || !!working}
                    >
                      <option value="axe">{t("阿岚 · 持斧战士")}</option>
                      <option value="shield">{t("小盾 · 守护者")}</option>
                      <option value="mage">{t("星火 · 法师")}</option>
                    </select>
                    <button
                      className="sm"
                      disabled={
                        busy ||
                        !!working ||
                        room.members.some((p) => p.ai_profile === aiProfile)
                      }
                      onClick={() =>
                        void act("ai-add", async () =>
                          apply(await api.addAI(room.code, aiProfile)),
                        )
                      }
                    >
                      {working === "ai-add" ? t("入场中…") : t("+ AI 队友")}
                    </button>
                  </div>
                )}
                {room.members.some((p) => p.kind === "ai") && (
                  <div className="coop-ai-status" role="status">
                    <span>
                      {room.ai_status === "thinking"
                        ? t("队友正在构思配合…")
                        : room.ai_status === "failed"
                          ? room.ai_error
                          : room.ai_status === "ready"
                            ? t("队友的配合动作已就绪")
                            : t("提交动作后，AI 队友自动跟上")}
                    </span>
                    {host &&
                      room.members
                        .filter((p) => p.kind !== "ai")
                        .every((p) => p.ready) && (
                        <button
                          className="ghost sm"
                          disabled={
                            busy || !!working || room.ai_status === "thinking"
                          }
                          onClick={() =>
                            void act("ai-respond", async () =>
                              apply(await api.respondAI(room.code)),
                            )
                          }
                        >
                          {room.ai_status === "failed" ? t("重试") : t("重新配合")}
                        </button>
                      )}
                  </div>
                )}
                {room.error && (
                  <div className="coop-error" role="alert">
                    {room.error}{t("。动作已保留。")}
                  </div>
                )}
              </div>
              <div className="coop-generate-footer">
                <section className="coop-fragment-tree" aria-label={t("片段树")}>
                  <div className="coop-fragment-tree-head">
                    <strong>{t("片段树")}</strong>
                    {fragmentNodes.length > 0 && <span>{t("{n} 段", { n: fragmentNodes.length })}</span>}
                  </div>
                  {fragmentNodes.length > 0 ? (
                    <div className="coop-fragment-tree-list">
                      {fragmentNodes.map((fragment, index) => {
                        const active = !room.preview_clip && (selectedClip || currentVideo?.clip) === fragment.clip;
                        return (
                          <div
                            key={fragment.key}
                            className={`coop-fragment-row ${active ? "is-current" : ""}`}
                          >
                            <span className="coop-fragment-rail" aria-hidden="true">
                              {index === fragmentNodes.length - 1 ? "└" : index === 0 ? "┌" : "│"}
                            </span>
                            <button
                              type="button"
                              className={`coop-fragment-node ${index === fragmentNodes.length - 1 ? "head" : ""} ${active ? "is-current" : ""}`}
                              onClick={() => setSelectedClip(fragment.clip)}
                              aria-current={active ? "step" : undefined}
                              title={fragment.label}
                            >
                              <span className="coop-fragment-number">{String(index + 1).padStart(2, "0")}</span>
                              {fragment.label}
                            </button>
                          </div>
                        );
                      })}
                    </div>
                  ) : (
                    <p className="coop-fragment-empty">{t("视频生成后会按顺序出现在这里")}</p>
                  )}
                </section>
                <span>{statusText}</span>
                {(room.status === "failed" || !room.auto_generate) && <button
                  className="primary"
                  disabled={
                    !host ||
                    busy ||
                    !!working ||
                    (!allReady && !room.opening_generation) ||
                    hasDraftChanges ||
                    !room.generation_enabled
                  }
                  onClick={() =>
                    void act("generate", async () =>
                      apply(await api.generate(room)),
                    )
                  }
                >
                  {busy ? t("正在共同生成…") : room.status === "failed" ? room.opening_generation ? t("重试序章视频") : t("重试本轮生成") : t("生成视频")}
                  <span>↗</span>
                </button>}
              </div>
            </aside>
          </main>
        )}
      </div>

      {room && panel && (
        <CoopDialog
          wide={panel === "log" || panel === "agent"}
          title={
            {
              character: local
                ? t("玩家 {p} 形象", { p: letter(room.members.findIndex((p) => p.user_id === selectedPlayer?.user_id)) })
                : portraitTarget
                  ? t("队友形象")
                  : t("我的角色"),
              scene: t("背景设定"),
              "scene-edit": t("场景设定"),
              monster: t("BOSS设定"),
              prompt: t("联合生成指令"),
              history: t("共同的冒险"),
              invite: t("邀请队友"),
              log: t("后台日志"),
              outline: t("当前世界大纲"),
              agent: t("Agent 轨迹"),
            }[panel]
          }
          onClose={() => setPanel("")}
        >
          {panel === "log" && <CoopLog code={room.code} initialRound={logRound} initialStage={logStage} />}
          {panel === "outline" && <StoryOutline room={room} />}
          {panel === "agent" && <CoopAgent room={room} busy={busy} />}
          {error && (
            <div className="coop-error" role="alert">
              {error}
            </div>
          )}
          {panel === "character" && (
            <div className="coop-character-editor">
              <div
                className={`coop-current-portrait ${selectedPlayer?.portrait && selectedPlayer.portrait.status !== "applied" ? "has-preview" : ""}`}
              >
                {selectedPlayer?.portrait &&
                selectedPlayer.portrait.status !== "applied" ? (
                  <PortraitPreview
                    portrait={selectedPlayer.portrait}
                    disabled={busy || !!working}
                    onUse={() =>
                      void act("portrait-use", async () => {
                        apply(
                          await api.usePortrait(
                            room.code,
                            selectedPlayer.portrait!.request_id,
                            portraitTarget || undefined,
                          ),
                        );
                        setPanel("");
                      })
                    }
                  />
                ) : (
                  <>
                    {selectedPlayer?.ref ? (
                      <img
                        src={mediaUrl("refs", selectedPlayer.ref)}
                        alt={selectedPlayer.character}
                      />
                    ) : (
                      <div className="coop-portrait-placeholder">
                        <DuoIcon />
                        <span>{t("你的角色，即将登场")}</span>
                      </div>
                    )}
                    <strong>
                      {selectedPlayer?.character || t("尚未绑定形象")}
                    </strong>
                    <p>
                      {selectedPlayer?.description ||
                        t("生成一张新形象，或上传你喜欢的角色。")}
                    </p>
                    <span className="tag">{t("当前使用的形象")}</span>
                  </>
                )}
              </div>
              <div className="coop-character-fields">
                {selectedPlayer?.kind === "ai" ? (
                  <>
                    <button
                      className="coop-random-portrait"
                      disabled={
                        busy ||
                        !!working ||
                        !room.image_generation_enabled ||
                        selectedPlayer.portrait?.status === "generating"
                      }
                      onClick={() => drawPortrait(true, portraitTarget)}
                    >
                      {t("随机生成队友形象")}
                    </button>
                    <p className="coop-muted">{t("生成后先预览，再决定是否替换。")}</p>
                    <button
                      className="ghost sm"
                      disabled={busy || !!working}
                      onClick={() =>
                        void act("ai-remove", async () => {
                          apply(await api.removeAI(room.code, portraitTarget));
                          setPanel("");
                        })
                      }
                    >
                      {t("移除这位 AI 队友")}
                    </button>
                  </>
                ) : (
                  <>
                    <div className="coop-portrait-start">
                      <button
                        type="button"
                        className="coop-random-portrait"
                        disabled={
                          busy ||
                          !!working ||
                          !room.image_generation_enabled ||
                          selectedPlayer?.portrait?.status === "generating"
                        }
                        onClick={() =>
                          drawPortrait(true, portraitTarget || undefined)
                        }
                      >
                        {t("🎲 随机角色并生图")}
                      </button>
                      <button
                        type="button"
                        className="sm"
                        disabled={
                          busy ||
                          !!working ||
                          !room.image_generation_enabled ||
                          selectedPlayer?.portrait?.status === "generating" ||
                          !name.trim() ||
                          !description.trim()
                        }
                        onClick={() =>
                          drawPortrait(false, portraitTarget || undefined)
                        }
                      >
                        {t("按角色描述生图")}
                      </button>
                      <small>
                        {room.image_generation_enabled
                          ? t("不用找参考图，先生成预览，再决定是否使用。")
                          : t("当前实例尚未连接本地生图服务。")}
                      </small>
                    </div>
                    {
                      <form onSubmit={(e) => void saveCharacter(e)}>
                        <fieldset disabled={busy || !!working}>
                          <label>
                            {t("角色名称")}
                            <input
                              value={name}
                              maxLength={60}
                              placeholder={t("例如：持剑的游侠")}
                              onChange={(e) => setName(e.target.value)}
                              required
                            />
                          </label>
                          <label>
                            {t("角色特点")}
                            <input
                              value={description}
                              maxLength={500}
                              placeholder={t("武器、外观、擅长的事情")}
                              onChange={(e) => setDescription(e.target.value)}
                            />
                          </label>
                          <label className="coop-upload">
                            <span>{t("＋ 上传角色参考图")}</span>
                            <small>
                              {file?.name || t("JPG / PNG / WebP · 最大 8 MB")}
                            </small>
                            <input
                              aria-label={t("角色参考图")}
                              type="file"
                              onClick={markFilePickerOpen}
                              accept="image/jpeg,image/png,image/webp"
                              onChange={(e) =>
                                setFile(e.target.files?.[0] || null)
                              }
                            />
                          </label>
                          <button
                            className="primary"
                            disabled={
                              !name.trim() || (!file && !selectedPlayer?.ref)
                            }
                          >
                            {working === "character"
                              ? t("绑定中…")
                              : local
                                ? t("保存这个角色")
                                : t("绑定我的角色")}
                          </button>
                        </fieldset>
                      </form>
                    }
                  </>
                )}
              </div>
            </div>
          )}
          {panel === "scene" && (
            <>
              <h3>{room.title}</h3>
              <p className="coop-setting-full">{room.setting}</p>
              <p className="coop-muted">{room.visual_style}</p>
              <div className="coop-room-actions">
                {host && (
                  <button className="ghost sm danger" disabled={busy || !!working}
                    onClick={() => askDelete(room)}>
                    {t("删除世界")}
                  </button>
                )}
                <button
                  className="ghost sm"
                  disabled={busy || !!working}
                  onClick={() =>
                    void act("leave", async () => {
                      await api.leaveRoom(room.code);
                      setPanel("");
                      lobby();
                    })
                  }
                >
                  {t("离开房间")}
                  {host &&
                  room.members.filter((p) => !p.kind || p.kind === "human")
                    .length > 1
                    ? t(" · 转交房主")
                    : ""}
                </button>
              </div>
            </>
          )}
          {panel === "scene-edit" && <WorldAssets kind="scene" room={room} host={host} apply={apply} />}
          {panel === "monster" && <WorldAssets kind="monster" room={room} host={host} apply={apply} />}
          {panel === "prompt" && (
            <>
              <p className="coop-muted">
                {t("每位角色的参考图与动作，一起进入同一次生成。")}
              </p>
              {room.members.map((p, i) => (
                <div className="coop-full-action" key={p.user_id}>
                  <span className="coop-letter">{letter(i)}</span>
                  <div>
                    <strong>{p.character || p.name}</strong>
                    <p>{p.action || t("等待提交动作…")}</p>
                  </div>
                </div>
              ))}
              <h3>{t("融合后的 Prompt")}</h3>
              <pre className="coop-prompt">
                {room.merged_prompt || t("提交动作后，联合指令会显示在这里。")}
              </pre>
            </>
          )}
          {panel === "history" && (
            <div className="coop-history">
              {room.opening_node && (
                <details open>
                  <summary>{t("序章 · 送入视频模型的完整记录")}</summary>
                  {room.opening_node.clip && (
                    <video src={mediaUrl("clips", room.opening_node.clip)} controls preload="none" />
                  )}
                  <p>{room.opening_node.reply}</p>
                  <details className="coop-history-prompt">
                    <summary>{t("查看序章完整视频 Prompt")}</summary>
                    <pre className="coop-prompt">{room.opening_node.prompt || t("旧记录没有保存完整 Prompt。")}</pre>
                  </details>
                  <button className="ghost sm" onClick={() => openLogs(0)}>{t("查看序章完整日志 ↗")}</button>
                </details>
              )}
              {room.rounds.length ? (
                [...room.rounds].sort((a, b) => a.round - b.round).map((r) => (
                  <details key={r.round}>
                    <summary>
                      {t("第 {n} 轮", { n: r.round })} ·{" "}
                      {r.inputs.map((p) => p.character).join(" × ")}
                    </summary>
                    {r.node.clip && (
                      <video
                        src={mediaUrl("clips", r.node.clip)}
                        controls
                        preload="none"
                      />
                    )}
                    <p>{r.node.reply}</p>
                    <details className="coop-history-prompt">
                      <summary>{t("查看本轮送入视频模型的完整 Prompt")}</summary>
                      <pre className="coop-prompt">{r.node.prompt || t("旧记录没有保存完整 Prompt。")}</pre>
                    </details>
                    <button className="ghost sm" onClick={() => openLogs(r.round)}>{t("查看本轮完整日志 ↗")}</button>
                    {r.inputs.map((p) => (
                      <p key={p.user_id}>
                        <b>{p.character}：</b>
                        {p.action}
                      </p>
                    ))}
                  </details>
                ))
              ) : !room.opening_node ? (
                <div className="coop-empty-history">
                  <DuoIcon />
                  <h3>{t("故事还未开始")}</h3>
                  <p>{t("每一轮共同生成的镜头，会保存在这里。")}</p>
                </div>
              ) : null}
            </div>
          )}
          {panel === "invite" && (
            <div className="coop-invite">
              <DuoIcon />
              <p>{t("让朋友打开链接，带上自己的角色加入。")}</p>
              <label>
                {t("房间邀请码")}<strong>{room.code}</strong>
              </label>
              <button className="primary" onClick={() => void copyInvite()}>
                {copied ? t("已复制链接 ✓") : t("复制邀请链接")}
              </button>
              <p className="coop-muted">
                {t("{n} / {cap} 人已加入", { n: room.members.length, cap: room.capacity })}
              </p>
            </div>
          )}
        </CoopDialog>
      )}
      <Dialog open={!!undoNode} title={t("撤回最新一段？")}
        busy={working === "undo"} okLabel={t("确认撤回")} okClassName="danger" note={error}
        onClose={() => { if (working !== "undo") setUndoNode(null); }}
        onOk={() => { if (room && undoNode) void act("undo", async () => {
          const next = await api.rollback(room.code, undoNode);
          silenceNarration(); setSelectedClip(null); setPlayingClip(null);
          setDirtySeats({}); apply(next); setUndoNode(null);
        }); }}>
        <p>{t("撤回共同世界最新的视频，回到生成前，并把两人的动作放回输入栏。所有队友都会同步退回。")}</p>
        <p>{t("原视频文件保留，重新生成后会形成新的分支。")}</p>
      </Dialog>
      <Dialog
        open={!!deleting}
        title={t("删除世界：{title}", { title: deleting?.title || "" })}
        busy={working === "delete"}
        okLabel={t("确认删除")}
        okClassName="danger"
        note={deleteNote}
        onClose={() => { if (working !== "delete") setDeleting(null); }}
        onOk={() => void deleteWorld()}
      >
        <p>{t("将永久删除这个世界、全部回合记录，以及仅供它使用的参考图和视频，无法恢复。")}</p>
        {deleting?.mode === "online" && <p>{t("房间中的所有队友都会退出，这个邀请码也将失效。")}</p>}
      </Dialog>
    </div>
  );
}


/** Where the round is in the agent loop: perceive, plan, check, act, commit. */
function AgentPhases({ steps }: { steps: TraceStep[] }) {
  const phases: TracePhase[] = PHASES.filter((phase) => phase !== "reflect");
  const current = [...steps].reverse().find((step) => step.status === "running") ?? steps[steps.length - 1];
  const at = phases.indexOf(current.phase);
  return (
    <div className="coop-agent-phases" aria-label={t("Agent 进度")}>
      {phases.map((phase, i) => (
        <span key={phase} className={i < at ? "done" : i === at ? "now" : ""}>{PHASE_LABEL[phase]}</span>
      ))}
    </div>
  );
}
