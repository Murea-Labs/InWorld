import { t } from '../i18n'
import { useEffect, useState } from "react";
import type { User } from "../types";
import * as api from "./api";
import type { Room, RoomSummary } from "./api";

export function DuoIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <circle cx="8" cy="7" r="3" />
      <circle cx="17" cy="8" r="2.5" />
      <path d="M2 20v-3a6 6 0 0 1 12 0v3M15 13a5 5 0 0 1 7 4v3" />
    </svg>
  );
}

const PRESETS = [
  {
    title: t("巨龙的宝库"),
    setting:
      t("古老山洞里，巨龙守着堆积如山的金币。你和队友闯入宝库，需要配合牵制巨龙，找到通向山外的出口。"),
  },
  {
    title: t("雾港迷踪"),
    setting:
      t("雾气笼罩的港口，有一艘不在航线上出现的船。你和队友分工调查码头、追踪线索，一起揭开船上的秘密。"),
  },
  {
    title: t("星际救援"),
    setting:
      t("空间站即将失去动力。你和队友必须互相配合，修复核心、保护船员，并赶在最后一班逃生舱离开前脱身。"),
  },
];

/** The co-op lobby: start a local or online room, join with a code, reopen a world. */
export default function CoopLobby({
  user, code, invited, rooms, working, act, go, askDelete,
}: {
  user: User;
  code: string;
  invited: boolean;
  rooms: RoomSummary[];
  working: string;
  act: (label: string, fn: () => Promise<void>) => Promise<void>;
  go: (next: Room) => void;
  askDelete: (target: RoomSummary) => void;
}) {
  const [invite, setInvite] = useState(code);
  const [title, setTitle] = useState(PRESETS[0].title);
  const [setting, setSetting] = useState(PRESETS[0].setting);
  const [style, setStyle] = useState(
    t("电影感奇幻冒险，细腻光影，统一的角色与场景风格"),
  );
  const [capacity, setCapacity] = useState(4);
  const [mode, setMode] = useState<"local" | "online">("local");
  useEffect(() => setInvite(code), [code]);

  return (
    <main className="coop-lobby">
      <div className="coop-lobby-title">
        <div>
          <span className="eyebrow">CO-OP WORLD</span>
          <h1>{t("一起进入下一个世界")}</h1>
          <p>
            {t("本地一人控制双角色，或邀请朋友联机。所有动作融入同一段视频。")}
          </p>
        </div>
      </div>
      {invited && (
        <section className="coop-invitation coop-panel">
          <div>
            <b>{t("队友邀请你一起出发")}</b>
            <p>{t("邀请码 {code}", { code })}</p>
          </div>
          <button
            className="primary"
            disabled={!!working}
            onClick={() =>
              void act("join", async () => go(await api.joinRoom(code)))
            }
          >
            {t("加入这个房间 →")}
          </button>
        </section>
      )}
      <div className="coop-mode-picker" aria-label={t("选择游玩模式")}>
        <button
          className={mode === "local" ? "on" : ""}
          aria-pressed={mode === "local"}
          onClick={() => setMode("local")}
        >
          <strong>{t("本地双人调试")}</strong>
          <span>{t("一个人控制 A、B · 无需切换账号")}</span>
        </button>
        <button
          className={mode === "online" ? "on" : ""}
          aria-pressed={mode === "online"}
          onClick={() => setMode("online")}
        >
          <strong>{t("联机房间")}</strong>
          <span>{t("2–4 位玩家 · 各自控制角色 · 共享镜头")}</span>
        </button>
      </div>
      <div className="coop-lobby-grid">
        <section className="coop-panel coop-create">
          <div className="coop-section-heading">
            <h2>{mode === "local" ? t("开始本地双人") : t("创建联机房间")}</h2>
            <span>
              {mode === "local"
                ? t("A、B 两个角色都由你控制")
                : t("邀请朋友，或与 AI 搭档")}
            </span>
          </div>
          <form
            onSubmit={(e) => {
              e.preventDefault();
              void act("create", async () =>
                go(
                  await api.createRoom({
                    title,
                    setting,
                    visual_style: style,
                    capacity: mode === "local" ? 2 : capacity,
                    mode,
                  }),
                ),
              );
            }}
          >
            <div className="coop-presets">
              {PRESETS.map((p) => (
                <button
                  type="button"
                  key={p.title}
                  className={title === p.title ? "on" : ""}
                  onClick={() => {
                    setTitle(p.title);
                    setSetting(p.setting);
                  }}
                >
                  {p.title}
                </button>
              ))}
            </div>
            <label>
              {t("房间名称")}
              <input
                value={title}
                maxLength={80}
                onChange={(e) => setTitle(e.target.value)}
                required
              />
            </label>
            <label>
              {t("你们的故事从哪里开始？")}
              <textarea
                rows={3}
                value={setting}
                maxLength={2000}
                onChange={(e) => setSetting(e.target.value)}
                required
              />
            </label>
            <div className="coop-form-row">
              <label>
                {t("画面风格")}
                <input
                  value={style}
                  maxLength={500}
                  onChange={(e) => setStyle(e.target.value)}
                />
              </label>
              <label>
                {mode === "local" ? t("本地角色") : t("队伍人数")}
                <select
                  disabled={mode === "local"}
                  value={mode === "local" ? 2 : capacity}
                  onChange={(e) => setCapacity(Number(e.target.value))}
                >
                  <option value={2}>{t("2 人")}</option>
                  <option value={3}>{t("3 人")}</option>
                  <option value={4}>{t("4 人")}</option>
                </select>
              </label>
            </div>
            <button
              className="primary coop-create-button"
              disabled={!!working || !title.trim() || !setting.trim()}
            >
              <DuoIcon />
              {working === "create"
                ? t("正在准备房间…")
                : mode === "local"
                  ? t("开始本地双人")
                  : t("创建联机房间")}
              <span>↗</span>
            </button>
          </form>
        </section>
        <div className="coop-lobby-side">
          <section className="coop-panel coop-join">
            <div className="coop-section-heading">
              <h2>{t("加入队伍")}</h2>
            </div>
            <form
              onSubmit={(e) => {
                e.preventDefault();
                void act("join", async () =>
                  go(await api.joinRoom(invite.trim().toUpperCase())),
                );
              }}
            >
              <label>
                {t("房间邀请码")}
                <input
                  className="coop-code-input"
                  value={invite}
                  maxLength={12}
                  placeholder={t("例如 A1B2C3D4E5F6")}
                  onChange={(e) =>
                    setInvite(e.target.value.toUpperCase())
                  }
                  autoComplete="off"
                  required
                />
              </label>
              <button
                className="coop-join-button"
                disabled={!!working || invite.trim().length !== 12}
              >
                {working === "join" ? t("加入中…") : t("加入队伍 →")}
              </button>
            </form>
          </section>
          <section className="coop-panel coop-recent">
            <div className="coop-section-heading">
              <h2>{t("继续冒险")}</h2>
              <span>{t("{n} 个世界", { n: rooms.length })}</span>
            </div>
            <div className="coop-recent-list">
              {rooms.length ? (
                rooms.map((r) => (
                  <div className="coop-recent-row" key={r.code}>
                  <button
                    className="coop-recent-open"
                    onClick={() => {
                      window.location.hash = `coop/${r.code}`;
                    }}
                  >
                    <DuoIcon />
                    <span>
                      <b>{r.title}</b>
                      <small>
                        {r.mode === "local"
                          ? t("本地双人")
                          : t("{n} 人联机", { n: r.members.length })}{" "}
                        · {t("第 {n} 轮", { n: r.round })} ·{" "}
                        {r.status === "generating"
                          ? t("正在生成")
                          : t("等待行动")}
                      </small>
                    </span>
                    <span>→</span>
                  </button>
                  {r.host_id === user.id && (
                    <button
                      className="ghost sm danger coop-delete"
                      aria-label={t("删除世界：{title}", { title: r.title })}
                      disabled={!!working || r.status === "generating"}
                      onClick={() => askDelete(r)}
                    >{t("删除")}</button>
                  )}
                  </div>
                ))
              ) : (
                <p className="coop-muted">
                  {t("创建或加入房间后，从这里继续。")}
                </p>
              )}
            </div>
          </section>
        </div>
      </div>
    </main>
  );
}
