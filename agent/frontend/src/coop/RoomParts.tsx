import { t } from '../i18n'
import { useEffect, useRef, useState, type ReactNode } from "react";
import { mediaUrl } from "../api";
import type { Player } from "./api";
import { isPickingFile } from "../filepicker";

export const letter = (index: number) => String.fromCharCode(65 + index);

export function PortraitPreview({
  portrait,
  disabled,
  onUse,
}: {
  portrait?: Player["portrait"];
  disabled: boolean;
  onUse: () => void;
}) {
  if (!portrait || portrait.status === "applied") return null;
  return (
    <div className="coop-portrait-preview" aria-live="polite">
      {portrait.status === "generating" ? (
        <div className="coop-portrait-loading">
          <div className="coop-spinner" />
          <span>{t("正在生成角色形象…")}</span>
          <small>{t("完成后先预览，由你决定是否使用。")}</small>
        </div>
      ) : portrait.status === "failed" ? (
        <p role="alert">{portrait.error}</p>
      ) : (
        <>
          <img
            src={mediaUrl("refs", portrait.ref)}
            alt={`${portrait.name} · ${t("生图预览")}`}
          />
          <strong>{portrait.name}</strong>
          <p>{portrait.description}</p>
          <small>{t("本地生图 · {n} 秒 · 尚未替换当前角色", { n: portrait.elapsed ?? 0 })}</small>
          <button
            type="button"
            className="primary"
            disabled={disabled}
            onClick={onUse}
          >
            {t("使用这张形象")}
          </button>
        </>
      )}
    </div>
  );
}

export function CoopDialog({
  title,
  onClose,
  children,
  wide = false,
}: {
  title: string;
  onClose: () => void;
  children: ReactNode;
  wide?: boolean;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const dialog = ref.current;
    dialog?.showModal();
    return () => dialog?.close();
  }, []);
  return (
    <dialog
      ref={ref}
      className={`coop-dialog ${wide ? "coop-dialog-wide" : ""}`}
      aria-label={title}
      onCancel={(e) => {
        e.preventDefault();
        if (!isPickingFile()) onClose();
      }}
    >
      <div className="coop-dialog-head">
        <h2>{title}</h2>
        <button className="icon ghost" aria-label={t("关闭弹窗")} onClick={onClose}>
          ×
        </button>
      </div>
      <div className="coop-dialog-body">{children}</div>
    </dialog>
  );
}

export function ActionComposer({
  player,
  index,
  local,
  active = true,
  tabs,
  busy,
  working,
  onCharacter,
  onSubmit,
  onDirty,
  suggestion,
}: {
  player: Player;
  index: number;
  local: boolean;
  active?: boolean;
  tabs: ReactNode;
  busy: boolean;
  working: boolean;
  onCharacter: () => void;
  onSubmit: (text: string, ready: boolean) => void;
  onDirty: (dirty: boolean) => void;
  suggestion?: string;
}) {
  const [draft, setDraft] = useState(player.action || "");
  const dirty = draft.trim() !== player.action.trim();
  useEffect(() => {
    onDirty(dirty);
  // eslint-disable-next-line react-hooks/exhaustive-deps -- onDirty is a new function every parent render; report only changes
  }, [dirty]);
  return (
    <section
      className={`coop-panel coop-composer ${active ? "is-active" : "is-hidden"}`}
      aria-label={t("玩家 {p} 动作区", { p: letter(index) })}
      aria-hidden={!active}
    >
      <div className="coop-composer-heading">
        <div className="coop-composer-identity">
          {tabs}
          <strong title={player.character || player.name}>{player.character || t("玩家 {p}", { p: letter(index) })}</strong>
        </div>
        <div className="coop-composer-guidance">
          {suggestion && <button type="button" className="coop-action-suggestion" title={suggestion}
            aria-label={t("采用建议：{s}", { s: suggestion })}
            disabled={busy || player.ready || working} onClick={() => setDraft(suggestion)}>
            <span>{t("下一步建议")}</span><b>{suggestion}</b><span>{t("采用 ↙")}</span>
          </button>}
        </div>
        <div className="coop-composer-tools">
          {player.ready && (
            <button
              type="button"
              className="ghost sm"
              aria-label={t("撤回玩家 {p} 的动作", { p: letter(index) })}
              disabled={busy || working}
              onClick={() => onSubmit(player.action, false)}
            >
              {t("撤回")}
            </button>
          )}
          <button type="button" className="ghost sm" onClick={onCharacter}>
            {t("角色形象")}
          </button>
        </div>
      </div>
      <div className="coop-composer-input">
        <textarea
          aria-label={local ? t("玩家 {p} 本轮动作", { p: letter(index) }) : t("本轮动作")}
          rows={2}
          maxLength={1200}
          placeholder={player.ref ? t("写下动作与目标…") : t("先设置角色形象")}
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          disabled={busy || player.ready || !player.ref}
        />
        <button
          type="button"
          className="primary coop-send-action"
          aria-label={player.ready ? t("玩家 {p} 已发送", { p: letter(index) }) : t("发送玩家 {p} 的动作", { p: letter(index) })}
          title={player.ready ? t("已发送") : t("发送动作")}
          disabled={busy || working || player.ready || !player.ref || !draft.trim()}
          onClick={() => onSubmit(draft, true)}
        >
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <path d={player.ready ? "m5 12 4 4L19 6" : "M12 19V5m-6 6 6-6 6 6"} />
          </svg>
        </button>
      </div>
    </section>
  );
}
