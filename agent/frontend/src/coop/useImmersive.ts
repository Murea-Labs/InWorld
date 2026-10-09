import { t } from '../i18n'
import { useCallback, useEffect, useRef, useState } from "react";
import * as api from "./api";
import type { Room } from "./api";

export default function useImmersive(code: string, room: Room | null, playingClip: string | null,
  apply: (room: Room) => void, onError: (message: string) => void) {
  const [immersive, setImmersive] = useState(false);
  const [ready, setReady] = useState(false);
  const [exiting, setExiting] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const session = useRef<{ id: string; code: string; started: Promise<Room> } | null>(null);
  const attempted = useRef("");
  const mounted = useRef(true);
  const activeCode = useRef(code);
  activeCode.current = code;

  const stop = useCallback(async () => {
    const current = session.current;
    session.current = null; // Reject late start/advance callbacks immediately.
    setImmersive(false);
    setReady(false);
    if (!current) return;
    setExiting(true);
    try {
      // Even a failed response may have reached the server. Always send stop,
      // ordered after start so a fast click on Exit cannot leave a live session.
      await current.started.catch(() => undefined);
      const next = await api.stopImmersive(current.code, current.id);
      if (mounted.current && activeCode.current === current.code) apply(next);
    } catch (error) {
      if (mounted.current) onError(t("取消沉浸生成失败：{msg}", { msg: (error as Error).message }));
    } finally {
      if (mounted.current) setExiting(false);
    }
  }, [apply, onError]);

  const start = useCallback(() => {
    if (!code || session.current || exiting) return;
    const id = crypto.randomUUID();
    const current = { id, code, started: api.startImmersive(code, id) };
    session.current = current;
    attempted.current = "";
    setImmersive(true);
    setReady(false);
    void current.started.then((next) => {
      if (session.current !== current) return;
      apply(next);
      setReady(true);
    }).catch((error: Error) => {
      if (session.current !== current) return;
      onError(t("开启沉浸模式失败：{msg}", { msg: error.message }));
      void stop();
    });
  }, [code, exiting, apply, onError, stop]);

  useEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; void stop(); };
  }, [code, stop]);

  useEffect(() => {
    if (!immersive) return;
    const onKey = (event: KeyboardEvent) => { if (event.key === "Escape") void stop(); };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [immersive, stop]);

  useEffect(() => {
    const current = session.current;
    const latest = room?.rounds.at(-1)?.node || room?.opening_node;
    // At most one future round: begin planning while the current head plays,
    // then wait for the player to reach the new head before advancing again.
    if (!current || !immersive || !ready || submitting || !room || room.chapter_transition?.pending
        || room.code !== current.code || room.status !== "waiting"
        || room.intro?.status !== "ready" || !latest?.clip
        || playingClip !== latest.clip) return;
    const key = `${current.id}:${room.round}`;
    if (attempted.current === key) return;
    attempted.current = key;
    setSubmitting(true);
    void api.advanceImmersive(current.code, current.id, room.round).then((next) => {
      if (session.current === current) apply(next);
    }).catch((error: Error) => {
      if (session.current === current) onError(t("沉浸模式自动推进失败：{msg}", { msg: error.message }));
    }).finally(() => { if (mounted.current) setSubmitting(false); });
    // Room SSE revisions must not cancel this already accepted submission.
  }, [immersive, ready, submitting, room, playingClip, apply, onError]);

  return { immersive, exiting, start, stop };
}
