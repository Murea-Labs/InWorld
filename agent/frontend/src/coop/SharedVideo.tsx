import { t } from '../i18n'
import { useEffect, useRef, useState } from "react";

interface Props {
  src: string;
  /** Chronological clips, including the prologue when one exists. */
  playlist?: string[];
  /** Stop waiting narration before this video's own audio starts. */
  onPlay?: () => void;
  immersive?: boolean;
  onClipChange?: (src: string) => void;
}

export default function SharedVideo({ src, playlist = [], onPlay, immersive = false, onClipChange }: Props) {
  const video = useRef<HTMLVideoElement>(null);
  const [muted, setMuted] = useState(false);
  const [blocked, setBlocked] = useState(false);
  const [continuous, setContinuous] = useState(false);
  const [index, setIndex] = useState(() => Math.max(0, playlist.indexOf(src)));
  const previousSrc = useRef(src);
  const [immersiveSrc, setImmersiveSrc] = useState(src);
  const endedWaiting = useRef(false);
  const wasImmersive = useRef(false);
  const activeSrc = immersive ? immersiveSrc : continuous ? (playlist[index] || src) : src;

  useEffect(() => {
    if (immersive && !wasImmersive.current) {
      setImmersiveSrc(src);
      endedWaiting.current = !!video.current?.ended;
      setContinuous(false);
    }
    wasImmersive.current = immersive;
    if (immersive && endedWaiting.current && src !== immersiveSrc) {
      const next = playlist[playlist.indexOf(immersiveSrc) + 1];
      setImmersiveSrc(next || src);
      endedWaiting.current = false;
    }
    if (!immersive) setImmersiveSrc(src);
  }, [immersive, src, playlist, immersiveSrc]);

  useEffect(() => { onClipChange?.(activeSrc); }, [activeSrc, onClipChange]);

  useEffect(() => {
    // Clicking the clip tree changes src. Whether or not we were playing the whole
    // chain, leave the internal queue and play the clip the user picked.
    if (previousSrc.current !== src) {
      previousSrc.current = src;
      setContinuous(false);
      const current = playlist.indexOf(src);
      setIndex(current >= 0 ? current : 0);
      setBlocked(false);
      return;
    }
    if (!continuous) {
      const current = playlist.indexOf(src);
      setIndex(current >= 0 ? current : 0);
    }
  }, [continuous, playlist, src]);

  useEffect(() => {
    let active = true;
    const el = video.current;
    if (!el) return;
    const play = async () => {
      try { await el.play(); }
      catch {
        if (!active) return;
        el.muted = true;
        setMuted(true);
        try { await el.play(); } catch { if (active) setBlocked(true); }
      }
    };
    void play();
    return () => { active = false; };
  }, [activeSrc]);

  const playAll = () => {
    if (playlist.length < 2) return;
    setBlocked(false);
    setContinuous(true);
    setIndex(0);
  };

  const ended = () => {
    if (immersive) {
      const position = playlist.indexOf(immersiveSrc);
      const next = position >= 0 ? playlist[position + 1] : undefined;
      if (next) {
        setImmersiveSrc(next);
        endedWaiting.current = false;
      } else {
        endedWaiting.current = true;
      }
      return;
    }
    if (continuous && index < playlist.length - 1) {
      setIndex((value) => value + 1);
      return;
    }
    setContinuous(false);
  };

  const enable = () => {
    const el = video.current;
    if (!el) return;
    el.muted = false;
    setMuted(false);
    void el.play().then(() => setBlocked(false)).catch(() => setBlocked(true));
  };
  return <div className="coop-shared-video">
    {!immersive && playlist.length > 1 && <div className="coop-video-toolbar">
      <button className="sm coop-play-all" onClick={playAll} disabled={continuous}>
        {continuous ? t("正在连播 · {i}/{n}", { i: index + 1, n: playlist.length }) : t("连播全片")}
      </button>
    </div>}
    <div className="coop-shared-video-frame">
      <video ref={video} src={activeSrc} controls autoPlay playsInline preload="auto"
        onPlay={onPlay}
        onEnded={ended}
        onVolumeChange={() => setMuted(!!video.current?.muted)} />
      {(muted || blocked) && <button className="sm coop-play-sound" onClick={enable}>
        {blocked ? t("点击播放共同视频 ▶") : t("已自动播放 · 开启声音 ♫")}
      </button>}
    </div>
  </div>;
}
