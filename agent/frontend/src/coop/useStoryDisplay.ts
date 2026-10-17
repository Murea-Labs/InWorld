import { language, t } from '../i18n'
import { useEffect, useState } from "react";
import { getStoryDisplay, type Room } from "./api";

// Older saves hold English outline prose; only Chinese rooms translate it for display.
const needsChinese = (value: string) => language() === "zh" && /[a-zA-Z]/.test(value) && !/[\u3400-\u9fff]/.test(value);
const requests = new Map<string, ReturnType<typeof getStoryDisplay>>();

export function useStoryDisplay(room: Room) {
  const outline = room.story_outline;
  const progress = room.story_progress;
  const plan = room.intro?.short_term_plan;
  const values = [outline?.overall_goal, progress?.summary, progress?.next_gap, plan?.change_reason];
  for (const stage of outline?.stages ?? []) {
    const state = progress?.stages?.[stage.id];
    values.push(stage.title, stage.goal, stage.completion_evidence, stage.alternative,
      state?.evidence, state?.remaining, state?.reason);
  }
  for (const step of plan?.steps ?? []) {
    values.push(step.goal, step.prerequisite, step.cooperation, step.completion_evidence, step.if_blocked);
  }
  const foreign = [...new Set(values.filter((v): v is string => !!v && needsChinese(v)))];
  const key = foreign.length ? JSON.stringify([room.code, foreign]) : "";
  const [retry, setRetry] = useState(0);
  const [display, setDisplay] = useState<{ key: string; translations: Record<string, string>; failed: boolean }>(
    { key: "", translations: {}, failed: false });
  useEffect(() => {
    if (!key) return;
    let active = true;
    setDisplay({ key, translations: {}, failed: false });
    let request = requests.get(key);
    if (!request) {
      request = getStoryDisplay(room.code);
      requests.set(key, request);
      if (requests.size > 32) requests.delete(requests.keys().next().value!);
    }
    request.then(({ translations }) => {
      const requested = (JSON.parse(key) as [string, string[]])[1];
      if (requested.some(value => !translations[value] || needsChinese(translations[value]))) {
        throw new Error(t("中文说明尚未完整返回"));
      }
      if (active) setDisplay({ key, translations, failed: false });
    }).catch(() => {
      requests.delete(key);
      if (active) setDisplay({ key, translations: {}, failed: true });
    });
    return () => { active = false; };
  }, [key, room.code, retry]);
  const translations = display.key === key ? display.translations : {};
  const pending = foreign.some(value => !translations[value] || needsChinese(translations[value]));
  const failed = display.key === key && display.failed;
  return {
    text(value: string | undefined) {
      if (!value || !needsChinese(value)) return value;
      const translated = translations[value];
      return translated && !needsChinese(translated) ? translated : failed ? t("中文说明暂未就绪") : t("正在整理中文…");
    },
    pending, failed,
    retry() { requests.delete(key); setRetry(value => value + 1); },
  };
}
