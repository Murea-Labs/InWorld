"""Assembling the six sections into the final caption and fitting it into the video model's length limit."""

from __future__ import annotations

import re
from murea_agent import config
from murea_agent.providers import llm
from murea_agent import reqlog
from murea_agent.util import _pretty
from murea_agent.director.agent_schema import (
    SECTION_ORDER,
)


# Hard limit on the video prompt (code points, `video.max_prompt_chars`); co-op compacts its captions to it.
PROMPT_MAX_CHARS = config.VIDEO_MAX_PROMPT_CHARS

# The largest share of the shot prose the compressor is asked to remove; deeper cuts are trimmed.
_MAX_MODEL_CUT = 0.3

_CJK = re.compile(r"[　-〿぀-ヿ㐀-䶿一-鿿가-힯＀-￯]")


def prompt_size(text: str) -> int:
    """The budget cost of a caption, in English characters.

    Budgets stand for text-encoder tokens. On Murea-Fast's Gemma tokenizer English prose takes
    4.3-5 characters per token, but a Chinese character (a dialogue line) is nearly a token on its
    own, so it costs four characters here.
    """
    return len(text) + 3 * len(_CJK.findall(text))


def assemble(fields: dict[str, str]) -> str:
    """Join the six prompt sections into the string the video model receives."""
    if not (fields.get("subject_definitions") or "").strip() and not (fields.get("summary") or "").strip():
        # A continuation without reference images has only the action prose; the images the other
        # sections point to are not sent this time. **But the two audio sections must stay**: the
        # video model generates video and audio together, and `overall_soundscape` tells it what the
        # audio should be. With only prose it gets no audio direction and invents voices: in one
        # world the first beat had no lines and no reference voice, yet the person on screen started
        # talking.
        body = (fields.get("detailed_description") or "").strip()
        # `N/A` is a placeholder required by the six-section format; pasted into a single paragraph
        # it is only noise.
        audio = [
            v
            for v in ((fields.get(k) or "").strip() for k in ("overall_soundscape", "non_diegetic_music"))
            if v and v.casefold() != "n/a"
        ]
        return "\n\n".join([body, *audio])
    return "\n\n".join((fields.get(k) or "N/A").strip() for k in SECTION_ORDER)


_SENTENCE = re.compile(r"[^.!?]+[.!?]+(?:\s+|$)")


def _trim_to_budget(fields: dict[str, str], limit: int) -> dict[str, str] | None:
    """Last resort after the compressor: keep the summary's first sentence, then
    drop trailing sentences of the shot until the caption fits. Never drops a
    <d> block, the [Shot 1] marker, the opening sentence, or the last remaining
    mention of a reference marker; returns None if that is not enough."""
    out = dict(fields)
    summary = out.get("summary", "")
    first = _SENTENCE.match(summary)
    if first:
        out["summary"] = first.group(0).strip()
    text = out.get("detailed_description", "")
    sentences = _split_sentences(text)
    marker = re.compile(r"<(?:Subject|Picture|Audio)\s+\d+>|\(S\d+\)")
    fixed = re.compile(r"<d>|</d>|\[Shot 1\]")

    def droppable(index: int) -> bool:
        sentence = sentences[index]
        if fixed.search(sentence):
            return False
        rest = out["summary"] + "".join(s for i, s in enumerate(sentences) if i != index)
        return all(m in rest for m in marker.findall(sentence))

    while prompt_size(assemble(out)) > limit:
        for index in range(len(sentences) - 1, 0, -1):
            if droppable(index):
                del sentences[index]
                break
        else:
            return None
        out["detailed_description"] = " ".join(sentences).strip()
    return out


def _split_sentences(text: str) -> list[str]:
    """Split a caption into sentences that join back to the same text.

    Dialogue blocks are masked first: a line such as `<d>[English] Stop!</d>, she calls.` has
    punctuation right before the closing tag, which a plain sentence regex misses, and the whole
    block must stay inside one sentence anyway.
    """
    blocks: list[str] = []

    def mask(m: re.Match) -> str:
        blocks.append(m.group(0))
        return f"\x00{len(blocks) - 1}\x00"

    masked = _D_BLOCK.sub(mask, text.strip())
    parts = [p for p in re.split(r"(?<=[.!?])\s+", masked) if p]
    return [re.sub(r"\x00(\d+)\x00", lambda m: blocks[int(m.group(1))], p) for p in parts]


def _fit_video_prompt(
    fields: dict[str, str], limit: int | None = None, *, compress: bool = True
) -> dict[str, str]:
    """Budget the final caption, including all server-injected constraints.

    First factor repeated reference-isolation rules without changing any action,
    identity or audio. Rare remaining overflows get one caption-only edit, never
    another world/quest/action plan or a silent slice through dialogue.

    `limit` is the video model's budget (`VideoProvider.max_prompt_chars`). `compress=False` skips
    the model edit and only trims: for prompts that are rarely sent (a continuation's fallback),
    a model call on every beat costs more than the rare case is worth.
    """
    limit = config.VIDEO_MAX_PROMPT_CHARS if limit is None else limit
    before = prompt_size(assemble(fields))
    if before <= limit:
        return fields
    out = dict(fields)
    pattern = re.compile(
        r"; only the entity's identity and appearance are retained\. The reference "
        r"backdrop, floor, shadows, lighting and framing are replaced by the current "
        r"story scene(?P<scene> \(<Subject \d+>\)| established in the shot) "
        r"throughout the shot\. The reference image itself is never displayed as a "
        r"frame or transition target\."
    )
    retention = out.get("retention_analysis", "")
    matches = list(pattern.finditer(retention))
    if len(matches) > 1 and len({m.group("scene") for m in matches}) == 1:
        shared = matches[0].group(0)[2:]
        retention = pattern.sub(".", retention)
        out["retention_analysis"] = retention + "\nFor every foreground reference, " + shared
    after = prompt_size(assemble(out))
    reqlog.record(
        "prompt_budget", before=before, after=after, limit=limit, method="deduplicate_reference_rules"
    )
    if after <= limit:
        return out

    # Keep the identity, retention and sound sections byte-for-byte. Only the
    # shot prose and its summary can be edited. Dialogue is protected verbatim.
    editable = ("summary", "detailed_description")
    frozen = {**out, **{key: "" for key in editable}}
    available = limit - prompt_size(assemble(frozen)) - 64
    if available < 400:
        raise ValueError(
            "Reference and voice descriptions fill the whole video prompt budget; put fewer assets in this shot"
        )
    dialogue = _D_BLOCK.findall(out.get("detailed_description", ""))
    refs = set(
        re.findall(
            r"<(?:Subject|Picture|Audio)\s+\d+>|\(S\d+\)", "\n".join(out.get(key, "") for key in editable)
        )
    )
    system = (
        "Compress only the supplied video caption, without replanning its action. "
        "Preserve ALL start/path/contact/end facts, camera geometry, action order, "
        "style, object and reference isolation constraints, and every numbered "
        "Subject/Picture/Audio and speaker marker. Remove repetitions and verbose "
        "phrasing; do not invent, omit or complete another action. Keep every <d> "
        "dialogue block verbatim, in order, with the same speaker. Keep [Shot 1] "
        "and [reference_generation] markers where present. Write English except "
        "inside dialogue. Return only summary and detailed_description as JSON."
    )
    problem = ""
    tries = config.VIDEO_PROMPT_BUDGET_TRIES if compress else 0
    editable_size = sum(prompt_size(out.get(key, "")) for key in editable)
    if tries and editable_size - available > editable_size * _MAX_MODEL_CUT:
        # The model is told to keep every fact, so it only removes repetition: measured, it took
        # 11 of 1552 characters. A cut this deep goes to sentence trimming first; the model is
        # asked only when trimming cannot fit (one long sentence).
        trimmed = _trim_to_budget(out, limit)
        if trimmed is not None:
            reqlog.record(
                "prompt_budget",
                before=after,
                after=prompt_size(assemble(trimmed)),
                limit=limit,
                method="trim_trailing_sentences",
            )
            return trimmed
    for attempt in range(1, tries + 1):
        current = sum(len(out.get(key, "")) for key in editable)
        payload = {
            "combined_character_budget": available,
            "current_characters": current,
            "targets": {
                "summary": min(240, available // 5),
                "detailed_description": available - min(240, available // 5) - 20,
            },
            "caption": {key: out.get(key, "") for key in editable},
        }
        if problem:
            # Without saying what went wrong last time, a retry just resamples the same distribution
            # and wastes a call.
            payload["previous_attempt_problem"] = problem
        result = llm.chat(
            system=system,
            user=_pretty(payload),
            schema={
                "type": "object",
                "properties": {key: {"type": "string"} for key in editable},
                "required": list(editable),
            },
            model=config.REWRITER_MODEL,
            thinking=0,
            temperature=0.1,
            tag="video_prompt_budget",
        )
        if not isinstance(result, dict) or any(not isinstance(result.get(k), str) for k in editable):
            problem = "the reply was not a complete JSON object with both fields"
            reqlog.record("prompt_budget_retry", attempt=attempt, reason=problem, limit=limit)
            continue
        candidate = {**out, **{key: result[key] for key in editable}}
        compressed_text = "\n".join(candidate[key] for key in editable)
        size = prompt_size(assemble(candidate))
        if not candidate["detailed_description"].strip():
            problem = "detailed_description came back empty"
        elif _D_BLOCK.findall(candidate["detailed_description"]) != dialogue:
            problem = (
                "the <d> dialogue blocks were changed, reordered or dropped; copy every one of them verbatim"
            )
        elif refs != set(re.findall(r"<(?:Subject|Picture|Audio)\s+\d+>|\(S\d+\)", compressed_text)):
            problem = (
                "some <Subject n>/<Picture n>/<Audio n>/(Sn) markers were lost or "
                "added; keep exactly the same set"
            )
        elif size > limit:
            problem = (
                f"the result was still {size - limit} characters over the limit; "
                "cut harder in the prose, never in dialogue or markers"
            )
        else:
            reqlog.record(
                "prompt_budget",
                before=after,
                after=size,
                limit=limit,
                method="caption_only",
                attempts=attempt,
            )
            return candidate
        if size > limit and size < prompt_size(assemble(out)) and problem.startswith("the result was still"):
            # A valid but not short enough result: compress that one further next
            # time instead of starting again from the original.
            out = candidate
        reqlog.record("prompt_budget_retry", attempt=attempt, reason=problem, size=size, limit=limit)
    trimmed = _trim_to_budget(out, limit)
    if trimmed is not None:
        reqlog.record(
            "prompt_budget",
            before=after,
            after=prompt_size(assemble(trimmed)),
            limit=limit,
            method="trim_trailing_sentences",
        )
        return trimmed
    reqlog.record(
        "prompt_budget_failed",
        limit=limit,
        available=available,
        sections={key: len(value or "") for key, value in out.items()},
    )
    raise ValueError(
        f"Video prompt still too long after {tries} compressions: {problem or 'nothing left to trim'}"
    )


_D_BLOCK = re.compile(r"<d\b[^>]*>.*?</d>", re.I | re.S)
