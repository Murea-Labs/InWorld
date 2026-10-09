"""Consistency checks. Each case has an explicit **expectation** and reports pass / fail.

  .venv/bin/python tests/live/test_consistency.py [case ...]

cases:
  roundtrip   walk to the door and back: must be continue throughout, no place like "the doorway" appearing
  recall      walk out and come back: returning to an old place must reuse the same reference image
  granularity walk all the way downstairs: transitional spaces must not each become a place
"""

from __future__ import annotations

import json
import sys
import time

import httpx

B = "http://127.0.0.1:8080"
SUBJECT = (
    "A Chinese man in his early thirties, short black hair, light blue dress "
    "shirt with sleeves rolled up, dark grey trousers, neutral expression"
)


def new_world(title, setting):
    w = httpx.post(f"{B}/api/worlds", json={"title": title, "setting": setting}, timeout=30).json()
    httpx.post(
        f"{B}/api/worlds/{w['id']}/subjects", data={"name": "阿明", "image_prompt": SUBJECT}, timeout=240
    )
    return w["id"]


def turn(wid, text):
    t, stages, node = time.time(), [], None
    with httpx.stream("POST", f"{B}/api/worlds/{wid}/turn", json={"text": text}, timeout=900) as r:
        for line in r.iter_lines():
            if not line.startswith("data:"):
                continue
            ev = json.loads(line[5:])
            if ev["type"] == "progress":
                stages.append(ev["stage"])
            elif ev["type"] == "error":
                raise RuntimeError(ev["message"])
            else:
                node = ev["node"]
    node["_stages"] = stages
    node["_wall"] = round(time.time() - t, 1)
    tm = node.get("timings", {})
    print(
        f"    [{node['mode']:8}] {node['place']:6} "
        f"{'+新图' if 'image' in stages else '复用图':6} "
        f"{node['_wall']:6.1f}s  (gm {tm.get('gm', 0):.0f} / 改写 {tm.get('rewrite', 0):.0f} "
        f"/ video {tm.get('video', 0):.0f}s)  ← {text}"
    )
    return node


def check(name, ok, detail=""):
    print(f"    {'✅' if ok else '❌'} {name}" + (f" —— {detail}" if detail else ""))
    return ok


# ---------------------------------------------------------------- cases


def case_roundtrip():
    """Walk to the door and back. All in one room, so it must continue throughout."""
    print("\n== roundtrip：走到门口又回去 ==")
    wid = new_world("往返测试", "现代写字楼会议室，阿明一个人在里面")
    ns = [
        turn(wid, t)
        for t in [
            "阿明坐在会议室的桌边，翻看手上的文件",
            "他站起来，走到会议室门口",
            "他在门口停下，转身走回桌边",
            "他坐回椅子上，继续看文件",
        ]
    ]
    W = httpx.get(f"{B}/api/worlds/{wid}", timeout=30).json()
    ok = True
    ok &= check(
        "第 2–4 段没被误判成切场景",
        all(n["mode"] == "continue" for n in ns[1:]),
        "实际 " + " / ".join(n["mode"] for n in ns),
    )
    ok &= check("没有多余地点", len(W["places"]) == 1, f"places={list(W['places'])}")
    cont = all(n["continued_from"] for n in ns[1:])
    ok &= check("留在同一个房间（mode=continue）", all(n["mode"] == "continue" for n in ns[1:]))
    print(f"    ·  latent 续写：{'有' if cont else '无（视频服务不支持，靠参考图保持）'}")
    ok &= check("只生了一次图", sum("image" in n["_stages"] for n in ns) == 1)
    return ok, wid


def case_recall():
    """Walk out and come back. Returning to an old place must reuse the same reference image."""
    print("\n== recall：出去再回来 ==")
    wid = new_world("复用测试", "现代写字楼，阿明从会议室出去又回来")
    ns = [
        turn(wid, t)
        for t in [
            "阿明坐在会议室里看文件",
            "他推门走出去，来到外面的走廊",
            "他在走廊上站了一会",
            "他推门走回会议室，坐回原来的位置",
        ]
    ]
    W = httpx.get(f"{B}/api/worlds/{wid}", timeout=30).json()
    ok = True
    ok &= check(
        "最后一段回到了会议室", ns[3]["place"] == ns[0]["place"], f"{ns[0]['place']} -> {ns[3]['place']}"
    )
    ok &= check(
        "复用了同一张场景参考图",
        ns[3]["refs"][-1] == ns[0]["refs"][-1],
        f"{ns[0]['refs'][-1]} vs {ns[3]['refs'][-1]}",
    )
    ok &= check("回去那一段没有重新生图", "image" not in ns[3]["_stages"])
    ok &= check("地点数 == 2", len(W["places"]) == 2, f"places={list(W['places'])}")
    return ok, wid


def case_granularity():
    """Walk all the way downstairs. Lifts and stairs must not each become a place."""
    print("\n== granularity：一路走到楼下 ==")
    wid = new_world("粒度测试", "现代写字楼，阿明从会议室下楼到街上")
    for t in [
        "阿明在会议室里收拾东西",
        "他走出会议室，穿过走廊到电梯口",
        "他坐电梯下到一楼",
        "他穿过大堂走到街上",
    ]:
        turn(wid, t)
    W = httpx.get(f"{B}/api/worlds/{wid}", timeout=30).json()
    n = len(W["places"])
    ok = check("地点数 ≤ 4（原来会到 5–6 个）", n <= 4, f"{n} 个：{list(W['places'])}")
    return ok, wid


def case_longrun():
    """Long session. After 12 clips: did character descriptions drift, did places get out of hand, are old places still recognized."""
    print("\n== longrun：12 段长对话 ==")
    wid = new_world("长对话测试", "现代写字楼，阿明的一个下午")
    beats = [
        "阿明坐在会议室里翻文件",
        "他抬头看了看窗外",
        "他站起来走到白板前",
        "他在白板上写了几个字",
        "他走回座位坐下",
        "他推门走出去，来到走廊",
        "他沿着走廊往前走",
        "他推开门进了茶水间",
        "他给自己倒了杯水",
        "他端着杯子靠在台子上发呆",
        "他走回走廊",
        "他推门回到会议室，坐回原来的位置",
    ]
    ns = [turn(wid, t) for t in beats]
    W = httpx.get(f"{B}/api/worlds/{wid}", timeout=30).json()

    first_place = ns[0]["place"]
    last = ns[-1]
    # The place image of the first clip vs the last (should be back in the same place)
    same_ref = last["refs"][-1] == ns[0]["refs"][-1] if last["place"] == first_place else None

    # Locked character descriptions: written once per session, then reused verbatim
    defs = set()
    for n in ns:
        for line in (n["six_section"].get("subject_definitions") or "").splitlines():
            if line.strip().startswith("<Subject 1>"):
                defs.add(line.strip())

    ok = True
    ok &= check("地点数 ≤ 5", len(W["places"]) <= 5, f"{len(W['places'])} 个：{list(W['places'])}")
    ok &= check(
        "角色描述全程一致（没漂）", len(defs) <= 1, f"出现了 {len(defs)} 种写法" if len(defs) > 1 else ""
    )
    if last["place"] == first_place:
        ok &= check(
            "最后回到会议室并复用了原图", bool(same_ref), f"{ns[0]['refs'][-1]} vs {last['refs'][-1]}"
        )
    else:
        ok &= check("最后回到了起点", False, f"停在 {last['place']}")
    ok &= check(
        "生图次数 == 地点数",
        sum("image" in n["_stages"] for n in ns) == len(W["places"]),
        f"生图 {sum('image' in n['_stages'] for n in ns)} 次 / 地点 {len(W['places'])} 个",
    )
    tot = sum(n["_wall"] for n in ns)
    print(f"    ⏱  {len(ns)} 段共 {tot:.0f}s，平均 {tot / len(ns):.1f}s/段")
    return ok, wid


CASES = {
    "roundtrip": case_roundtrip,
    "recall": case_recall,
    "granularity": case_granularity,
    "longrun": case_longrun,
}


def main() -> int:
    names = sys.argv[1:] or list(CASES)
    results = {}
    for name in names:
        if name not in CASES:
            print(f"未知 case: {name}")
            return 2
        try:
            ok, wid = CASES[name]()
            results[name] = (ok, wid)
        except Exception as exc:
            print(f"    !! {type(exc).__name__}: {exc}")
            results[name] = (False, "")
    print("\n" + "=" * 56)
    for name, (ok, wid) in results.items():
        print(f"{'PASS' if ok else 'FAIL'}  {name:12} {wid}")
    return 0 if all(ok for ok, _ in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
