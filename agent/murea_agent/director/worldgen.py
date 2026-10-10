"""Roll a world.

Why not a preset table in the frontend: after ten rolls people recognize them. Here the
model writes a new one each time.

⚠️ **Models repeat themselves when asked for "any X"**: unconstrained, most rolls come back
as the same rainy cyberpunk detective. So the randomness happens **on the server**: draw a few
ingredients (genre, era, kind of place, a problem) and let the model weave them into a world.
The model makes it coherent, not random.

⚠️ The model does **not write the visual style**. visual_style is real conditioning for every
character image, place image and video of the world; an improvised one easily contradicts
itself. The model only **picks an id** from the presets the frontend sends, and the curated
text from the frontend is used, so there is one copy of the style text.
"""

from __future__ import annotations

import random
from typing import Any

from murea_agent import config
from murea_agent.util import _in_story_language, cap
from murea_agent.providers import llm

# Ingredients. The model never sees this table, only the drawn words; the table is the randomness.
_GENRES = [
    "武侠",
    "赛博朋克",
    "克苏鲁式民俗恐怖",
    "太空歌剧",
    "蒸汽朋克",
    "校园日常",
    "职场群像",
    "废土求生",
    "东方仙侠",
    "北欧神话",
    "维多利亚推理",
    "西部拓荒",
    "深海探险",
    "田园牧歌",
    "谍战",
    "怪兽灾害",
    "时间循环",
    "魔法学院",
    "海盗远航",
    "末日温情",
    "宫廷权谋",
    "赛博禅意",
    "民国志怪",
    "近未来医疗",
    "地心探秘",
]
_ERAS = [
    "远古",
    "中世纪",
    "十九世纪末",
    "1930 年代",
    "当代",
    "近未来（二十年后）",
    "远未来",
    "架空年代",
    "灾变之后第三年",
]
_PLACES = [
    "一座孤岛",
    "地下城",
    "移动的列车",
    "山中村落",
    "轨道空间站",
    "老城区的一条巷子",
    "沙漠绿洲",
    "被淹没的城市",
    "边境小镇",
    "巨树内部",
    "废弃医院",
    "港口货仓",
    "雪山驿站",
    "水上市集",
    "钟楼里",
]
_TROUBLES = [
    "有人失踪了",
    "一样东西被偷走",
    "一个约定到期了",
    "外面的人要进来",
    "有人认错了你",
    "一封迟到很久的信",
    "水正在上涨",
    "债主找上门",
    "一个不该醒的东西醒了",
    "所有人都在等同一班船",
    "你身上有一样自己也解释不了的东西",
    "今晚之后这地方就没了",
]

# The same ingredients for English stories: drawing Chinese words made the model title the world
# in Chinese even when told to write in English.
_GENRES_EN = [
    "wuxia",
    "cyberpunk",
    "folk horror",
    "space opera",
    "steampunk",
    "school life",
    "workplace drama",
    "wasteland survival",
    "eastern fantasy",
    "Norse myth",
    "Victorian mystery",
    "frontier western",
    "deep-sea expedition",
    "pastoral idyll",
    "spy thriller",
    "monster disaster",
    "time loop",
    "magic academy",
    "pirate voyage",
    "gentle apocalypse",
    "court intrigue",
    "cyber zen",
    "1920s ghost tale",
    "near-future medicine",
    "journey to the centre of the earth",
]
_ERAS_EN = [
    "ancient times",
    "the Middle Ages",
    "the late 19th century",
    "the 1930s",
    "the present day",
    "twenty years from now",
    "the far future",
    "an invented age",
    "the third year after a catastrophe",
]
_PLACES_EN = [
    "a lone island",
    "an underground city",
    "a moving train",
    "a mountain village",
    "an orbital station",
    "an alley in an old quarter",
    "a desert oasis",
    "a drowned city",
    "a border town",
    "the inside of a giant tree",
    "an abandoned hospital",
    "a harbour warehouse",
    "a snowbound mountain lodge",
    "a floating market",
    "a clock tower",
]
_TROUBLES_EN = [
    "someone has gone missing",
    "something has been stolen",
    "an old promise falls due today",
    "people outside want to get in",
    "someone has mistaken you for someone else",
    "a letter has arrived years too late",
    "the water is rising",
    "a creditor has come calling",
    "something that should not wake has woken",
    "everyone is waiting for the same boat",
    "you carry something even you cannot explain",
    "after tonight this place will be gone",
]


def _ingredients() -> tuple[list[str], list[str], list[str], list[str]]:
    if config.LANGUAGE == "zh":
        return _GENRES, _ERAS, _PLACES, _TROUBLES
    return _GENRES_EN, _ERAS_EN, _PLACES_EN, _TROUBLES_EN


_SYSTEM = """You invent openings for a short interactive video story.

Return ONE world concept. Write the text fields in {LANGUAGE}.

Rules:
- `title`: {TITLE_RULE} Not a sentence. No punctuation.
- `setting`: ONE sentence. Where this is and what kind of place it is. No plot.
- `mainline`: ONE sentence describing a durable objective or conflict that can
  drive many short video turns. It must name what the protagonist ultimately
  needs to accomplish, not a one-second pose or emergency state. Do not write
  "you are trapped", "ten seconds remain", or "you are looking at ..." here.
- `opening`: ONE short sentence in second person, present tense ("You ..."),
  describing what the player is doing at the very first moment. This is where
  a trapped body, countdown, first glimpse, or other transient situation
  belongs; it is context for the first shot, not the quest.
- `style_id`: pick the one id from the provided list that best fits.

Keep every field short. Do not explain, do not add fields."""

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "setting": {"type": "string"},
        "mainline": {"type": "string"},
        "style_id": {"type": "string"},
        "opening": {"type": "string"},
    },
    "required": ["title", "setting", "mainline", "style_id", "opening"],
}

_ASSET_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": ["player", "character", "creature", "object", "place"]},
            "name": {"type": "string"},
            "prompt": {"type": "string"},
        },
        "required": ["kind", "name", "prompt"],
    },
}


def _opening_assets(value: Any) -> list[dict[str, str]]:
    """Validate the opening cast before creating a world or drawing its references."""
    if not isinstance(value, list) or not 2 <= len(value) <= 5:
        raise ValueError("World planning needs 2-5 opening assets; please retry")
    assets = []
    for item in value:
        if not isinstance(item, dict) or any(
            not isinstance(item.get(key), str) or not item[key].strip() for key in ("kind", "name", "prompt")
        ):
            raise ValueError("World planning returned an incomplete opening asset; please retry")
        asset = {key: item[key].strip() for key in ("kind", "name", "prompt")}
        if asset["kind"] not in ("player", "character", "creature", "object", "place"):
            raise ValueError("World planning returned an invalid asset kind; please retry")
        assets.append(asset)
    kinds = [asset["kind"] for asset in assets]
    if kinds.count("player") != 1 or kinds.count("place") != 1:
        raise ValueError("Opening assets need one player and one starting place; please retry")
    if len({asset["name"] for asset in assets}) != len(assets):
        raise ValueError("Opening asset names must be distinct; please retry")
    return assets


# Fallback when the model is unreachable. The dice must still work when an external service
# is down: a random button that does nothing is much worse than an ordinary world.
_FALLBACK = [
    {
        "title": "末班车",
        "setting": "当代城市，深夜，一列开往郊区的地铁末班车。",
        "mainline": "查明列车越过终点仍不停站的原因，并在抵达未知终点前找到脱身办法。",
        "opening": "你坐在空荡的末班车厢里，抬头发现窗外的站名你一个都没见过。",
    },
    {
        "title": "潮汐旅店",
        "setting": "海边一座只在退潮时才能走到的旅店，常年起雾。",
        "mainline": "查清旅店房间重复预订背后的秘密，并找出那个冒名入住者的目的。",
        "opening": "你推开旅店房门，看见床边坐着一个正在整理行李的陌生人。",
    },
    {
        "title": "山中信号",
        "setting": "架空年代，一座只有三个人值守的山顶气象站。",
        "mainline": "追查深夜频道中以你名字开头的呼叫，并找到正在向你求救的人。",
        "opening": "你戴上耳机，听见电流声里有人清清楚楚地念出了你的名字。",
    },
]
_FALLBACK_EN = [
    {
        "title": "Last Train",
        "setting": "A modern city at midnight, on the last metro train to the suburbs.",
        "mainline": "Find out why the train keeps running past the final stop, and get off before it arrives wherever it is going.",
        "opening": "You sit in the empty last carriage and notice that none of the station names outside look familiar.",
    },
    {
        "title": "Tide Inn",
        "setting": "A fog-bound inn on the shore that can only be reached at low tide.",
        "mainline": "Uncover why every room is booked twice, and learn what the guest using your name is after.",
        "opening": "You open the door of your room and find a stranger sitting on the bed, unpacking.",
    },
    {
        "title": "Mountain Signal",
        "setting": "A mountaintop weather station staffed by three people, in no particular era.",
        "mainline": "Trace the late-night radio call that begins with your name, and find the person asking you for help.",
        "opening": "You put on the headphones and hear someone in the static say your name, clear as day.",
    },
]


def _fallbacks() -> list[dict[str, str]]:
    return _FALLBACK if config.LANGUAGE == "zh" else _FALLBACK_EN


def spin(styles: list[dict[str, str]] | None = None, idea: str = "") -> dict[str, Any]:
    """Develop a player's idea, or roll a world when no idea is supplied."""
    idea = idea.strip()
    styles = [s for s in (styles or []) if s.get("id")][:24]
    ids = [s["id"] for s in styles]
    genres, eras, places, troubles = _ingredients()
    seed = {
        "genre": random.choice(genres),
        "era": random.choice(eras),
        "place": random.choice(places),
        "trouble": random.choice(troubles),
    }
    menu = "\n".join(f"- {s['id']}: {s.get('label', s['id'])}" for s in styles)
    user = (
        f"Genre: {seed['genre']}\n"
        f"Era: {seed['era']}\n"
        f"Location type: {seed['place']}\n"
        f"Something is wrong: {seed['trouble']}\n\n"
        f"Available style_id values:\n{menu or '- (none)'}\n\n"
        "Build the world from these four ingredients. Use all four."
    )
    if idea:
        user = (
            f"Player's world idea:\n{idea}\n\n"
            f"Available style_id values:\n{menu or '- (none)'}\n\n"
            "Develop this idea into a playable world. Preserve its named places, beings, genre, "
            "and the player's intent. Fill unspecified details with a coherent starting location, "
            "the player's role, the surroundings and a reason to explore. The mainline should "
            "support an unfolding adventure; the opening is its first visible moment."
        )
    title_rule = (
        "a short Chinese name of 2-8 characters."
        if config.LANGUAGE == "zh"
        else "a short English name of one to four words."
    )
    system = _SYSTEM.replace("{LANGUAGE}", config.LANGUAGE_NAME).replace("{TITLE_RULE}", title_rule)
    schema = _SCHEMA
    if idea:
        system = system.replace(
            f"Write the text fields in {config.LANGUAGE_NAME}.",
            f"Write the story fields in {config.LANGUAGE_NAME}; asset image prompts are in English.",
        ).replace(
            "Keep every field short. Do not explain, do not add fields.",
            "Keep the story fields short. Return only the fields in the response schema.",
        )
        system = system.replace(
            "`setting`: ONE sentence. Where this is and what kind of place it is. No plot.",
            "`setting`: Two or three concrete sentences establishing the world, the player's "
            "role and the starting location. Include distinctive environmental details.",
        )
        system += (
            "\nAlso return `assets`: 2-5 references to prepare before filming the opening. "
            "Include exactly one player (kind=player) and one starting environment (kind=place). "
            "Add characters, creatures or objects only when visually central to this opening. "
            "Each asset has kind, a distinct name in the story language, and an English prompt "
            "of 30-60 words describing its stable visible appearance. The player prompt describes "
            "their face, hair, clothing and equipment; the place prompt describes its layout, "
            "materials and lighting. Use these same names in the opening. These references "
            "establish the first scene; later locations and encounters belong to later turns."
        )
        schema = {
            **_SCHEMA,
            "properties": {**_SCHEMA["properties"], "assets": _ASSET_SCHEMA},
            "required": [*_SCHEMA["required"], "assets"],
        }
    try:
        for attempt in range(2):
            out = llm.chat(
                system=system,
                user=user,
                model=config.GM_MODEL,
                schema=schema,
                temperature=1.0,
                thinking=0,
                tag="world_idea" if idea else "dice",
            )
            # The ingredients can pull a model into another language; ask once more if they did.
            text = (
                [str(out.get(k) or "") for k in ("title", "setting", "mainline", "opening")]
                if isinstance(out, dict)
                else []
            )
            if attempt or all(_in_story_language(t) for t in text):
                break
            user += (
                f"\n\nWrite the title, setting, mainline and opening in {config.LANGUAGE_NAME}. "
                "Asset image prompts remain in English."
            )
    except Exception:
        if idea:
            raise
        out = dict(random.choice(_fallbacks()))
        out["style_id"] = random.choice(ids) if ids else ""
        return out

    if not isinstance(out, dict):
        out = {}
    got: dict[str, Any] = {
        k: " ".join(str(out.get(k) or "").split())[: 1200 if idea and k == "setting" else 200]
        for k in ("title", "setting", "mainline", "opening")
    }
    if idea and not all(got.values()):
        raise ValueError("World planning returned incomplete details; please retry")
    if idea:
        got["assets"] = _opening_assets(out.get("assets"))
    # The model sometimes invents an id that is not in the list or leaves it out; then pick one at
    # random instead of sending an id the frontend does not know, which would silently apply no
    # style.
    style = str(out.get("style_id") or "").strip()
    got["style_id"] = style if style in ids else (random.choice(ids) if ids else "")
    if not got["title"] or not got["mainline"]:
        spare = random.choice(_fallbacks())
        for k, v in spare.items():
            got[k] = got.get(k) or v
    got["title"] = cap(got["title"], 40)
    return got
