"""When two people's lines are on one line, both must not be locked to the same speaker."""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from types import SimpleNamespace
from murea_agent.shot.dialogue import (
    _dialogue_speaker_labels,
    _lock_dialogue_speakers,
)

assert _dialogue_speaker_labels("艾拉尔喊道：快动手，我撑不住了！莱拉咬牙喊道：别放弃，我正看着它！") == [
    "艾拉尔",
    "莱拉咬牙",
]
assert (
    _dialogue_speaker_labels("艾拉尔虚弱地大喊：“别管印记了……快射碎它！”莱拉咬牙回应：“坚持住！”")[1]
    == "莱拉咬牙"
)
assert _dialogue_speaker_labels("莱拉：走！\n艾拉尔：好。") == ["莱拉", "艾拉尔"]

world = SimpleNamespace(
    subjects={
        "S1": {"name": "莱拉", "kind": "人物", "brief": "你扮演的精灵游侠"},
        "S2": {"name": "艾拉尔", "kind": "人物", "brief": "被封印困住的精灵向导"},
    }
)
pics = [{"kind": "character", "id": "S1", "n": 1}, {"kind": "character", "id": "S2", "n": 2}]
six = {
    "detailed_description": "S2 speaks on-screen as S2: (S2)<d>[Chinese] 快射碎它！</d>"
    "S1 speaks on-screen as S1: (S2)<d>[Chinese] 闭嘴，艾拉尔！</d>"
}
out = _lock_dialogue_speakers(six, "艾拉尔喊道：快射碎它！莱拉咬牙喊道：闭嘴，艾拉尔！", world, pics)
d = out["detailed_description"]
assert "(S2)<d>[Chinese] 快射碎它" in d and "(S1)<d>[Chinese] 闭嘴" in d, d
assert "<Subject 1> speaks on-screen as S1: (S1)" in d, d
# When one name is unknown, the recognized line is still locked and the unknown one stays as is, not
# locked to someone else
six2 = {
    "detailed_description": six["detailed_description"].replace(
        "S1 speaks on-screen as S1: (S2)", "S1 speaks on-screen as S1: (S1)"
    )
}
out = _lock_dialogue_speakers(six2, "艾拉尔喊道：快射碎它！某人：闭嘴！", world, pics)
assert "(S2)<d>[Chinese] 快射碎它" in out["detailed_description"], out
assert "(S1)<d>[Chinese] 闭嘴" in out["detailed_description"], out
print("ok")
