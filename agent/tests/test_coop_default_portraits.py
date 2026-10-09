"""Automatic portraits apply to each player without granting host edit rights."""

import unittest
from unittest.mock import patch
from tests import test_coop as helpers
from murea_agent.coop import portrait as coop_portrait


class DefaultPortraitTest(unittest.TestCase):
    setUp = helpers.CoopTest.setUp
    tearDown = helpers.CoopTest.tearDown
    create = helpers.CoopTest.create
    local_room = helpers.CoopTest.local_room
    local_role = helpers.CoopTest.local_role

    def render_defaults(self):
        with patch.object(coop_portrait, "draw", return_value=helpers.picture()):
            coop_portrait._run_defaults(self.code, coop_portrait._guard())
        room = self.a.get(self.url).json()
        self.assertEqual(2, len(room["members"]))
        for member in room["members"]:
            self.assertTrue(member["ref"], member)
            self.assertEqual("applied", member["portrait"]["status"])
        return room

    def test_online_guest_receives_default_without_host_edit_permission(self):
        self.create(capacity=2)
        self.assertEqual(200, self.b.post(self.url + "/join").status_code)
        room = self.render_defaults()
        guest = next(m for m in room["members"] if m["user_id"] == self.ub["id"])
        response = self.a.post(
            self.url + "/portrait/use",
            json={"target_uid": guest["user_id"], "request_id": guest["portrait"]["request_id"]},
        )
        self.assertEqual(403, response.status_code)

    def test_local_seats_remain_controlled_by_host(self):
        self.local_room()
        self.render_defaults()

    def test_an_upload_during_the_default_drawing_is_kept(self):
        # Real run: the host uploaded a character right after creating the room; the default
        # portrait, drawn meanwhile (~6 s), then replaced the upload.
        room = self.local_room()
        first = room["members"][0]["user_id"]
        uploaded = {}

        def seat():
            return next(m for m in self.a.get(self.url).json()["members"] if m["user_id"] == first)

        def draw_while_player_uploads(*args, **kwargs):
            if not uploaded:
                self.local_role(first, "Courier")
                uploaded.update(seat())
            return helpers.picture()

        with patch.object(coop_portrait, "draw", side_effect=draw_while_player_uploads):
            coop_portrait._run_defaults(self.code, coop_portrait._guard())
        member = seat()
        self.assertTrue(uploaded.get("ref"))
        self.assertEqual(uploaded["ref"], member["ref"])
        self.assertEqual("Courier", member["character"])
        # The drawn default stays a preview the player can still adopt.
        self.assertEqual("ready", member["portrait"]["status"])


if __name__ == "__main__":
    unittest.main()
