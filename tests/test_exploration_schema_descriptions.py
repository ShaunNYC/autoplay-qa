"""Schema descriptions guide grounding without enforcing semantic truth."""

import json
import unittest
from unittest.mock import patch

from autoplay_qa.config import Settings
from autoplay_qa.exploration import (
    EXPLORATION_SCHEMA, Coverage, ExplorationDecision, exploration_messages, outcome_view, turn_schema,
)
from autoplay_qa.game import KeyGateGame
from autoplay_qa.nebius import NebiusClient
from tests.test_exploration import decision, hypothesis, response


class SchemaDescriptionTests(unittest.TestCase):
    def test_grounding_descriptions_reach_context_and_schema_decoder(self):
        game = KeyGateGame()
        before = game.observe()
        coverage = Coverage(before)
        after = game.act("south")
        coverage.record(before, "south", after, step=1, episode=0)
        history = [{"step": 1, "episode_before": 0, "episode_after": 0,
                    "experiment_id": "chosen-question", "action": "south",
                    "prediction": "Player changes position.",
                    "before": outcome_view(before), "after": outcome_view(after)}]
        schema = turn_schema(["finish"], {1}, 1, require_learning=True)
        fields = schema["properties"]
        self.assertIn("ONE narrow experiment question", fields["experiment_id"]["description"])
        self.assertIn("distinct IDs", fields["experiment_id"]["description"])
        self.assertIn("separate from the broader session question", fields["experiment_goal"]["description"])
        self.assertIn("immediate observable outcome", fields["prediction"]["description"])
        self.assertIn("unexecuted future trials", fields["prediction"]["description"])
        claim = fields["hypotheses"]["items"]["properties"]["claim"]
        self.assertIn("actual recorded actions", claim["description"])
        self.assertIn("untested proposals may have empty evidence", claim["description"])
        messages = exploration_messages(after, 0, None, {}, history, coverage,
                                        1, {"actions": 23}, response_schema=schema)
        self.assertEqual(json.loads(messages[1]["content"])["response_schema"], schema)
        client = NebiusClient(Settings("unit-test-key", response_format="json_schema"))
        answer = decision("finish", 1, [hypothesis(status="uncertain", evidence=[1])])
        with patch.object(client, "_request", return_value=response(answer)) as transport:
            client.complete(messages, response_schema=schema)
        self.assertEqual(transport.call_args.args[1]["response_format"]["json_schema"]["schema"], schema)

    def test_descriptions_preserve_string_limits_and_do_not_enforce_semantics(self):
        for name, limit in (("experiment_id", 32), ("experiment_goal", 240), ("prediction", 240)):
            field = EXPLORATION_SCHEMA["properties"][name]
            self.assertEqual({key: value for key, value in field.items() if key != "description"},
                             {"type": "string", "minLength": 1, "maxLength": limit})
        claim = EXPLORATION_SCHEMA["properties"]["hypotheses"]["items"]["properties"]["claim"]
        self.assertEqual({key: value for key, value in claim.items() if key != "description"},
                         {"type": "string", "minLength": 1, "maxLength": 240})
        value = decision("south", goal="A broad combined movement and interaction question",
                         prediction="Future trials will all match after later resets",
                         hypotheses=[hypothesis(status="untested", evidence=[])])
        parsed = ExplorationDecision.parse(json.dumps(value), ["south"], set(), None)
        # These structurally valid but poorly scoped fields still need review.
        self.assertEqual(parsed.experiment_goal, value["experiment_goal"])
        self.assertEqual(parsed.prediction, value["prediction"])
        self.assertEqual(parsed.hypotheses[0].evidence_steps, ())


if __name__ == "__main__":
    unittest.main()
