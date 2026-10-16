"""Explicit model-plan fixtures test commitment facts, not live research success."""

import copy
import json
import unittest

from autoplay_qa.game import KeyGateGame
from autoplay_qa.nebius import InvalidDecision
from autoplay_qa.research import (
    PLAN_CONTROLS, plan_context, plan_schema, research_context, validate_plan,
)


def proposal(controls, *, mode="new", reason="", evidence=()):
    return {"mode": mode, "controls": list(controls), "revision_reason": reason, "evidence_steps": list(evidence)}


def completed_plans(plans, *, identifiers=None, seed=0):
    """Run only these explicit local fixture controls; no model/inference occurs."""
    game, episode, records = KeyGateGame(seed), 0, {}
    identifiers = identifiers or ["question"] * len(plans)
    for number, (plan, identifier) in enumerate(zip(plans, identifiers), 1):
        action = plan["controls"][0]
        before = game.observe()
        if action == "reset":
            previous = episode
            episode += 1
            game = KeyGateGame(seed)
            metadata = {"type": "reset", "episode_before": previous, "episode_after": episode}
            after = game.observe()
        else:
            metadata = {"type": "step", "episode": episode}
            after = game.act(action)
        records[number] = {
            **metadata, "step": number,
            "decision": {"action": action, "experiment_id": identifier, "plan": copy.deepcopy(plan)},
            "before": before, "after": after,
        }
    return records, game.observe()


class ResearchContractTests(unittest.TestCase):
    def validate(self, value, *, records=None, action=None, identifier="question", allowed=PLAN_CONTROLS):
        return validate_plan(value, action=action or value["controls"][0], experiment_id=identifier,
                             records={} if records is None else records, allowed=list(allowed))

    def baseline(self):
        return completed_plans([proposal(["interact", "reset", "interact"])])

    def test_every_current_control_is_available_in_a_new_voluntary_plan(self):
        for control in PLAN_CONTROLS:
            value = proposal([control])
            with self.subTest(control=control):
                self.assertEqual(self.validate(value), value)

    def test_terminal_head_permissions_do_not_forbid_future_controls_after_reset(self):
        value = proposal(["reset", "north", "interact", "finish"])
        self.assertEqual(self.validate(value, allowed=("reset", "finish")), value)
        with self.assertRaisesRegex(InvalidDecision, "allowed"):
            self.validate(proposal(["north"]), allowed=("reset", "finish"))

    def test_first_control_must_be_the_actual_action_not_an_imagined_preparation(self):
        with self.assertRaisesRegex(InvalidDecision, "first planned control"):
            self.validate(proposal(["south", "interact"]), action="interact")

    def test_actual_row_consumes_one_head_and_only_retains_unexecuted_intent(self):
        records, observation = self.baseline()
        context = plan_context(records, observation)
        self.assertEqual(context["pending_plan"], {"source_step": 1, "experiment_id": "question", "controls": ["reset", "interact"]})
        self.assertEqual(context["completed_steps"], [1])
        self.assertEqual(context["current_state_action_attempts"]["interact"], 1)
        self.assertFalse(context["claims_verified"])
        self.assertFalse(context["session_finished"])
        self.assertEqual(research_context(records, observation), context)

    def test_accepting_or_checking_an_unexecuted_proposal_does_not_consume_its_head(self):
        records, observation = self.baseline()
        before = copy.deepcopy(records)
        current = plan_context(records, observation)
        continuation = proposal(["reset", "interact"], mode="continue")
        self.validate(continuation, records=records)
        self.assertEqual(plan_context(records, observation), current)
        self.assertEqual(records, before)

    def test_explicit_local_across_reset_sequence_continues_the_models_own_plan(self):
        first = proposal(["interact", "reset", "interact"])
        records, _ = completed_plans([first])
        second = proposal(["reset", "interact"], mode="continue")
        self.assertEqual(self.validate(second, records=records), second)
        records, _ = completed_plans([first, second])
        third = proposal(["interact"], mode="continue")
        self.assertEqual(self.validate(third, records=records), third)
        records, observation = completed_plans([first, second, third])
        context = plan_context(records, observation)
        self.assertIsNone(context["pending_plan"])
        self.assertEqual(context["completed_steps"], [1, 2, 3])
        self.assertEqual(records[1]["episode"], 0)
        self.assertEqual(records[3]["episode"], 1)
        self.assertEqual(context["current_state_action_attempts"]["interact"], 2)

    def test_the_observed_interact_loop_cannot_pretend_to_continue_its_reset_tail(self):
        records, _ = self.baseline()
        for changed in (proposal(["interact", "reset", "interact"], mode="continue"), proposal(["interact"], mode="new")):
            with self.subTest(plan=changed), self.assertRaisesRegex(InvalidDecision, "remaining controls|tail remains"):
                self.validate(changed, records=records)
        with self.assertRaisesRegex(InvalidDecision, "first planned control"):
            self.validate(proposal(["reset", "interact"], mode="continue"), records=records, action="interact")

    def test_continuation_keeps_exact_tail_and_question_identity(self):
        records, _ = self.baseline()
        for value, identifier in ((proposal(["reset"], mode="continue"), "question"),
                                  (proposal(["reset", "interact"], mode="continue"), "renamed-question")):
            with self.subTest(plan=value, identifier=identifier), self.assertRaisesRegex(InvalidDecision, "experiment ID"):
                self.validate(value, records=records, identifier=identifier)

    def test_revisions_freely_choose_every_control_with_real_evidence(self):
        records, _ = self.baseline()
        for control in PLAN_CONTROLS:
            value = proposal([control], mode="revise", reason="The observed feedback changed my question.", evidence=[1])
            with self.subTest(control=control):
                self.assertEqual(self.validate(value, records=records, identifier="new-question"), value)

    def test_revised_evidence_may_cite_actual_reset_for_replanning(self):
        records, _ = completed_plans([proposal(["interact", "reset", "interact"]), proposal(["reset", "interact"], mode="continue")])
        value = proposal(["north"], mode="revise", reason="The restart restored the sampled conditions; I chose another question.", evidence=[2])
        self.assertEqual(self.validate(value, records=records, identifier="movement"), value)

    def test_revisions_need_reason_and_real_unique_past_references(self):
        records, _ = self.baseline()
        failures = (
            proposal(["interact"], mode="revise", evidence=[1]),
            proposal(["interact"], mode="revise", reason="Reconsidered."),
            proposal(["interact"], mode="revise", reason="Reconsidered.", evidence=[2]),
            proposal(["interact"], mode="revise", reason="Reconsidered.", evidence=[True]),
            proposal(["interact"], mode="revise", reason="Reconsidered.", evidence=[1, 1]),
        )
        for value in failures:
            before = copy.deepcopy(records)
            with self.subTest(plan=value), self.assertRaises(InvalidDecision):
                self.validate(value, records=records)
            self.assertEqual(records, before)

    def test_empty_tail_allows_further_sampling_without_a_hidden_quota(self):
        records, _ = completed_plans([proposal(["interact"])] * 12)
        value = proposal(["interact", "interact", "interact", "interact"])
        self.assertEqual(self.validate(value, records=records), value)
        changed = proposal(["north"], mode="revise", reason="I selected another question after reviewing the trial.", evidence=[12])
        self.assertEqual(self.validate(changed, records=records, identifier="movement"), changed)
        with self.assertRaisesRegex(InvalidDecision, "No recorded plan tail"):
            self.validate(proposal(["interact"], mode="continue"), records=records)

    def test_new_and_continue_do_not_smuggle_revision_metadata(self):
        records, _ = self.baseline()
        for mode, refs in (("new", []), ("new", [1]), ("continue", []), ("continue", [1])):
            value = proposal(["reset", "interact"] if mode == "continue" else ["north"], mode=mode, reason="Unannounced revision.", evidence=refs)
            with self.subTest(mode=mode, evidence=refs), self.assertRaises(InvalidDecision):
                self.validate(value, records=records if mode == "continue" or refs else {})

    def test_all_control_array_text_and_reference_bounds_are_checked(self):
        good = proposal(["north"])
        bad_values = [None, [], {**good, "extra": 1}]
        for field, values in {
            "mode": [None, [], True, "unknown"],
            "controls": [None, "north", [], ["north"] * 5, [True], ["teleport"], ["finish", "north"], ["north", "finish", "finish"]],
            "revision_reason": [None, False, "x" * 241],
            "evidence_steps": [None, 1, [1] * 9],
        }.items():
            bad_values.extend({**good, field: value} for value in values)
        for value in bad_values:
            with self.subTest(value=value), self.assertRaises(InvalidDecision):
                validate_plan(value, action="north", experiment_id="question", records={}, allowed=list(PLAN_CONTROLS))

    def test_finish_stops_the_chosen_plan_and_is_never_counted_as_a_game_action(self):
        records, observation = self.baseline()
        value = proposal(["finish"], mode="revise", reason="Question remains inconclusive within this budget.", evidence=[1])
        self.assertEqual(self.validate(value, records=records), value)
        unfinished, finished = plan_context(records, observation), plan_context(records, observation, finished=True)
        self.assertIsNotNone(unfinished["pending_plan"])
        self.assertIsNone(finished["pending_plan"])
        self.assertTrue(finished["session_finished"])
        self.assertEqual(unfinished["current_state_action_attempts"], finished["current_state_action_attempts"])
        self.assertEqual(unfinished["completed_steps"], finished["completed_steps"])
        with self.assertRaises(InvalidDecision):
            plan_context(records, observation, finished=1)

    def test_current_state_attempt_counts_exclude_ticks_feedback_and_withheld_top_level_fields(self):
        records, observation = completed_plans([proposal(["interact"]), proposal(["interact"]), proposal(["north"])])
        modified = copy.deepcopy(observation)
        modified.update(tick=999, events=["An irrelevant previous message."], rules=["withheld-rule-marker"], goal="withheld-objective-marker", strategy="withheld-route-marker", passability_hints="withheld-hint-marker")
        context = plan_context(records, modified)
        self.assertEqual(context["current_state_action_attempts"]["interact"], 2)
        self.assertEqual(context["current_state_action_attempts"]["north"], 1)
        for marker in ("withheld-rule-marker", "withheld-objective-marker", "withheld-route-marker", "withheld-hint-marker"):
            self.assertNotIn(marker, json.dumps(context))

    def test_changed_position_inventory_objects_map_or_status_is_not_the_same_start(self):
        records, observation = self.baseline()
        changes = (
            lambda value: value["player"].update(position=[2, 1]),
            lambda value: value["player"].update(inventory=["key"]),
            lambda value: value["objects"]["gate"].update(open=True),
            lambda value: value["map"].__setitem__(1, "#. @#...#"),
            lambda value: value.update(status="won"),
        )
        for change in changes:
            current = copy.deepcopy(observation)
            change(current)
            with self.subTest(state=current):
                self.assertEqual(plan_context(records, current)["current_state_action_attempts"]["interact"], 0)

    def test_revisions_are_recorded_facts_and_do_not_verify_the_reason(self):
        records, observation = completed_plans([
            proposal(["interact", "reset", "interact"]),
            proposal(["north"], mode="revise", reason="A model-written explanation remains unverified.", evidence=[1]),
        ], identifiers=["interaction", "movement"])
        context = plan_context(records, observation)
        self.assertEqual(context["plan_revisions"], 1)
        self.assertFalse(context["claims_verified"])
        self.assertNotIn("model-written explanation", json.dumps(context))

    def test_history_rejects_inconsistent_control_episode_reference_and_plan_metadata(self):
        records, observation = self.baseline()
        mutations = (
            lambda value: value[1].update(type="finish"),
            lambda value: value[1].update(step=True),
            lambda value: value[1].update(episode=True),
            lambda value: value[1]["decision"].update(action="reset"),
            lambda value: value[1]["decision"]["plan"].update(controls=["north"]),
            lambda value: value[1]["decision"]["plan"].update(mode="revise", revision_reason="Future evidence.", evidence_steps=[1]),
        )
        for mutate in mutations:
            wrong = copy.deepcopy(records)
            mutate(wrong)
            with self.subTest(records=wrong), self.assertRaises(InvalidDecision):
                plan_context(wrong, observation)
        for bad_keys in ({True: records[1]}, {0: records[1]}, {"1": records[1]}):
            with self.subTest(records=bad_keys), self.assertRaises(InvalidDecision):
                plan_context(bad_keys, observation)

    def test_history_rejects_silent_plan_abandonment_under_a_new_id(self):
        records, observation = completed_plans([proposal(["interact", "reset", "interact"]), proposal(["interact"])], identifiers=["first-question", "renamed-question"])
        with self.assertRaisesRegex(InvalidDecision, "tail remains"):
            plan_context(records, observation)

    def test_schema_keeps_all_current_heads_and_all_future_controls(self):
        context = plan_context({}, KeyGateGame().observe())
        schema = plan_schema(list(PLAN_CONTROLS), context)
        self.assertEqual(schema["required"], ["mode", "controls", "revision_reason", "evidence_steps"])
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(schema["properties"]["controls"]["items"]["enum"], list(PLAN_CONTROLS))
        self.assertEqual(schema["anyOf"][0]["properties"]["mode"]["enum"], ["new"])
        for control in PLAN_CONTROLS:
            branch = plan_schema(list(PLAN_CONTROLS), context, action=control)
            arrays = branch["properties"]["controls"]["anyOf"]
            self.assertTrue(arrays)
            for value in arrays:
                self.assertEqual(value["prefixItems"][0]["enum"], [control])
                self.assertEqual(value["minItems"], value["maxItems"])
                self.assertLessEqual(value["maxItems"], 4)
            if control == "finish":
                self.assertEqual(len(arrays), 1)
                self.assertEqual(arrays[0]["maxItems"], 1)

    def test_schema_filters_incompatible_continuation_without_restricting_revision(self):
        records, observation = self.baseline()
        context = plan_context(records, observation)
        for control in PLAN_CONTROLS:
            schema = plan_schema(list(PLAN_CONTROLS), context, action=control)
            alternatives = {branch["properties"]["mode"]["enum"][0]: branch for branch in schema["anyOf"]}
            with self.subTest(control=control):
                self.assertIn("revise", alternatives)
                self.assertNotIn("new", alternatives)
                self.assertEqual("continue" in alternatives, control == "reset")
                self.assertEqual(alternatives["revise"]["properties"]["evidence_steps"]["minItems"], 1)
                self.assertEqual(alternatives["revise"]["properties"]["evidence_steps"]["items"]["enum"], [1])
                if control == "reset":
                    self.assertEqual(alternatives["continue"]["properties"]["controls"]["enum"], [["reset", "interact"]])

    def test_schema_and_context_are_pure_and_revision_does_not_supply_a_route(self):
        records, observation = self.baseline()
        original = copy.deepcopy((records, observation))
        context = plan_context(records, observation)
        copied = copy.deepcopy(context)
        schema = plan_schema(list(PLAN_CONTROLS), context, action="north")
        self.assertEqual(context, copied)
        self.assertEqual((records, observation), original)
        self.assertEqual(schema["properties"]["controls"]["anyOf"][1]["prefixItems"][1]["enum"], list(PLAN_CONTROLS))
        returned = self.validate(proposal(["reset", "interact"], mode="continue"), records=records)
        returned["controls"][0] = "finish"
        self.assertEqual(context["pending_plan"]["controls"], ["reset", "interact"])

    def test_schema_rejects_invalid_context_permissions_and_finished_session(self):
        context = plan_context({}, KeyGateGame().observe())
        values = (None, {}, {**context, "contract_version": True}, {**context, "completed_steps": [True]}, {**context, "completed_steps": [2, 1]}, {**context, "completed_steps": [1, 1]}, {**context, "session_finished": True}, {**context, "pending_plan": {"controls": []}})
        for value in values:
            with self.subTest(context=value), self.assertRaises(InvalidDecision):
                plan_schema(list(PLAN_CONTROLS), value)
        for allowed in ([], ["north", "north"], [False], ["teleport"]):
            with self.subTest(allowed=allowed), self.assertRaises(InvalidDecision):
                plan_schema(allowed, context)
        with self.assertRaises(InvalidDecision):
            plan_schema(["reset", "finish"], context, action="north")

    def test_schema_pending_plan_metadata_keeps_source_control_and_type_bounds(self):
        records, observation = self.baseline()
        context = plan_context(records, observation)
        wrong = (
            {"source_step": True, "experiment_id": "question", "controls": ["reset"]},
            {"source_step": 2, "experiment_id": "question", "controls": ["reset"]},
            {"source_step": 1, "experiment_id": "question", "controls": ["reset"] * 4},
            {"source_step": 1, "experiment_id": "question", "controls": [True]},
            {"source_step": 1, "experiment_id": "question", "controls": ["finish", "north"]},
            {"source_step": 1, "experiment_id": None, "controls": ["reset"]},
        )
        for pending in wrong:
            with self.subTest(pending=pending), self.assertRaises(InvalidDecision):
                plan_schema(list(PLAN_CONTROLS), {**context, "pending_plan": pending})


if __name__ == "__main__":
    unittest.main()
