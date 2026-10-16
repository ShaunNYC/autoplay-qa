"""Question switches and cited learning are facts, not an autonomous policy."""

import copy
import json
import unittest

from autoplay_qa.game import KeyGateGame
from autoplay_qa.nebius import InvalidDecision
from autoplay_qa.research import RESEARCH_CONTRACT_VERSION as OLD_VERSION
from autoplay_qa.research2 import (
    PLAN_CONTROLS, RESEARCH_CONTRACT_VERSION, experiment_goals, plan_context,
    plan_schema, research_context, scope_progress, unnoted_trial_steps,
    validate_goal, validate_plan,
)


def plan(controls, mode="new", reason="", refs=()):
    return {"mode": mode, "controls": list(controls), "revision_reason": reason, "evidence_steps": list(refs)}


def choice(controls, *, identifier="movement", target="east", role=None, goal=None, mode="new", reason="", refs=()):
    action = controls[0]
    return {
        "action": action, "experiment_id": identifier, "experiment_control": target,
        "experiment_goal": goal or f"Question {identifier}: observe {target} under sampled conditions.",
        "action_role": role or ("trial" if action == target else "setup"),
        "plan": plan(controls, mode, reason, refs),
    }


def executed(choices, seed=0):
    """Actual local fixture transitions; no model or API chooses these actions."""
    game, episode, records = KeyGateGame(seed), 0, {}
    for number, decision in enumerate(choices, 1):
        action, before = decision["action"], game.observe()
        if action == "reset":
            previous = episode
            episode += 1
            game = KeyGateGame(seed)
            metadata = {"type": "reset", "episode_before": previous, "episode_after": episode}
            after = game.observe()
        else:
            metadata = {"type": "step", "episode": episode}
            after = game.act(action)
        records[number] = {**metadata, "step": number, "decision": copy.deepcopy(decision), "before": before, "after": after}
    return records, game.observe()


def note(identifier, control, refs=(), *, status="uncertain", comparison=None, claim="Tentative model claim."):
    return {"id": identifier, "control": control, "evidence_steps": list(refs),
            "status": status, "comparison": comparison, "claim": claim}


class Research2Tests(unittest.TestCase):
    def validate(self, value, *, records=None, action=None, identifier="movement", allowed=PLAN_CONTROLS):
        return validate_plan(value, action=action or value["controls"][0], experiment_id=identifier,
                             records={} if records is None else records, allowed=list(allowed))

    def retained(self):
        return executed([choice(["east", "interact", "reset", "interact"])])

    def pair(self):
        return executed([
            choice(["east", "reset", "east"]),
            choice(["reset", "east"], mode="continue"),
            choice(["east"], mode="continue"),
        ])

    def test_old_research_contract_is_unchanged(self):
        self.assertEqual(OLD_VERSION, 1)
        self.assertEqual(RESEARCH_CONTRACT_VERSION, 2)

    def test_all_allowed_heads_remain_available_in_new_and_revised_plans(self):
        records, _ = self.retained()
        for control in PLAN_CONTROLS:
            with self.subTest(control=control):
                fresh = plan([control])
                self.assertEqual(self.validate(fresh), fresh)
                revised = plan([control], "revise", "I chose another operation after observing a trial.", [1])
                self.assertEqual(self.validate(revised, records=records, identifier="other"), revised)

    def test_terminal_permissions_only_limit_the_actual_head(self):
        value = plan(["reset", "north", "interact", "finish"])
        self.assertEqual(self.validate(value, allowed=("reset", "finish")), value)
        with self.assertRaises(InvalidDecision):
            self.validate(plan(["north"]), allowed=("reset", "finish"))

    def test_retarget_preserves_controls_and_changes_only_the_question(self):
        records, _ = self.retained()
        value = plan(["interact", "reset", "interact"], "retarget", "I chose a separate interaction question.", [1])
        self.assertEqual(self.validate(value, records=records, identifier="interaction"), value)

    def test_retarget_can_resume_an_old_scope(self):
        records, _ = executed([
            choice(["east"]),
            choice(["wait", "reset"], identifier="pause", target="wait"),
        ])
        value = plan(["reset"], "retarget", "Return to the earlier sampled movement question.", [1])
        self.assertEqual(self.validate(value, records=records, identifier="movement"), value)

    def test_retarget_requires_pending_tail_different_id_reason_and_past_evidence(self):
        records, _ = self.retained()
        good = plan(["interact", "reset", "interact"], "retarget", "A chosen independent question.", [1])
        bad = [
            (good, {}, "interaction"),
            (good, records, "movement"),
            ({**good, "controls": ["interact", "reset"]}, records, "interaction"),
            ({**good, "revision_reason": " "}, records, "interaction"),
            ({**good, "evidence_steps": []}, records, "interaction"),
            ({**good, "evidence_steps": [1, 1]}, records, "interaction"),
            ({**good, "evidence_steps": [True]}, records, "interaction"),
            ({**good, "evidence_steps": [2]}, records, "interaction"),
        ]
        for value, history, identifier in bad:
            with self.subTest(value=value, identifier=identifier), self.assertRaises(InvalidDecision):
                self.validate(value, records=history, identifier=identifier)

    def test_retarget_cannot_hide_a_different_action_or_disallowed_head(self):
        records, _ = self.retained()
        value = plan(["interact", "reset", "interact"], "retarget", "Different question.", [1])
        with self.assertRaises(InvalidDecision):
            self.validate(value, records=records, action="north", identifier="interaction")
        with self.assertRaises(InvalidDecision):
            self.validate(value, records=records, identifier="interaction", allowed=("reset", "finish"))

    def test_continue_does_not_silently_switch_ids_and_new_does_not_abandon_tail(self):
        records, _ = self.retained()
        with self.assertRaises(InvalidDecision):
            self.validate(plan(["interact", "reset", "interact"], "continue"), records=records, identifier="interaction")
        with self.assertRaises(InvalidDecision):
            self.validate(plan(["interact"]), records=records)

    def test_validating_or_rejecting_a_switch_is_atomic_and_does_not_consume_it(self):
        records, observation = self.retained()
        original = copy.deepcopy((records, observation))
        context = plan_context(records, observation)
        proposal = plan(["interact", "reset", "interact"], "retarget", "Another question.", [1])
        returned = self.validate(proposal, records=records, identifier="interaction")
        returned["controls"][0] = "finish"
        with self.assertRaises(InvalidDecision):
            self.validate({**proposal, "evidence_steps": [9]}, records=records, identifier="interaction")
        self.assertEqual((records, observation), original)
        self.assertEqual(plan_context(records, observation), context)
        self.assertEqual(context["scope_switches"], 0)

    def test_actual_switch_consumes_one_control_and_preserves_the_reset_tail(self):
        records, observation = executed([
            choice(["east", "interact", "reset", "interact"]),
            choice(["interact", "reset", "interact"], identifier="interaction", target="interact",
                   mode="retarget", reason="Chosen second question.", refs=[1]),
            choice(["reset", "interact"], identifier="interaction", target="interact", mode="continue"),
        ])
        context = plan_context(records, observation)
        self.assertEqual(context["pending_plan"], {"source_step": 3, "experiment_id": "interaction", "controls": ["interact"]})
        self.assertEqual(context["scope_switches"], 1)
        self.assertEqual(context["plan_revisions"], 0)
        self.assertEqual(context["completed_steps"], [1, 2, 3])
        self.assertEqual(research_context(records, observation), context)
        self.assertFalse(context["claims_verified"])

    def test_revisions_and_scope_switches_are_distinct_executed_counts(self):
        records, observation = executed([
            choice(["east", "interact"]),
            choice(["interact"], identifier="interaction", target="interact", mode="retarget", reason="Another question.", refs=[1]),
            choice(["wait"], identifier="pause", target="wait", mode="revise", reason="I changed the controls.", refs=[2]),
        ])
        context = plan_context(records, observation)
        self.assertEqual((context["scope_switches"], context["plan_revisions"]), (1, 1))

    def test_finish_clears_only_pending_intent_and_does_not_invent_an_action(self):
        records, observation = self.retained()
        before, after = plan_context(records, observation), plan_context(records, observation, finished=True)
        self.assertIsNotNone(before["pending_plan"])
        self.assertIsNone(after["pending_plan"])
        self.assertTrue(after["session_finished"])
        for field in ("completed_steps", "scope_switches", "plan_revisions", "current_state_action_attempts"):
            self.assertEqual(before[field], after[field])
        with self.assertRaises(InvalidDecision):
            plan_context(records, observation, finished=1)

    def test_repeated_single_control_plans_have_no_hidden_quota(self):
        records, observation = executed([choice(["interact"], identifier="interaction", target="interact") for _ in range(24)])
        self.assertEqual(self.validate(plan(["interact"]), records=records, identifier="interaction"), plan(["interact"]))
        self.assertEqual(plan_context(records, observation)["current_state_action_attempts"]["interact"], 24)

    def test_attempt_counts_ignore_ticks_feedback_and_private_rules(self):
        records, observation = executed([choice(["wait"], identifier="pause", target="wait")])
        observation.update(tick=999, events=["irrelevant"], rules=["secret-rule"], goal="secret-goal", passability_hints="secret-hint")
        context = plan_context(records, observation)
        self.assertEqual(context["current_state_action_attempts"]["wait"], 1)
        for marker in ("secret-rule", "secret-goal", "secret-hint"):
            self.assertNotIn(marker, json.dumps(context))

    def test_new_schema_has_no_retarget_without_actual_history(self):
        context = plan_context({}, KeyGateGame().observe())
        schema = plan_schema(list(PLAN_CONTROLS), context)
        self.assertEqual([branch["properties"]["mode"]["enum"] for branch in schema["anyOf"]], [["new"]])
        self.assertEqual(schema["required"], ["mode", "controls", "revision_reason", "evidence_steps"])
        self.assertFalse(schema["additionalProperties"])

    def test_schema_switch_and_continue_only_fit_the_actual_pending_head(self):
        records, observation = self.retained()
        context = plan_context(records, observation)
        for action in PLAN_CONTROLS:
            schema = plan_schema(list(PLAN_CONTROLS), context, action=action)
            alternatives = {branch["properties"]["mode"]["enum"][0]: branch for branch in schema["anyOf"]}
            with self.subTest(action=action):
                self.assertIn("revise", alternatives)
                self.assertNotIn("new", alternatives)
                self.assertEqual("continue" in alternatives, action == "interact")
                self.assertEqual("retarget" in alternatives, action == "interact")
                if action == "interact":
                    switch = alternatives["retarget"]["properties"]
                    self.assertEqual(switch["controls"]["enum"], [["interact", "reset", "interact"]])
                    self.assertEqual(switch["revision_reason"]["minLength"], 1)
                    self.assertEqual(switch["evidence_steps"]["minItems"], 1)
                    self.assertEqual(switch["evidence_steps"]["items"]["enum"], [1])

    def test_schema_keeps_all_future_controls_and_finish_only_last(self):
        context = plan_context({}, KeyGateGame().observe())
        schema = plan_schema(["reset", "finish"], context, action="reset")
        arrays = schema["properties"]["controls"]["anyOf"]
        self.assertEqual(arrays[1]["prefixItems"][1]["enum"], list(PLAN_CONTROLS))
        self.assertNotIn("finish", arrays[3]["prefixItems"][2]["enum"])
        self.assertEqual(arrays[3]["prefixItems"][3]["enum"], list(PLAN_CONTROLS))
        finish = plan_schema(["reset", "finish"], context, action="finish")
        self.assertEqual(len(finish["properties"]["controls"]["anyOf"]), 1)

    def test_schema_rejects_wrong_contract_finished_session_and_bad_pending_metadata(self):
        context = plan_context({}, KeyGateGame().observe())
        bad = (None, {}, {**context, "contract_version": 1}, {**context, "contract_version": True},
               {**context, "session_finished": True}, {**context, "completed_steps": [True]},
               {**context, "pending_plan": {"source_step": 1, "experiment_id": "movement", "controls": ["east"]}})
        for value in bad:
            with self.subTest(context=value), self.assertRaises(InvalidDecision):
                plan_schema(list(PLAN_CONTROLS), value)

    def test_recorded_retarget_with_future_evidence_or_silent_tail_change_is_rejected(self):
        records, observation = executed([
            choice(["east", "interact"]),
            choice(["interact"], identifier="interaction", target="interact", mode="retarget", reason="New question.", refs=[1]),
        ])
        for change in (
            lambda value: value[2]["decision"]["plan"].update(evidence_steps=[2]),
            lambda value: value[2]["decision"].update(experiment_id="movement"),
            lambda value: value[2]["decision"]["plan"].update(controls=["interact", "reset"]),
            lambda value: value[2].update(episode=True),
        ):
            wrong = copy.deepcopy(records)
            change(wrong)
            with self.subTest(records=wrong), self.assertRaises(InvalidDecision):
                plan_context(wrong, observation)

    def test_goals_are_first_executed_trimmed_and_stable_for_reused_ids(self):
        records, _ = executed([choice(["east"], goal="  Does this east effect repeat?  ")])
        self.assertEqual(experiment_goals(records), {"movement": "Does this east effect repeat?"})
        self.assertEqual(validate_goal("movement", " Does this east effect repeat? ", records), "Does this east effect repeat?")
        self.assertEqual(validate_goal("interaction", "A new interaction question.", records), "A new interaction question.")
        with self.assertRaisesRegex(InvalidDecision, "original experiment_goal"):
            validate_goal("movement", "The broader session question.", records)

    def test_unexecuted_goal_validation_is_atomic_and_does_not_create_a_binding(self):
        records, _ = self.retained()
        original = copy.deepcopy(records)
        validate_goal("new-scope", "Chosen question.", records)
        self.assertEqual(records, original)
        self.assertNotIn("new-scope", experiment_goals(records))

    def test_history_rejects_silent_question_changes_and_bad_goal_types(self):
        records, _ = executed([choice(["east"]), choice(["east"], goal="Another question.")])
        with self.assertRaisesRegex(InvalidDecision, "original experiment_goal"):
            experiment_goals(records)
        for value in (None, True, "", "x" * 241):
            with self.subTest(goal=value), self.assertRaises(InvalidDecision):
                validate_goal("chosen", value, {})

    def test_assessed_first_trial_needs_its_own_nonuntested_cited_note(self):
        records, _ = executed([choice(["east"]), choice(["interact"], identifier="interaction", target="interact")])
        learned = [note("east-note", "east", [1])]
        self.assertEqual(unnoted_trial_steps(records, {1, 2}, learned), [2])
        self.assertEqual(unnoted_trial_steps(records, {1, 2}, learned + [note("proposal", "interact", [2], status="untested")]), [2])
        self.assertEqual(unnoted_trial_steps(records, {1, 2}, learned + [note("uncited", "interact")]), [2])
        self.assertEqual(unnoted_trial_steps(records, {1, 2}, learned + [note("interaction-note", "interact", [2])]), [])

    def test_setup_citation_and_another_same_control_experiment_do_not_satisfy_a_trial(self):
        records, _ = executed([
            choice(["interact"], identifier="original", target="interact"),
            choice(["interact"], identifier="other", target="interact", role="setup"),
            choice(["interact"], identifier="other", target="interact"),
        ])
        self.assertEqual(unnoted_trial_steps(records, {1, 2, 3}, [note("original-note", "interact", [1])]), [3])
        self.assertEqual(unnoted_trial_steps(records, {1, 2, 3}, [note("setup-note", "interact", [2])]), [1, 3])
        self.assertEqual(unnoted_trial_steps(records, {1, 2, 3}, [note("other-note", "interact", [3])]), [1])

    def test_pending_unassessed_trial_does_not_create_a_learning_requirement(self):
        records, _ = executed([choice(["interact"], identifier="interaction", target="interact")])
        self.assertEqual(unnoted_trial_steps(records, set(), []), [])
        self.assertEqual(unnoted_trial_steps(records, {1}, []), [1])

    def test_reset_citation_never_substitutes_for_game_trial_learning(self):
        records, _ = self.pair()
        self.assertEqual(unnoted_trial_steps(records, {1, 2, 3}, [note("reset-note", "east", [2])]), [1])

    def test_progress_pairs_include_full_conditions_actual_episodes_and_assessment(self):
        records, _ = self.pair()
        progress = scope_progress(records, {1, 2, 3}, [note("east-note", "east", [1])])
        entry = progress["experiments"][0]
        self.assertEqual(entry["trial_steps"], [1, 3])
        self.assertEqual(entry["setup_steps"], [2])
        self.assertEqual(entry["assessed_trial_steps"], [1, 3])
        self.assertEqual(entry["eligible_pairs"], [{"original_step": 1, "repeated_step": 3, "original_episode": 0, "repeated_episode": 1, "result": "same", "control": "east"}])
        self.assertEqual((entry["note_count"], entry["comparison_note_count"]), (1, 0))
        self.assertFalse(progress["claims_verified"])
        partial = scope_progress(records, {1, 2}, [note("east-note", "east", [1])])
        self.assertEqual(partial["experiments"][0]["eligible_pair_count"], 0)

    def test_pair_bookkeeping_counts_cited_pairs_without_verifying_the_claim(self):
        records, _ = self.pair()
        pair = {"original_step": 1, "repeated_step": 3, "original_episode": 0, "repeated_episode": 1, "result": "same"}
        learned = note("east-note", "east", [1, 3], status="supported", comparison=pair, claim="Freeform prose could still be false.")
        progress = scope_progress(records, {1, 3}, [learned])
        self.assertEqual(progress["experiments"][0]["comparison_note_ids"], ["east-note"])
        self.assertFalse(progress["claims_verified"])
        self.assertNotIn(learned["claim"], json.dumps(progress))

    def test_conditions_ignore_ticks_events_and_private_fields_but_include_inventory(self):
        records, _ = self.pair()
        changed = copy.deepcopy(records)
        changed[3]["before"].update(tick=900, events=["Irrelevant"], rules=["secret"], goal="secret-goal")
        self.assertEqual(scope_progress(changed, {1, 3}, [note("n", "east", [1])])["experiments"][0]["eligible_pair_count"], 1)
        changed[3]["before"]["player"]["inventory"] = ["key"]
        self.assertEqual(scope_progress(changed, {1, 3}, [note("n", "east", [1])])["experiments"][0]["eligible_pair_count"], 0)

    def test_same_episode_or_different_experiment_trials_do_not_create_a_retention_pair(self):
        records, _ = executed([choice(["interact"], identifier="a", target="interact"), choice(["interact"], identifier="a", target="interact"), choice(["interact"], identifier="b", target="interact")])
        progress = scope_progress(records, {1, 2, 3}, [])
        self.assertTrue(all(entry["eligible_pair_count"] == 0 for entry in progress["experiments"]))

    def test_retention_pair_outcome_counts_feedback_and_changed_state(self):
        records, _ = self.pair()
        wrong = copy.deepcopy(records)
        wrong[3]["after"]["events"] = ["A different recorded outcome."]
        progress = scope_progress(wrong, {1, 3}, [note("n", "east", [1])])
        self.assertEqual(progress["experiments"][0]["eligible_pairs"][0]["result"], "different")

    def test_family_counts_keep_movement_interaction_and_wait_separate(self):
        records, _ = executed([
            choice(["east"]),
            choice(["interact"], identifier="interaction", target="interact"),
            choice(["wait"], identifier="pause", target="wait"),
        ])
        progress = scope_progress(records, {1, 2, 3}, [note("m", "east", [1]), note("i", "interact", [2])])
        for family in ("movement", "interaction", "wait"):
            self.assertEqual(progress["families"][family]["experiment_count"], 1)
            self.assertEqual(progress["families"][family]["trial_count"], 1)
        self.assertEqual(progress["families"]["movement"]["note_count"], 1)
        self.assertEqual(progress["families"]["interaction"]["note_count"], 1)
        self.assertEqual(progress["families"]["wait"]["note_count"], 0)

    def test_helpers_are_pure_and_accept_id_keyed_notes(self):
        records, observation = self.pair()
        notes = {"east-note": note("east-note", "east", [1])}
        original = copy.deepcopy((records, observation, notes))
        context = plan_context(records, observation)
        before = copy.deepcopy(context)
        plan_schema(list(PLAN_CONTROLS), context)
        scope_progress(records, {1, 3}, notes)
        unnoted_trial_steps(records, {1, 3}, notes)
        self.assertEqual((records, observation, notes), original)
        self.assertEqual(context, before)

    def test_progress_rejects_nonexistent_bool_or_duplicate_assessment_and_note_refs(self):
        records, _ = self.pair()
        for assessed in (None, [True], [4], [1, 1]):
            with self.subTest(assessed=assessed), self.assertRaises(InvalidDecision):
                scope_progress(records, assessed, [])
        for refs in ([True], [4], [1, 1], list(range(1, 10))):
            with self.subTest(refs=refs), self.assertRaises(InvalidDecision):
                scope_progress(records, {1}, [note("n", "east", refs)])

    def test_progress_rejects_inconsistent_declared_controls_and_trial_roles(self):
        records, _ = self.pair()
        for change in (
            lambda value: value[2]["decision"].update(action_role="trial"),
            lambda value: value[3]["decision"].update(experiment_control="interact"),
            lambda value: value[1]["decision"].update(action_role="finish"),
            lambda value: value[1]["decision"].update(experiment_control=True),
        ):
            wrong = copy.deepcopy(records)
            change(wrong)
            with self.subTest(records=wrong), self.assertRaises(InvalidDecision):
                scope_progress(wrong, {1}, [])


if __name__ == "__main__":
    unittest.main()
