"""Observe -> infer -> validate -> act -> evaluate -> observe, with bounded memory."""

from dataclasses import asdict, dataclass
import json
import time
from typing import Callable

from .game import GameAdapter, evaluate_transition
from .nebius import InvalidDecision, Decision, NebiusClient, NebiusError, decision_messages
from .session import SessionLog, redact_secrets


@dataclass(frozen=True)
class RunLimits:
    max_steps: int = 24
    max_requests: int = 28
    max_seconds: float = 300
    history_size: int = 6

    def __post_init__(self):
        if min(self.max_steps, self.max_requests, self.max_seconds, self.history_size) <= 0:
            raise ValueError("All run limits must be positive.")


def run_agent(
    game: GameAdapter,
    client: NebiusClient,
    log: SessionLog,
    limits: RunLimits = RunLimits(),
    on_step: Callable[[dict], None] | None = None,
    evaluator: Callable[[dict, str, dict], dict] = evaluate_transition,
) -> dict:
    started = time.monotonic()
    deadline = started + limits.max_seconds
    steps = requests = rejected = consecutive_invalid = 0
    plan = feedback = ""
    history = []
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    status = "step_limit"
    error = None
    try:
        while steps < limits.max_steps:
            before = game.observe()
            if before["status"] == "won":
                status = "won"
                break
            if requests >= limits.max_requests:
                status = "request_limit"
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                status = "time_limit"
                break
            messages = decision_messages(before, plan, history, feedback)
            requests += 1
            log.write("request", request_number=requests, observation_tick=before["tick"], messages=messages)
            completion = client.complete(messages, timeout=min(client.timeout, remaining))
            for name in totals:
                totals[name] += completion.metadata["usage"].get(name, 0)
            # Some endpoints place an undelimited thinking trace in content.
            # Persist only complete validated action JSON, never that free text.
            safe_answer = None
            if completion.metadata["finish_reason"] == "stop":
                try:
                    validated = Decision.parse(completion.content, before["actions"])
                    safe_answer = json.dumps(redact_secrets(asdict(validated), log.secrets))
                except InvalidDecision:
                    pass
            log.write("completion", request_number=requests, answer=safe_answer, answer_characters=len(completion.content), **completion.metadata)
            if time.monotonic() >= deadline:
                status = "time_limit"
                break
            try:
                if completion.metadata["finish_reason"] != "stop":
                    raise InvalidDecision("Completion was truncated or interrupted. Increase --max-tokens if finish_reason is length; no action was taken.")
                decision = Decision.parse(completion.content, before["actions"])
            except InvalidDecision as exc:
                rejected += 1
                consecutive_invalid += 1
                feedback = str(exc)
                log.write("rejected_decision", request_number=requests, error=feedback)
                if consecutive_invalid >= 2:
                    status = "invalid_decision"
                    error = feedback
                    break
                # One bounded repair request, using the unchanged actual observation.
                continue
            consecutive_invalid = 0
            feedback = ""
            after = game.act(decision.action)
            evaluation = evaluator(before, decision.action, after)
            steps += 1
            record = {"step": steps, "request_number": requests, "before": before, "decision": asdict(decision), "after": after, "evaluation": evaluation}
            log.write("step", **record)
            if on_step:
                on_step(record)
            plan = decision.plan
            history.append({"tick": after["tick"], "action": decision.action, "position": after["player"]["position"], "inventory": after["player"]["inventory"], "evaluation": evaluation})
            history = history[-limits.history_size:]
            if evaluation["contract_violations"]:
                status = "contract_failure"
                break
            if after["status"] == "won":
                status = "won"
                break
    except NebiusError as exc:
        status, error = "inference_error", str(exc)
    except KeyboardInterrupt:
        status = "interrupted"
    summary = {
        "status": status,
        "steps": steps,
        "inference_requests": requests,
        "rejected_decisions": rejected,
        "usage": totals,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "final_observation": game.observe(),
    }
    if error:
        summary["error"] = error
    log.write("summary", **summary)
    return summary
