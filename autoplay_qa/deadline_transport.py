"""Opt-in, isolated end-to-end deadlines for validation inference requests.

The legacy client stays unchanged. One private worker performs one request; the
parent owns its monotonic response deadline, rejects late results, and kills/reaps
the worker on timeout or interruption. Cleanup has one additional bounded
one-second window and is measured separately. Socket timeouts alone cannot bound DNS,
connection setup and a continuously trickling response as one operation.

Keys and request data travel only through stdin. Worker stderr is discarded and
stdout is bounded private IPC, never diagnostic output. A timeout cannot cancel
provider billing: callers must keep its pre-request reservation as unknown.
"""

from contextlib import redirect_stderr, redirect_stdout
import copy
import io
import json
import math
import os
from pathlib import Path
import re
import secrets
import signal
import subprocess
import sys
import time

from .config import Settings
from .experience import public_observation
from .nebius import Completion, DEFAULT_MAX_TOKENS, NebiusClient, NebiusError, TransientNebiusError
from .session import redact_secrets


DEADLINE_TRANSPORT_VERSION = "isolated-deadline-v1"
MAX_WORKER_INPUT_BYTES = 8 * 1_048_576
MAX_WORKER_OUTPUT_BYTES = 2 * 1_048_576
_CLEANUP_SECONDS = 1.0
_NONCE = re.compile(r"^[0-9a-f]{32}$")
_TRANSIENT_MESSAGES = {
    "timeout": "Nebius end-to-end request deadline expired; usage is unknown and the request may still have been billed.",
    "connection_reset": "The Nebius connection was reset before a response was received.",
    "rate_limited": "Nebius HTTP 429: Rate limited by the provider.",
    "http_5xx": "Nebius HTTP 5xx: The provider is temporarily unavailable.",
}
_ERROR_MESSAGES = {
    "connection_failed": "Nebius connection failed; no usable response was received.",
    "response_limit": "Nebius response exceeded the configured response size limit.",
    "invalid_response": "Nebius returned an unusable response.",
    "refusal": "Nemotron refused the request; no action was accepted.",
    "worker_failed": "The isolated inference worker failed; no response was accepted.",
    "worker_protocol": "The isolated inference worker returned unusable private IPC; no response was accepted.",
    "worker_start": "The isolated inference worker could not start; no response was accepted.",
}
_SAFE_METADATA = {
    "completion_id", "requested_model", "returned_model", "finish_reason", "latency_ms", "usage",
    "reasoning_tokens", "raw_content_characters", "discarded_inline_thinking",
    "discarded_unusable_content",
}


def _json_bytes(value):
    return json.dumps(value, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")


def _positive_seconds(value, label):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be a finite positive number.")
    return float(value)


class RequestAdmissionDeclined(NebiusError):
    """Proven pre-worker/pre-provider stop after bookkeeping used session time."""

    def __init__(self, remaining_seconds, request_timeout):
        super().__init__("No provider request was dispatched: insufficient session time for a complete request window.")
        self.remaining_seconds = remaining_seconds
        self.request_timeout = request_timeout
        self.required_seconds = request_timeout + 1
        self.no_provider_dispatched = True
        self.possibly_billed = False


def _safe_completion(completion, api_key):
    if not isinstance(completion, Completion) or not isinstance(completion.content, str) or not isinstance(completion.metadata, dict):
        raise ValueError("Invalid worker completion shape.")
    metadata = {key: copy.deepcopy(value) for key, value in completion.metadata.items() if key in _SAFE_METADATA}
    metadata = public_observation({"player": {"metadata": metadata}}, secrets=(api_key,))["player"]["metadata"]
    for key in ("completion_id", "requested_model", "returned_model"):
        value = metadata.get(key)
        if value is not None and (not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9/_.:-]{0,255}", value) is None):
            metadata[key] = None
    if metadata.get("finish_reason") not in {None, "stop", "length", "content_filter", "tool_calls", "function_call"}:
        metadata["finish_reason"] = None
    usage = metadata.get("usage", {})
    metadata["usage"] = {key: value for key, value in usage.items()
                         if key in {"prompt_tokens", "completion_tokens", "total_tokens"}
                         and type(value) is int and 0 <= value <= 2**63 - 1} if isinstance(usage, dict) else {}
    for key in ("latency_ms", "reasoning_tokens", "raw_content_characters"):
        if key in metadata and (type(metadata[key]) is not int or not 0 <= metadata[key] <= 2**63 - 1):
            metadata.pop(key)
    for key in ("discarded_inline_thinking", "discarded_unusable_content"):
        if key in metadata and type(metadata[key]) is not bool:
            metadata.pop(key)
    content = redact_secrets(completion.content, (api_key,))
    # Invalid/truncated text is not executable, and can contain an unfinished
    # model trace. Keep lengths/status, never put that text into diagnostics.
    if metadata.get("finish_reason") != "stop" or not content.lstrip().startswith("{"):
        content = ""
        metadata["discarded_unusable_content"] = True
    return {"content": content, "metadata": metadata}


def _failure_packet(error):
    """Convert only finite categories; never serialize exception messages."""
    category, status, retry_after = "worker_failed", None, None
    if isinstance(error, TransientNebiusError):
        category = error.reason
        retry_after = error.retry_after_seconds
        match = re.match(r"^Nebius HTTP ([1-5][0-9]{2}):", str(error))
        if match and ((category == "rate_limited" and match.group(1) == "429")
                      or (category == "http_5xx" and match.group(1).startswith("5"))):
            status = int(match.group(1))
    elif isinstance(error, NebiusError):
        message = str(error)
        match = re.match(r"^Nebius HTTP ([1-5][0-9]{2}):", message)
        if match:
            category, status = "http_error", int(match.group(1))
        elif message.startswith("Nebius connection failed"):
            category = "connection_failed"
        elif message.startswith("Nebius response exceeded"):
            category = "response_limit"
        elif message.startswith("Nemotron refused"):
            category = "refusal"
        else:
            category = "invalid_response"
    return {"category": category, "http_status": status, "retry_after_seconds": retry_after}


def _worker_result(payload, *, client_factory=NebiusClient, clock=time.monotonic):
    """One request, injectable only by offline unit tests (not an IPC option)."""
    permitted = {"version", "nonce", "settings", "client", "request", "deadline"}
    if not isinstance(payload, dict) or set(payload) != permitted or payload.get("version") != DEADLINE_TRANSPORT_VERSION:
        raise ValueError("Invalid private worker input.")
    if not isinstance(payload.get("nonce"), str) or not _NONCE.fullmatch(payload["nonce"]):
        raise ValueError("Invalid worker nonce.")
    deadline = payload["deadline"]
    if type(deadline) not in (int, float) or not math.isfinite(deadline):
        raise ValueError("Invalid worker deadline.")
    settings, config, request = payload["settings"], payload["client"], payload["request"]
    if not isinstance(settings, dict) or set(settings) != {"api_key", "base_url", "model", "response_format", "enable_thinking"}:
        raise ValueError("Invalid private worker settings.")
    if not isinstance(config, dict) or set(config) != {"timeout", "max_tokens", "temperature", "top_p"}:
        raise ValueError("Invalid private worker configuration.")
    if not isinstance(request, dict) or set(request) != {"messages", "response_schema", "enable_thinking", "max_tokens"}:
        raise ValueError("Invalid private worker request.")
    remaining = deadline - clock()
    if remaining <= 0:
        return {"status": "error", "error": {"category": "timeout", "http_status": None, "retry_after_seconds": None}}
    try:
        client = client_factory(Settings(**settings), **config)
        remaining = deadline - clock()
        if remaining <= 0:
            raise TransientNebiusError(_TRANSIENT_MESSAGES["timeout"], reason="timeout")
        completion = client.complete(request["messages"], timeout=min(config["timeout"], remaining),
            response_schema=request["response_schema"], enable_thinking=request["enable_thinking"], max_tokens=request["max_tokens"])
        if clock() >= deadline:
            raise TransientNebiusError(_TRANSIENT_MESSAGES["timeout"], reason="timeout")
        return {"status": "completed", "completion": _safe_completion(completion, settings["api_key"])}
    except Exception as error:
        return {"status": "error", "error": _failure_packet(error)}


def _worker_main(source=None, target=None, *, client_factory=NebiusClient, clock=time.monotonic):
    source = sys.stdin.buffer if source is None else source
    target = sys.stdout.buffer if target is None else target
    nonce = "0" * 32
    try:
        raw = source.read(MAX_WORKER_INPUT_BYTES + 1)
        if len(raw) > MAX_WORKER_INPUT_BYTES:
            raise ValueError("Private worker input exceeded its bound.")
        payload = json.loads(raw)
        candidate = payload.get("nonce") if isinstance(payload, dict) else None
        if isinstance(candidate, str) and _NONCE.fullmatch(candidate):
            nonce = candidate
        # No dependency print or provider error may contaminate private IPC or
        # escape onto the terminal. Only the finite envelope below is written.
        with open(os.devnull, "w") as sink, redirect_stdout(sink), redirect_stderr(sink):
            result = _worker_result(payload, client_factory=client_factory, clock=clock)
    except Exception:
        result = {"status": "error", "error": {"category": "worker_failed", "http_status": None, "retry_after_seconds": None}}
    envelope = {"version": DEADLINE_TRANSPORT_VERSION, "nonce": nonce, **result}
    encoded = _json_bytes(envelope)
    if len(encoded) > MAX_WORKER_OUTPUT_BYTES:
        encoded = _json_bytes({"version": DEADLINE_TRANSPORT_VERSION, "nonce": nonce, "status": "error",
            "error": {"category": "response_limit", "http_status": None, "retry_after_seconds": None}})
    target.write(encoded)
    target.flush()
    return 0


class DeadlineNebiusClient(NebiusClient):
    """Nebius completion client with one private, killable worker per request."""

    def __init__(self, settings, timeout=45, max_tokens=DEFAULT_MAX_TOKENS, *, temperature=0.2, top_p=None):
        _positive_seconds(timeout, "timeout")
        super().__init__(settings, timeout=timeout, max_tokens=max_tokens, temperature=temperature, top_p=top_p)
        self._session_deadline = None

    def set_session_deadline(self, deadline):
        if deadline is not None and (type(deadline) not in (int, float) or not math.isfinite(deadline)):
            raise ValueError("Session deadline must be a finite monotonic number or None.")
        self._session_deadline = None if deadline is None else float(deadline)

    def _spawn_worker(self):
        environment = os.environ.copy()
        # The validated settings in private stdin are the only worker source of
        # provider credentials, even if the caller originally loaded an env key.
        for name in ("NEBIUS_API_KEY", "NEBIUS_BASE_URL", "NEBIUS_MODEL", "NEBIUS_RESPONSE_FORMAT", "NEBIUS_ENABLE_THINKING"):
            environment.pop(name, None)
        return subprocess.Popen([sys.executable, "-m", "autoplay_qa.deadline_transport", "--worker"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            cwd=str(Path(__file__).resolve().parents[1]), env=environment,
            start_new_session=(os.name == "posix"))

    @staticmethod
    def _kill_and_reap(process):
        if process is None:
            return True
        cleanup_deadline = time.monotonic() + _CLEANUP_SECONDS
        try:
            if process.poll() is None:
                if os.name == "posix":
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        process.kill()
                else:
                    process.kill()
        except (OSError, ProcessLookupError):
            pass
        try:
            process.communicate(timeout=max(0.0, cleanup_deadline - time.monotonic()))
        except (subprocess.TimeoutExpired, OSError, ValueError):
            # Descendants must not keep stdout open indefinitely. The worker is
            # killed first; close our pipes and reap that direct child.
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass
            try:
                process.kill()
                process.wait(timeout=max(0.0, cleanup_deadline - time.monotonic()))
            except (OSError, subprocess.TimeoutExpired):
                pass
        return process.poll() is not None

    @staticmethod
    def _raise_failure(packet):
        if not isinstance(packet, dict) or set(packet) != {"category", "http_status", "retry_after_seconds"}:
            raise NebiusError(_ERROR_MESSAGES["worker_protocol"])
        category, status, retry = packet["category"], packet["http_status"], packet["retry_after_seconds"]
        if status is not None and (type(status) is not int or not 100 <= status <= 599):
            raise NebiusError(_ERROR_MESSAGES["worker_protocol"])
        if retry is not None and (type(retry) not in (int, float) or not math.isfinite(retry) or retry < 0):
            raise NebiusError(_ERROR_MESSAGES["worker_protocol"])
        if category in _TRANSIENT_MESSAGES:
            if status is not None and not ((category == "rate_limited" and status == 429) or (category == "http_5xx" and 500 <= status <= 599)):
                raise NebiusError(_ERROR_MESSAGES["worker_protocol"])
            message = f"Nebius HTTP {status}: The provider is temporarily unavailable." if category == "http_5xx" and status is not None else _TRANSIENT_MESSAGES[category]
            error = TransientNebiusError(message, reason=category, retry_after_seconds=retry)
            if status is not None:
                error.http_status = status
            raise error
        if category == "http_error" and status is not None and retry is None:
            error = NebiusError(f"Nebius HTTP {status}: Provider rejected the request; no response was accepted.")
            error.http_status = status
            raise error
        if category not in _ERROR_MESSAGES or status is not None or retry is not None:
            raise NebiusError(_ERROR_MESSAGES["worker_protocol"])
        raise NebiusError(_ERROR_MESSAGES[category])

    def complete(self, messages, timeout=None, *, response_schema=None, enable_thinking=None, max_tokens=None):
        started = time.monotonic()
        requested = _positive_seconds(self.timeout if timeout is None else timeout, "timeout")
        if max_tokens is not None and max_tokens <= 0:
            raise ValueError("max_tokens must be positive.")
        if self._session_deadline is not None:
            remaining = self._session_deadline - started
            if remaining < requested + 1:
                raise RequestAdmissionDeclined(remaining, requested)
        deadline = min(started + requested, self._session_deadline) if self._session_deadline is not None else started + requested
        nonce = secrets.token_hex(16)
        packet = {"version": DEADLINE_TRANSPORT_VERSION, "nonce": nonce, "deadline": deadline,
            "settings": {"api_key": self.settings.api_key, "base_url": self.settings.base_url, "model": self.settings.model,
                         "response_format": self.settings.response_format, "enable_thinking": self.settings.enable_thinking},
            "client": {"timeout": requested, "max_tokens": self.max_tokens, "temperature": self.temperature, "top_p": self.top_p},
            "request": {"messages": messages, "response_schema": response_schema, "enable_thinking": enable_thinking, "max_tokens": max_tokens}}
        try:
            encoded = _json_bytes(packet)
        except (TypeError, ValueError, OverflowError, RecursionError):
            raise ValueError("Deadline request is not finite JSON data.") from None
        if len(encoded) > MAX_WORKER_INPUT_BYTES:
            raise ValueError("Deadline request exceeded its private IPC input bound.")
        process = None
        result_metadata = None
        reported_error = None
        try:
            if time.monotonic() >= deadline:
                raise TransientNebiusError(_TRANSIENT_MESSAGES["timeout"], reason="timeout")
            process = self._spawn_worker()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TransientNebiusError(_TRANSIENT_MESSAGES["timeout"], reason="timeout")
            output, _stderr = process.communicate(input=encoded, timeout=remaining)
            if time.monotonic() >= deadline:
                raise TransientNebiusError(_TRANSIENT_MESSAGES["timeout"], reason="timeout")
            if process.returncode != 0:
                raise NebiusError(_ERROR_MESSAGES["worker_failed"])
            if not isinstance(output, bytes) or len(output) > MAX_WORKER_OUTPUT_BYTES:
                raise NebiusError(_ERROR_MESSAGES["worker_protocol"])
            try:
                result = json.loads(output)
            except (ValueError, UnicodeDecodeError, RecursionError):
                raise NebiusError(_ERROR_MESSAGES["worker_protocol"]) from None
            if (not isinstance(result, dict) or result.get("version") != DEADLINE_TRANSPORT_VERSION or result.get("nonce") != nonce
                    or result.get("status") not in {"completed", "error"}):
                raise NebiusError(_ERROR_MESSAGES["worker_protocol"])
            if result["status"] == "error":
                if set(result) != {"version", "nonce", "status", "error"}:
                    raise NebiusError(_ERROR_MESSAGES["worker_protocol"])
                self._raise_failure(result["error"])
            if set(result) != {"version", "nonce", "status", "completion"}:
                raise NebiusError(_ERROR_MESSAGES["worker_protocol"])
            completion = result["completion"]
            if not isinstance(completion, dict) or set(completion) != {"content", "metadata"}:
                raise NebiusError(_ERROR_MESSAGES["worker_protocol"])
            safe = _safe_completion(Completion(completion["content"], completion["metadata"]), self.settings.api_key)
            response_finished = time.monotonic()
            if response_finished >= deadline:
                raise TransientNebiusError(_TRANSIENT_MESSAGES["timeout"], reason="timeout")
            elapsed = round((response_finished - started) * 1000)
            metadata = {**safe["metadata"], "transport": DEADLINE_TRANSPORT_VERSION,
                        "request_timeout_seconds": requested, "transport_elapsed_ms": elapsed, "latency_ms": elapsed}
            result_metadata = metadata
            return Completion(safe["content"], metadata)
        except subprocess.TimeoutExpired:
            error = TransientNebiusError(_TRANSIENT_MESSAGES["timeout"], reason="timeout")
            error.transport = DEADLINE_TRANSPORT_VERSION
            error.request_timeout_seconds = requested
            error.transport_elapsed_ms = round((time.monotonic() - started) * 1000)
            reported_error = error
            raise error from None
        except NebiusError as error:
            error.transport = DEADLINE_TRANSPORT_VERSION
            error.request_timeout_seconds = requested
            error.transport_elapsed_ms = round((time.monotonic() - started) * 1000)
            if not isinstance(error, TransientNebiusError):
                error.possibly_billed = process is not None
            reported_error = error
            raise
        except (OSError, ValueError, TypeError, KeyError, RecursionError):
            error = NebiusError(_ERROR_MESSAGES["worker_start"] if process is None else _ERROR_MESSAGES["worker_protocol"])
            error.transport = DEADLINE_TRANSPORT_VERSION
            error.request_timeout_seconds = requested
            error.transport_elapsed_ms = round((time.monotonic() - started) * 1000)
            error.possibly_billed = process is not None
            reported_error = error
            raise error from None
        finally:
            cleanup_started = time.monotonic()
            reaped = self._kill_and_reap(process)
            finished = time.monotonic()
            cleanup_ms = round((finished - cleanup_started) * 1000)
            if result_metadata is not None:
                result_metadata.update(cleanup_elapsed_ms=cleanup_ms, cleanup_worker_reaped=reaped,
                                       latency_ms=round((finished - started) * 1000))
            if reported_error is not None:
                reported_error.cleanup_elapsed_ms = cleanup_ms
                reported_error.cleanup_worker_reaped = reaped


if __name__ == "__main__":
    if sys.argv[1:] != ["--worker"]:
        raise SystemExit(2)
    raise SystemExit(_worker_main())
