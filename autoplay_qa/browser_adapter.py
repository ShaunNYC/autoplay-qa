"""Generic local visible-UI adapter with isolated, fresh browser contexts.

Only public rendered text/controls are observed. The bridge has no game rules,
provider credentials, hidden-state reads, action retries or supplied routes.
"""
from __future__ import annotations

from copy import deepcopy
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from math import isfinite
import os
from pathlib import Path
import queue
import subprocess
import threading
from urllib.parse import urlsplit
from uuid import uuid4

from .protocol import ProtocolError, read_message, write_message

BROWSER_PROTOCOL = "autoplay-browser/1"
ROOT = Path(__file__).resolve().parents[1]


def validate_action(action: dict) -> dict:
    fields = {
        "click": {"kind", "target"}, "input": {"kind", "target", "value"},
        "select": {"kind", "target", "value"}, "wait": {"kind", "milliseconds"},
    }
    if not isinstance(action, dict) or not isinstance(action.get("kind"), str) or set(action) != fields.get(action["kind"]):
        raise ValueError("Invalid atomic browser control fields.")
    if action["kind"] == "wait":
        value = action["milliseconds"]
        if type(value) is not int or not 0 <= value <= 10_000:
            raise ValueError("Wait must be an integer from 0 through 10000 milliseconds.")
    elif not isinstance(action["target"], str) or not 1 <= len(action["target"]) <= 96:
        raise ValueError("A visible control ID is required.")
    if action["kind"] in {"input", "select"} and (not isinstance(action["value"], str) or len(action["value"]) > 4000):
        raise ValueError("Input must contain at most 4000 characters.")
    return deepcopy(action)


class BrowserAdapter:
    """A fresh local browser. Every acknowledged control is executed once."""
    def __init__(self, url: str, *, node: str = "node", playwright_module: str | Path | None = None,
                 executable_path: str | Path | None = None, timeout: float = 20,
                 source_manifest: dict | None = None):
        parts = urlsplit(url)
        if parts.scheme != "http" or parts.hostname not in {"localhost", "127.0.0.1", "::1"} or parts.username or parts.password:
            raise ValueError("Only a loopback HTTP origin without credentials is permitted.")
        if type(timeout) not in {int, float} or not isfinite(timeout) or timeout <= 0:
            raise ValueError("Browser timeout must be positive.")
        command = [node, str(ROOT / "tools/browser_bridge.mjs"), "--url", url]
        if playwright_module is not None:
            command += ["--playwright-module", str(Path(playwright_module).resolve())]
        if executable_path is not None:
            command += ["--executable", str(Path(executable_path).resolve())]
        env = {key: os.environ[key] for key in ("PATH", "TMPDIR", "SystemRoot", "HOME") if key in os.environ}
        self._process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env)
        self._closed = False
        self._lock = threading.Lock()
        self._responses: queue.Queue = queue.Queue()
        self._next_id = 1
        self.timeout = float(timeout)
        self._reader = threading.Thread(target=self._read_responses, args=(self._process.stdout, self._responses), daemon=True)
        self._reader.start()
        try:
            hello = self._exchange("hello")
            if hello.get("protocol") != BROWSER_PROTOCOL or not isinstance(hello.get("manifest"), dict):
                raise ProtocolError("invalid_response", "Browser protocol identity is invalid.")
            self.source_manifest = deepcopy(source_manifest or {})
            self.source_manifest["browser_transport"] = deepcopy(hello["manifest"])
            self.source_manifest["instance_id"] = str(uuid4())
            self.source_manifest["backend"] = "native-playwright-browser"
            self._observation = self._observation_from(self._exchange("observe"))
        except BaseException:
            self._close_transport()
            raise

    @staticmethod
    def _read_responses(stream, responses):
        try:
            while True:
                response = read_message(stream)
                responses.put(response)
                if response is None:
                    return
        except BaseException as error:
            responses.put(error)

    def _exchange(self, request_type: str, **fields) -> dict:
        with self._lock:
            if self._closed:
                raise ProtocolError("closed", "Browser adapter is closed.")
            request_id = str(self._next_id)
            self._next_id += 1
            try:
                write_message(self._process.stdin, {"id": request_id, "type": request_type, **fields})
                response = self._responses.get(timeout=self.timeout)
            except queue.Empty:
                self._close_transport()
                raise ProtocolError("timeout", "Browser operation timed out; the transport was stopped without retry.") from None
            except (OSError, BrokenPipeError):
                self._close_transport()
                raise ProtocolError("connection_closed", "Browser stopped before acknowledging the operation.") from None
            if isinstance(response, BaseException):
                self._close_transport()
                raise response
            if response is None:
                self._close_transport()
                raise ProtocolError("connection_closed", "Local browser could not start or has stopped.")
            if response.get("id") != request_id:
                self._close_transport()
                raise ProtocolError("invalid_response", "Browser response ID does not match.")
            if response.get("ok") is not True:
                error = response.get("error", {})
                # A browser failure may have applied input before failing. Close to
                # ensure callers cannot turn an ambiguous outcome into a retry.
                self._close_transport()
                raise ProtocolError(str(error.get("code", "browser_error")), str(error.get("message", "Browser operation failed.")))
            if response.get("type") != request_type:
                self._close_transport()
                raise ProtocolError("invalid_response", "Browser response type does not match.")
            return response

    @staticmethod
    def _observation_from(response: dict) -> dict:
        observation = response.get("observation")
        if not isinstance(observation, dict) or observation.get("kind") != "browser-visible-v1" or not isinstance(observation.get("controls"), list) or not isinstance(observation.get("text"), str) or type(observation.get("tick")) is not int:
            raise ProtocolError("invalid_response", "Browser returned an invalid visible observation.")
        return deepcopy(observation)

    def observe(self) -> dict:
        self._observation = self._observation_from(self._exchange("observe"))
        return deepcopy(self._observation)

    def act(self, action: dict) -> dict:
        action = validate_action(action)
        self._observation = self._observation_from(self._exchange("act", action=action))
        return deepcopy(self._observation)

    def _close_transport(self):
        self._closed = True
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=2)
        for stream in (self._process.stdin, self._process.stdout):
            if stream is not None:
                stream.close()
        if self._reader is not threading.current_thread():
            self._reader.join(timeout=1)

    def close(self):
        if not self._closed:
            try:
                self._exchange("close")
            finally:
                self._close_transport()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()

    def __del__(self):
        if hasattr(self, "_closed") and not self._closed:
            self._close_transport()


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *_args):
        pass


class LocalBrowserSite:
    """Serve one already built isolated fixture, never the original workspace."""
    def __init__(self, directory: str | Path):
        self.directory = Path(directory).resolve()
        if not (self.directory / "index.html").is_file():
            raise ValueError("The local browser fixture needs an index.html.")
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), partial(_QuietHandler, directory=str(self.directory)))
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self.url = f"http://127.0.0.1:{self._server.server_port}"

    def close(self):
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def prepare_mars_browser(project_dir: str | Path, output_dir: str | Path, *, node: str = "node", timeout: float = 120) -> dict:
    """Fresh source copy/build; no credentials, source modifications or inference."""
    env = {key: os.environ[key] for key in ("PATH", "TMPDIR", "SystemRoot", "HOME") if key in os.environ}
    result = subprocess.run([node, str(ROOT / "tools/prepare_mars_browser.mjs"), "--project", str(Path(project_dir).resolve()), "--output", str(Path(output_dir).resolve())], capture_output=True, text=True, env=env, timeout=timeout, check=False)
    if result.returncode:
        raise RuntimeError("Isolated Mars source preparation failed.")
    return json.loads(result.stdout)
