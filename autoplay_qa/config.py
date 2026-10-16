"""Small, non-executable .env reader; credentials stay in the local process."""

from dataclasses import dataclass, field
import os
from pathlib import Path
from urllib.parse import urlsplit


DEFAULT_MODEL = "nvidia/Nemotron-3_5-Lightning"
DEFAULT_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
CONFIG_NAMES = {"NEBIUS_API_KEY", "NEBIUS_BASE_URL", "NEBIUS_MODEL", "NEBIUS_RESPONSE_FORMAT", "NEBIUS_ENABLE_THINKING"}


class ConfigurationError(ValueError):
    pass


def read_env_file(path: Path) -> dict[str, str]:
    """Read literal KEY=value lines without evaluating shell code or interpolation."""
    if not path.exists():
        return {}
    values = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        name, separator, value = line.partition("=")
        name = name.strip()
        if not separator:
            raise ConfigurationError(f"Malformed .env line {number}; expected NAME=value.")
        if name not in CONFIG_NAMES:
            continue
        value = value.strip()
        if value.startswith(("'", '"')):
            if len(value) < 2 or value[-1] != value[0]:
                raise ConfigurationError(f"Unclosed quote on .env line {number}.")
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].rstrip()
        values[name] = value
    return values


@dataclass(frozen=True)
class Settings:
    api_key: str = field(repr=False)
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    response_format: str = "json_schema"
    enable_thinking: bool = True

    @classmethod
    def load(cls, env_file: Path = Path(".env")) -> "Settings":
        values = read_env_file(env_file)
        values.update({name: os.environ[name] for name in CONFIG_NAMES if name in os.environ})
        key = values.get("NEBIUS_API_KEY", "").strip()
        if not key or key.lower() in {"your_api_key", "your-api-key", "replace_me"}:
            raise ConfigurationError(
                "NEBIUS_API_KEY is missing. Create a Token Factory key at "
                "https://tokenfactory.nebius.com/ (API keys), copy .env.example to .env, "
                "and set NEBIUS_API_KEY there, or export it in your terminal. "
                "No inference request was sent. Human play and replay do not require a key."
            )
        if any(character.isspace() for character in key):
            raise ConfigurationError("NEBIUS_API_KEY must not contain whitespace.")
        base_url = values.get("NEBIUS_BASE_URL") or DEFAULT_BASE_URL
        parsed = urlsplit(base_url)
        # Limit credential-bearing requests to Nebius. A separate transport is used in tests.
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or not parsed.hostname.endswith(".nebius.com")
            or not parsed.hostname.startswith("api.tokenfactory.")
            or parsed.port not in (None, 443)
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path.rstrip("/") != "/v1"
        ):
            raise ConfigurationError("NEBIUS_BASE_URL must be a Nebius Token Factory HTTPS /v1/ endpoint.")
        model = values.get("NEBIUS_MODEL") or DEFAULT_MODEL
        if not model.lower().startswith("nvidia/") or "nemotron" not in model.lower():
            raise ConfigurationError("NEBIUS_MODEL must be an NVIDIA Nemotron model ID from your Nebius catalog.")
        response_format = values.get("NEBIUS_RESPONSE_FORMAT") or "json_schema"
        if response_format not in {"json_schema", "json_object", "plain"}:
            raise ConfigurationError("NEBIUS_RESPONSE_FORMAT must be json_schema, json_object, or plain.")
        thinking = values.get("NEBIUS_ENABLE_THINKING", "true").lower()
        if thinking not in {"true", "false"}:
            raise ConfigurationError("NEBIUS_ENABLE_THINKING must be true or false.")
        return cls(key, base_url.rstrip("/") + "/", model, response_format, thinking == "true")

