"""YAML / env mapping for the LangLangBot Hermes platform plugin."""

from __future__ import annotations

import logging
import os
from importlib import metadata
from pathlib import Path
from typing import Any

logger = logging.getLogger("langlangbot.hermes.config")


PLATFORM_NAME = "langlangbot"
AGENT_RUNTIME_NAME = "Hermes"
SIDECAR_URL_ENV = "LANGLANGBOT_SIDECAR_URL"
PLUGIN_TOKEN_ENV = "LANGLANGBOT_PLUGIN_TOKEN"
DEFAULT_SIDECAR_URL = "https://127.0.0.1:9528"

_CACHED_ADAPTER_VERSION: str | None = None
_CACHED_HOST_VERSION: str | None = None
_HOST_VERSION_RESOLVED = False


def _strip(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def hermes_home() -> Path:
    raw = os.getenv("HERMES_HOME", "").strip()
    if raw:
        return Path(raw).expanduser()
    return Path.home() / ".hermes"


def hermes_config_path() -> Path:
    override = os.getenv("HERMES_CONFIG_PATH", "").strip()
    if override:
        return Path(override).expanduser()
    return hermes_home() / "config.yaml"


def load_yaml_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        import yaml  # type: ignore
    except ImportError:
        return {}
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as err:
        logger.warning("failed to parse %s: %s", path, err)
        return {}
    return loaded if isinstance(loaded, dict) else {}


def adapter_version() -> str:
    global _CACHED_ADAPTER_VERSION
    if _CACHED_ADAPTER_VERSION is None:
        try:
            _CACHED_ADAPTER_VERSION = metadata.version("langlangbot-hermes")
        except metadata.PackageNotFoundError:
            _CACHED_ADAPTER_VERSION = "0.1.0"
    return _CACHED_ADAPTER_VERSION


def host_version() -> str | None:
    global _CACHED_HOST_VERSION, _HOST_VERSION_RESOLVED
    if _HOST_VERSION_RESOLVED:
        return _CACHED_HOST_VERSION
    for dist_name in ("hermes-agent", "hermes", "nous-hermes"):
        try:
            _CACHED_HOST_VERSION = metadata.version(dist_name)
            _HOST_VERSION_RESOLVED = True
            return _CACHED_HOST_VERSION
        except metadata.PackageNotFoundError:
            continue
    env = os.getenv("HERMES_VERSION", "").strip()
    _CACHED_HOST_VERSION = env or None
    _HOST_VERSION_RESOLVED = True
    return _CACHED_HOST_VERSION


def load_hermes_config() -> dict[str, Any]:
    return load_yaml_file(hermes_config_path())


def platform_yaml_section(yaml_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = yaml_cfg if yaml_cfg is not None else load_hermes_config()
    gateway = cfg.get("gateway")
    if not isinstance(gateway, dict):
        gateway = {}
    platforms = gateway.get("platforms")
    if isinstance(platforms, dict) and isinstance(platforms.get(PLATFORM_NAME), dict):
        return platforms[PLATFORM_NAME]
    if isinstance(gateway.get(PLATFORM_NAME), dict):
        return gateway[PLATFORM_NAME]
    top = cfg.get(PLATFORM_NAME)
    return top if isinstance(top, dict) else {}


def _sidecar_from_platform_cfg(platform_cfg: dict[str, Any]) -> tuple[str, str]:
    extra = platform_cfg.get("extra")
    extra = extra if isinstance(extra, dict) else {}
    sidecar = _strip(platform_cfg.get("sidecar_url")) or _strip(extra.get("sidecar_url"))
    token = _strip(platform_cfg.get("plugin_token")) or _strip(extra.get("plugin_token"))
    return sidecar, token


def configured_sidecar_url(platform_cfg: dict[str, Any] | None = None) -> str:
    env = _strip(os.getenv(SIDECAR_URL_ENV))
    if env:
        return env
    section = platform_cfg if platform_cfg is not None else platform_yaml_section()
    sidecar, _token = _sidecar_from_platform_cfg(section)
    return sidecar


def configured_plugin_token(platform_cfg: dict[str, Any] | None = None) -> str:
    env = _strip(os.getenv(PLUGIN_TOKEN_ENV))
    if env:
        return env
    section = platform_cfg if platform_cfg is not None else platform_yaml_section()
    _sidecar, token = _sidecar_from_platform_cfg(section)
    return token


def check_requirements() -> bool:
    """Passive probe: sidecar URL must be present via env or Hermes YAML."""
    return bool(configured_sidecar_url())


def validate_config(config: Any) -> bool:
    extra = getattr(config, "extra", None) or {}
    if not isinstance(extra, dict):
        extra = {}
    return bool(
        _strip(extra.get("sidecar_url"))
        or configured_sidecar_url()
    )


def env_enablement() -> dict[str, Any] | None:
    sidecar_url = _strip(os.getenv(SIDECAR_URL_ENV))
    if not sidecar_url:
        return None
    seed: dict[str, Any] = {"sidecar_url": sidecar_url}
    token = _strip(os.getenv(PLUGIN_TOKEN_ENV))
    if token:
        seed["plugin_token"] = token
    return seed


def apply_yaml_config(
    yaml_cfg: dict[str, Any],
    platform_cfg: dict[str, Any],
) -> dict[str, Any] | None:
    """Map YAML sidecar_url / plugin_token into extras (env wins)."""
    section = platform_cfg if isinstance(platform_cfg, dict) else {}
    if not section and isinstance(yaml_cfg, dict):
        section = platform_yaml_section(yaml_cfg)
    sidecar, token = _sidecar_from_platform_cfg(section)
    extras: dict[str, Any] = {}
    if sidecar and not _strip(os.getenv(SIDECAR_URL_ENV)):
        extras["sidecar_url"] = sidecar
    if token and not _strip(os.getenv(PLUGIN_TOKEN_ENV)):
        extras["plugin_token"] = token
    return extras or None


def resolve_sidecar_settings(config: Any) -> tuple[str, str | None]:
    extra = getattr(config, "extra", None) or {}
    if not isinstance(extra, dict):
        extra = {}
    sidecar = (
        _strip(extra.get("sidecar_url"))
        or configured_sidecar_url()
        or DEFAULT_SIDECAR_URL
    )
    token = _strip(extra.get("plugin_token")) or configured_plugin_token()
    return sidecar, token or None
