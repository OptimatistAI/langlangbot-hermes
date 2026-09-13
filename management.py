"""In-process Hermes status / models / session-scoped set_model."""

from __future__ import annotations

from typing import Any

try:
    from .config import _strip, load_hermes_config
except ImportError:
    from config import _strip, load_hermes_config


class ManagementError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _clean(value: Any) -> str | None:
    stripped = _strip(value)
    return stripped or None


def default_model_from_config(cfg: dict[str, Any] | None = None) -> tuple[str | None, str | None]:
    data = cfg if cfg is not None else load_hermes_config()
    model = data.get("model")
    if not isinstance(model, dict):
        model = {}
    provider = _clean(model.get("provider"))
    default = _clean(model.get("default")) or _clean(model.get("model"))
    return default, provider


def _explicit_provider_ids() -> set[str]:
    """Providers the operator explicitly configured in Hermes-owned storage."""
    from hermes_cli.auth import PROVIDER_REGISTRY, get_auth_status
    from hermes_cli.config import load_config, load_env

    cfg = load_config()
    env = load_env()
    configured: set[str] = set()

    model_cfg = cfg.get("model")
    if isinstance(model_cfg, dict):
        current = _clean(model_cfg.get("provider"))
        if current:
            configured.add(current)

    user_providers = cfg.get("providers")
    if isinstance(user_providers, dict):
        configured.update(
            str(provider).strip()
            for provider in user_providers
            if str(provider).strip()
        )

    for provider_id, provider in PROVIDER_REGISTRY.items():
        if any(_clean(env.get(key)) for key in provider.api_key_env_vars):
            configured.add(provider_id)
            continue
        if provider.auth_type != "api_key":
            try:
                if get_auth_status(provider_id).get("logged_in"):
                    configured.add(provider_id)
            except Exception:
                pass
    return configured


def list_hermes_models() -> dict[str, Any]:
    try:
        from dataclasses import replace

        from hermes_cli.auth import PROVIDER_REGISTRY
        from hermes_cli.inventory import (
            build_model_options_payload,
            load_picker_context,
        )

        explicit_providers = _explicit_provider_ids()
        picker_context = load_picker_context()
        excluded = set(picker_context.excluded_providers or [])
        excluded.update(
            provider
            for provider in PROVIDER_REGISTRY
            if provider not in explicit_providers
        )
        catalog = build_model_options_payload(
            replace(picker_context, excluded_providers=sorted(excluded)),
            explicit_only=False,
            include_unconfigured=False,
        )
    except Exception as err:
        catalog_error = err
    else:
        catalog_error = None

    models: list[dict[str, Any]] = []
    for provider_row in catalog.get("providers", []) if catalog_error is None else []:
        if not isinstance(provider_row, dict):
            continue
        provider = _clean(provider_row.get("slug"))
        if not provider:
            continue
        if provider not in explicit_providers:
            continue
        provider_name = _clean(provider_row.get("name")) or provider
        for raw_model in provider_row.get("models") or []:
            model = _clean(raw_model)
            if not model:
                continue
            models.append(
                {
                    "id": f"{provider}:{model}",
                    "name": model,
                    "provider": provider,
                    "provider_name": provider_name,
                }
            )
    default_model, default_provider = default_model_from_config()
    if not models:
        if default_model:
            models.append(
                {
                    "id": default_model,
                    "name": default_model,
                    "provider": default_provider or "",
                }
            )
        elif catalog_error:
            raise ManagementError(
                "runtime_unavailable",
                f"Hermes model catalog is unavailable: {catalog_error}",
            )
        else:
            raise ManagementError(
                "runtime_unavailable",
                "Hermes model catalog is empty (no provider configured)",
            )
    return {
        "models": models,
        "default_model": default_model,
        "default_provider": default_provider,
    }


def parse_model_selector(model_id: str) -> tuple[str | None, str]:
    requested = model_id.strip()
    if not requested:
        raise ManagementError("invalid_request", "model is required for set_model")
    # custom:<provider>:<model> is Hermes' documented custom-provider syntax.
    # Built-in provider:model must be split; otherwise Hermes treats the whole
    # string as a model alias on the current provider.
    if requested.startswith("custom:"):
        return None, requested
    provider, separator, model = requested.partition(":")
    if separator and provider and model:
        return provider, model
    return None, requested


def session_model_command(model_id: str) -> str:
    provider, model = parse_model_selector(model_id)
    if provider:
        return f"/model --session --provider {provider} {model}"
    return f"/model --session {model}"


def model_selector_provider(model_id: str) -> str | None:
    try:
        provider, _model = parse_model_selector(model_id)
    except ManagementError:
        return None
    return provider


def build_status(
    *,
    conversation_id: str,
    session_key: str | None,
    session_model: str | None,
    session_provider: str | None,
    host_version: str | None,
    adapter_version: str,
) -> dict[str, Any]:
    default_model, default_provider = default_model_from_config()
    model = session_model or default_model
    provider = session_provider or default_provider
    if session_model:
        measurement = "session_store"
    elif default_model or default_provider:
        measurement = "config"
    else:
        measurement = "unknown"
    return {
        "agent_id": "hermes",
        "session_key": session_key,
        "conversation_id": conversation_id,
        "model": model,
        "model_provider": provider,
        "measurement": measurement,
        "runtime_kind": "hermes",
        "runtime_name": "Hermes",
        "host_version": host_version,
        "adapter_version": adapter_version,
    }


def _cached_agent_for_conversation(gateway_runner: Any, conversation_id: str | None) -> Any:
    if not conversation_id:
        return None
    marker = f":conversation:{conversation_id}"
    running = getattr(gateway_runner, "_running_agents", None)
    if running is not None:
        try:
            for key, agent in running.items():
                if marker in str(key) and getattr(agent, "context_compressor", None):
                    return agent
        except Exception:
            pass
    cache = getattr(gateway_runner, "_agent_cache", None)
    lock = getattr(gateway_runner, "_agent_cache_lock", None)
    if cache is None:
        return None
    try:
        if lock:
            with lock:
                entries = list(cache.items())
        else:
            entries = list(cache.items())
    except Exception:
        return None
    for key, entry in entries:
        if marker not in str(key):
            continue
        agent = entry[0] if isinstance(entry, (tuple, list)) and entry else entry
        if getattr(agent, "context_compressor", None):
            return agent
    return None


def live_context_usage(
    gateway_runner: Any,
    gateway_session_key: str | None,
    conversation_id: str | None = None,
) -> dict[str, Any]:
    """Read current context occupancy from Hermes' resident session agent."""
    if gateway_runner is None:
        return {}
    agent = None
    resolver = getattr(gateway_runner, "_resident_agent_for", None)
    if callable(resolver) and gateway_session_key:
        try:
            agent = resolver(gateway_session_key)
        except Exception:
            pass
    if agent is None:
        agent = _cached_agent_for_conversation(gateway_runner, conversation_id)
    compressor = getattr(agent, "context_compressor", None) if agent is not None else None
    if compressor is None:
        return {}
    try:
        from agent.context_breakdown import context_usage_fields

        usage = context_usage_fields(compressor)
    except Exception:
        used = max(0, int(getattr(compressor, "last_prompt_tokens", 0) or 0))
        maximum = max(0, int(getattr(compressor, "context_length", 0) or 0))
        usage = (
            {
                "context_used": used,
                "context_max": maximum,
                "context_percent": max(0, min(100, round(used / maximum * 100))),
            }
            if used and maximum
            else {}
        )
    used = usage.get("context_used")
    maximum = usage.get("context_max")
    if not isinstance(used, (int, float)) or used <= 0:
        return {}
    if not isinstance(maximum, (int, float)) or maximum <= 0:
        return {}
    return {
        "context_used_tokens": int(used),
        "context_window_tokens": int(maximum),
        "context_usage_percent": int(
            usage.get("context_percent")
            or max(0, min(100, round(used / maximum * 100)))
        ),
        "context_measurement": usage.get("context_source") or "provider_usage",
    }


def context_usage_with_cache(
    cache: dict[str, dict[str, Any]],
    conversation_id: str,
    live_usage: dict[str, Any],
) -> dict[str, Any]:
    """Keep the last context occupancy after Hermes evicts the resident agent."""
    if live_usage:
        snapshot = dict(live_usage)
        cache[conversation_id] = snapshot
        return snapshot
    return dict(cache.get(conversation_id, {}))
