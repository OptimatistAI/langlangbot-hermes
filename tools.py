"""Outbound Hermes client tools (loaded eagerly via provides_tools)."""

from __future__ import annotations

import json
from typing import Any

try:
    from .config import (
        adapter_version,
        configured_plugin_token,
        configured_sidecar_url,
        host_version,
    )
    from .hooks import active_conversation_id, active_runtime_status
    from .management import build_status, default_model_from_config
    from .sidecar import LanglangbotSidecarClient
except ImportError:
    from config import (  # type: ignore
        adapter_version,
        configured_plugin_token,
        configured_sidecar_url,
        host_version,
    )
    from hooks import active_conversation_id, active_runtime_status  # type: ignore
    from management import build_status, default_model_from_config  # type: ignore
    from sidecar import LanglangbotSidecarClient  # type: ignore


TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "conversation_id": {
            "type": "string",
            "description": (
                "LangLangBot conversation id. Defaults to the active "
                "langlangbot session when omitted."
            ),
        }
    },
}


def _client() -> LanglangbotSidecarClient:
    url = configured_sidecar_url()
    if not url:
        raise RuntimeError(
            "LANGLANGBOT_SIDECAR_URL is not configured; pair the sidecar first"
        )
    return LanglangbotSidecarClient(url, configured_plugin_token() or None)


def _conversation_id(args: dict[str, Any] | None, session_id: str | None = None) -> str:
    value = (args or {}).get("conversation_id")
    if isinstance(value, str) and value.strip():
        return value.strip().removeprefix("conversation:")
    value = active_conversation_id(session_id)
    if isinstance(value, str) and value.strip():
        return value.strip().removeprefix("conversation:")
    raise ValueError(
        "conversation_id is required when no active langlangbot session is available"
    )


def langlangbot_connection_current(
    args: dict[str, Any] | None = None,
    **kwargs: Any,
) -> str:
    conversation_id = _conversation_id(args, kwargs.get("session_id"))
    result = _client().get_plugin_connection_current(conversation_id)
    return json.dumps(result, ensure_ascii=False, default=str)


def langlangbot_operator_runtime_status(
    args: dict[str, Any] | None = None,
    **kwargs: Any,
) -> str:
    conversation_id = _conversation_id(args, kwargs.get("session_id"))
    status = active_runtime_status(conversation_id)
    source = "live" if status is not None else "hermes_config"
    if status is None:
        default_model, default_provider = default_model_from_config()
        status = build_status(
            conversation_id=conversation_id,
            session_key=None,
            session_model=default_model,
            session_provider=default_provider,
            host_version=host_version(),
            adapter_version=adapter_version(),
        )
    return json.dumps(
        {
            "session_key": status.get("session_key"),
            "model": status.get("model"),
            "model_provider": status.get("model_provider"),
            "measurement": status.get("measurement"),
            "context_used_tokens": status.get("context_used_tokens"),
            "context_window_tokens": status.get("context_window_tokens"),
            "context_usage_percent": status.get("context_usage_percent"),
            "context_measurement": status.get("context_measurement"),
            "source": source,
            "operator_runtime_bar_aligned": True,
            "guidance": (
                "Use these values when answering Operator questions about the "
                "current Hermes model."
            ),
        },
        ensure_ascii=False,
        default=str,
    )


def register_tools(ctx: Any) -> None:
    ctx.register_tool(
        name="langlangbot_connection_current",
        toolset="langlangbot",
        schema=TOOL_SCHEMA,
        handler=langlangbot_connection_current,
    )
    ctx.register_tool(
        name="langlangbot_operator_runtime_status",
        toolset="langlangbot",
        schema=TOOL_SCHEMA,
        handler=langlangbot_operator_runtime_status,
    )
