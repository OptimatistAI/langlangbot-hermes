"""Hermes plugin hooks → sidecar turn phases and session model tracking."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from adapter import LanglangbotAdapter

logger = logging.getLogger("langlangbot.hermes.hooks")

_adapter: Any = None


def bind_adapter(adapter: Any) -> None:
    global _adapter
    _adapter = adapter


def active_conversation_id(session_id: str | None = None) -> str | None:
    adapter: LanglangbotAdapter | None = _adapter
    if adapter is None:
        return None
    return adapter.conversation_for_session(session_id)


def active_runtime_status(conversation_id: str) -> dict[str, Any] | None:
    adapter: LanglangbotAdapter | None = _adapter
    if adapter is None:
        return None
    return adapter._status_payload(conversation_id)


def _is_langlangbot(kwargs: dict[str, Any]) -> bool:
    surface = str(kwargs.get("surface") or kwargs.get("platform") or "").lower()
    return not surface or "langlangbot" in surface


def on_session_start(session_id: str = "", model: str = "", **kwargs: Any) -> None:
    adapter: LanglangbotAdapter | None = _adapter
    if adapter is None or not _is_langlangbot(kwargs):
        return
    provider = kwargs.get("provider")
    adapter.remember_session(
        session_id,
        model=model,
        provider=str(provider) if provider else None,
    )


def on_stream_start(session_id: str = "", **kwargs: Any) -> None:
    adapter: LanglangbotAdapter | None = _adapter
    if adapter is None or not _is_langlangbot(kwargs):
        return
    adapter.remember_session(
        session_id,
        model=str(kwargs.get("model") or "") or None,
        provider=str(kwargs.get("provider") or "") or None,
    )
    adapter.report_hook_phase(session_id, "working")


def on_stream_delta(
    delta: str = "",
    kind: str = "text",
    session_id: str = "",
    **kwargs: Any,
) -> None:
    adapter: LanglangbotAdapter | None = _adapter
    if adapter is None or not _is_langlangbot(kwargs):
        return
    if kind == "reasoning":
        adapter.report_hook_phase(session_id, "thinking", (delta or "")[:200] or None)
        return
    adapter.report_hook_phase(session_id, "streaming")


def on_stream_end(
    session_id: str = "",
    finished: bool = True,
    error: str | None = None,
    **kwargs: Any,
) -> None:
    adapter: LanglangbotAdapter | None = _adapter
    if adapter is None or not _is_langlangbot(kwargs):
        return
    if error or finished is False:
        logger.debug(
            "Hermes stream attempt ended without final delivery; gateway may retry "
            "session=%s turn=%s error=%s",
            session_id,
            kwargs.get("turn_id"),
            error,
        )
        return
    # send() / send_draft report idle after the Operator-visible reply is posted.


def pre_tool_call(tool_name: str = "", session_id: str = "", **kwargs: Any) -> None:
    adapter: LanglangbotAdapter | None = _adapter
    if adapter is None:
        return
    if not _is_langlangbot(kwargs) and not adapter.session_is_ours(session_id):
        return
    adapter.report_hook_phase(session_id, "tool", tool_name or None)


def post_tool_call(tool_name: str = "", session_id: str = "", **kwargs: Any) -> None:
    adapter: LanglangbotAdapter | None = _adapter
    if adapter is None:
        return
    if not _is_langlangbot(kwargs) and not adapter.session_is_ours(session_id):
        return
    adapter.report_hook_phase(session_id, "working", tool_name or None)


def pre_approval_request(**kwargs: Any) -> None:
    logger.debug(
        "hermes approval requested session=%s tool=%s",
        kwargs.get("session_key") or kwargs.get("session_id"),
        kwargs.get("tool_call_id"),
    )


def post_approval_response(**kwargs: Any) -> None:
    logger.debug(
        "hermes approval resolved session=%s choice=%s",
        kwargs.get("session_key") or kwargs.get("session_id"),
        kwargs.get("choice"),
    )


def register_hooks(ctx: Any) -> None:
    ctx.register_hook("on_session_start", on_session_start)
    ctx.register_hook("on_stream_start", on_stream_start)
    ctx.register_hook("on_stream_delta", on_stream_delta)
    ctx.register_hook("on_stream_end", on_stream_end)
    ctx.register_hook("pre_tool_call", pre_tool_call)
    ctx.register_hook("post_tool_call", post_tool_call)
    ctx.register_hook("pre_approval_request", pre_approval_request)
    ctx.register_hook("post_approval_response", post_approval_response)
