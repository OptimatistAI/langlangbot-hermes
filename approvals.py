"""Bridge Hermes gateway approvals to the LangLangBot sidecar SSE."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Callable
from uuid import uuid4

try:
    from .sidecar import LanglangbotSidecarClient
except ImportError:
    from sidecar import LanglangbotSidecarClient

logger = logging.getLogger("langlangbot.hermes.approvals")

HERMES_APPROVAL_KIND = "hermes.exec"

_HERMES_DECISIONS = frozenset({"once", "session", "always", "deny"})
_OPERATOR_TO_HERMES = {
    "allow-once": "once",
    "once": "once",
    "session": "session",
    "allow-always": "always",
    "always": "always",
    "deny": "deny",
}

DEFAULT_ACTIONS = [
    {"decision": "allow-once", "label": "Allow once", "style": "primary"},
    {"decision": "allow-always", "label": "Always allow", "style": "secondary"},
    {"decision": "deny", "label": "Deny", "style": "danger"},
]


def approval_actions(data: dict[str, Any]) -> list[dict[str, str]]:
    actions = [dict(DEFAULT_ACTIONS[0])]
    if not data.get("smart_denied"):
        if data.get("allow_session", True):
            actions.append(
                {
                    "decision": "session",
                    "label": "Allow session",
                    "style": "secondary",
                }
            )
        if data.get("allow_permanent", True):
            actions.append(dict(DEFAULT_ACTIONS[1]))
    actions.append(dict(DEFAULT_ACTIONS[2]))
    return actions


def map_operator_decision(decision: str | None) -> str | None:
    if not decision:
        return None
    mapped = _OPERATOR_TO_HERMES.get(decision.strip().lower())
    if mapped in _HERMES_DECISIONS:
        return mapped
    return None


def approval_title(data: dict[str, Any]) -> str:
    command = str(data.get("command") or data.get("command_preview") or "").strip()
    description = str(data.get("description") or "").strip()
    if command:
        return f"Exec: {command[:120]}"
    if description:
        return description[:120]
    return "Hermes needs approval"


def approval_description(data: dict[str, Any]) -> str | None:
    lines: list[str] = []
    command = str(data.get("command") or data.get("command_preview") or "").strip()
    description = str(data.get("description") or "").strip()
    if command:
        lines.append(command[:2000])
    if description and description != command:
        lines.append(description[:2000])
    pattern = data.get("pattern_key") or data.get("pattern")
    if pattern:
        lines.append(f"pattern: {pattern}")
    return "\n".join(lines) if lines else None


def approval_id_for(data: dict[str, Any]) -> str:
    for key in ("request_id", "approval_id", "id"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return f"hermes:{uuid4()}"


def expires_at_iso(data: dict[str, Any], default_seconds: int = 120) -> str:
    raw = data.get("expires_at") or data.get("expiresAt")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    timeout = data.get("timeout_ms") or data.get("timeoutMs")
    seconds = default_seconds
    if isinstance(timeout, (int, float)) and timeout > 0:
        seconds = max(1, int(timeout / 1000))
    return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat()


def pending_payload(
    data: dict[str, Any],
    conversation_id: str,
) -> dict[str, Any]:
    return {
        "approval_id": approval_id_for(data),
        "kind": HERMES_APPROVAL_KIND,
        "conversation_id": conversation_id,
        "title": approval_title(data),
        "description": approval_description(data),
        "actions": approval_actions(data),
        "metadata": {
            "approval_kind": HERMES_APPROVAL_KIND,
            "session_key": data.get("session_key"),
            "pattern_key": data.get("pattern_key"),
            "tool_call_id": data.get("tool_call_id"),
        },
        "expires_at": expires_at_iso(data),
    }


def _approval_api() -> tuple[Any, Any, Any] | None:
    try:
        from tools.approval import (
            register_gateway_notify,
            resolve_gateway_approval,
            unregister_gateway_notify,
        )
    except ImportError:
        return None
    return register_gateway_notify, unregister_gateway_notify, resolve_gateway_approval


def register_notify(
    session_key: str,
    callback: Callable[[dict[str, Any]], None],
) -> bool:
    api = _approval_api()
    if api is None:
        logger.warning("tools.approval.register_gateway_notify is unavailable")
        return False
    api[0](session_key, callback)
    return True


def unregister_notify(session_key: str) -> None:
    api = _approval_api()
    if api is None:
        return
    api[1](session_key)


def resolve_hermes_approval(
    session_key: str,
    choice: str,
    request_id: str | None = None,
) -> int:
    api = _approval_api()
    if api is None:
        return 0
    return api[2](session_key, choice, request_id=request_id)


def deliver_pending(
    client: LanglangbotSidecarClient,
    conversation_id: str,
    data: dict[str, Any],
) -> str:
    payload = pending_payload(data, conversation_id)
    client.register_approval_pending(**payload)
    return str(payload["approval_id"])


def apply_plugin_event(
    event: dict[str, Any],
    pending: dict[str, tuple[str, str]],
    *,
    mark_resolved: Callable[[str, str | None], None] | None = None,
) -> None:
    """pending maps approval_id → (session_key, request_id)."""
    approval_id = str(event.get("approval_id") or "")
    if not approval_id:
        return
    if approval_id not in pending:
        if event.get("type") == "approval_decided":
            logger.info(
                "ignoring unmatched approval decision id=%s",
                approval_id,
            )
        return
    session_key, request_id = pending[approval_id]
    event_type = event.get("type")
    if event_type == "approval_resolved":
        pending.pop(approval_id, None)
        return
    if event_type != "approval_decided":
        return
    choice = map_operator_decision(event.get("decision"))
    if not choice:
        logger.warning("unrecognized Operator approval decision: %s", event.get("decision"))
        return
    resolved = resolve_hermes_approval(
        session_key,
        choice,
        request_id=request_id or None,
    )
    if resolved < 1:
        logger.warning(
            "Operator approval no longer matches a pending Hermes request: %s",
            approval_id,
        )
        return
    if mark_resolved:
        try:
            mark_resolved(approval_id, event.get("decision"))
        except Exception as err:  # noqa: BLE001
            logger.warning("mark_approval_resolved failed: %s", err)
    pending.pop(approval_id, None)
