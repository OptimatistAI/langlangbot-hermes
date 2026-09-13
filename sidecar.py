"""LangLangBot sidecar HTTP / SSE client used by the Hermes adapter."""

from __future__ import annotations

import json
import ssl
from dataclasses import dataclass, field
from typing import Any
from urllib import request
from urllib.parse import quote, urlparse

try:
    from .config import AGENT_RUNTIME_NAME, DEFAULT_SIDECAR_URL
except ImportError:
    from config import AGENT_RUNTIME_NAME, DEFAULT_SIDECAR_URL  # type: ignore


_TERMINAL_ATTACHMENT_STATUSES = frozenset({"ready", "failed"})
_AGENT_RUNTIME_KIND = "hermes"
_DEFAULT_ACCOUNT_ID = "default"


@dataclass
class InboundUserMessage:
    conversation_id: str
    message_id: str
    text: str
    received_at: str
    seq: str | None = None
    parts: list[dict[str, Any]] = field(default_factory=list)

    def awaiting_attachments(self) -> bool:
        return any_pending_attachments(self.parts)


@dataclass
class InboundAttachmentEvent:
    kind: str
    conversation_id: str
    upload_id: str
    message_id: str | None = None
    seq: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)


InboundEvent = InboundUserMessage | InboundAttachmentEvent


def any_pending_attachments(parts: list[dict[str, Any]] | None) -> bool:
    for part in parts or []:
        if part.get("type") != "attachment":
            continue
        if part.get("status") not in _TERMINAL_ATTACHMENT_STATUSES:
            return True
    return False


def local_paths_from_parts(parts: list[dict[str, Any]] | None) -> list[str]:
    paths: list[str] = []
    for part in parts or []:
        if part.get("type") != "attachment":
            continue
        if part.get("status") != "ready":
            continue
        path = part.get("local_path")
        if isinstance(path, str) and path.strip():
            paths.append(path.strip())
    return paths


class LanglangbotSidecarClient:
    def __init__(self, base_url: str, plugin_token: str | None = None) -> None:
        self.base_url = (base_url or DEFAULT_SIDECAR_URL).rstrip("/")
        self.plugin_token = plugin_token
        self._ssl_context = _loopback_insecure_ssl_context(self.base_url)

    def _urlopen(self, req: request.Request, timeout: float | None):
        return request.urlopen(req, timeout=timeout, context=self._ssl_context)

    def _headers(self, accept: str | None = None) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if accept:
            headers["Accept"] = accept
        if self.plugin_token:
            headers["X-Langlangbot-Plugin-Token"] = self.plugin_token
        return headers

    def health(self) -> dict[str, Any]:
        return self._json_request("GET", "/health")

    def send_delta(self, conversation_id: str, text: str) -> None:
        self._json_request(
            "POST",
            f"/v1/conversations/{conversation_id}/outbound/delta",
            {"text": text},
        )

    def send_message(
        self,
        conversation_id: str,
        text: str,
        parts: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"text": text, "parts": parts or []}
        return self._json_request(
            "POST",
            f"/v1/conversations/{conversation_id}/outbound/message",
            body,
        )

    def report_turn_phase(
        self,
        conversation_id: str,
        message_id: str,
        phase: str,
        detail: str | None = None,
    ) -> None:
        body: dict[str, Any] = {"message_id": message_id, "phase": phase}
        if detail is not None:
            body["detail"] = detail
        self._json_request(
            "POST",
            f"/v1/conversations/{conversation_id}/outbound/turn",
            body,
        )

    def ack_inbound(self, cursor: str) -> None:
        self._json_request("POST", "/v1/inbound/ack", {"cursor": cursor})

    def update_runtime_status(
        self,
        *,
        connected: bool,
        agent_runtime_ready: bool,
        reason: str | None = None,
        last_dispatch_error: str | None = None,
        host_version: str | None = None,
        adapter_version: str | None = None,
    ) -> None:
        self._json_request(
            "PUT",
            "/v1/plugin/runtime/status",
            {
                "connected": connected,
                "agent_runtime_ready": agent_runtime_ready,
                "runtime_name": AGENT_RUNTIME_NAME,
                "kind": _AGENT_RUNTIME_KIND,
                "host_version": host_version,
                "adapter_version": adapter_version,
                "account_id": _DEFAULT_ACCOUNT_ID,
                "reason": reason,
                "last_dispatch_error": last_dispatch_error,
            },
        )

    def post_management_result(
        self,
        request_id: str,
        *,
        ok: bool,
        result: dict[str, Any] | None = None,
        error: dict[str, str] | None = None,
    ) -> None:
        body: dict[str, Any] = {"ok": ok}
        if result is not None:
            body["result"] = result
        if error is not None:
            body["error"] = error
        self._json_request(
            "POST",
            f"/v1/plugin/management/{request_id}/result",
            body,
        )

    def register_outbound_attachment(
        self,
        conversation_id: str,
        *,
        local_path: str,
        filename: str,
        status: str = "ready",
        account_id: str = _DEFAULT_ACCOUNT_ID,
    ) -> dict[str, Any]:
        headers = self._headers()
        headers["X-Langlangbot-Account-Id"] = account_id
        return self._json_request(
            "POST",
            f"/v1/conversations/{conversation_id}/outbound/attachments",
            {
                "local_path": local_path,
                "filename": filename,
                "status": status,
            },
            headers=headers,
        )

    def register_approval_pending(
        self,
        *,
        approval_id: str,
        kind: str,
        title: str,
        expires_at: str,
        conversation_id: str | None = None,
        description: str | None = None,
        actions: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._json_request(
            "POST",
            "/v1/approvals/pending",
            {
                "approval_id": approval_id,
                "kind": kind,
                "conversation_id": conversation_id,
                "title": title,
                "description": description,
                "actions": actions or [],
                "metadata": metadata or {},
                "expires_at": expires_at,
            },
        )

    def get_approval_decision(self, approval_id: str) -> dict[str, Any]:
        return self._json_request(
            "GET",
            f"/v1/approvals/{approval_id}/decision",
        )

    def mark_approval_resolved(
        self,
        approval_id: str,
        decision: str | None = None,
    ) -> dict[str, Any]:
        return self._json_request(
            "POST",
            f"/v1/approvals/{approval_id}/resolved",
            {"decision": decision},
        )

    def get_plugin_connection_current(self, conversation_id: str) -> dict[str, Any]:
        return self._json_request(
            "GET",
            f"/v1/plugin/connection/current?conversation_id={quote(conversation_id, safe='')}",
        )

    def iter_sse(self, path: str) -> Any:
        url = f"{self.base_url}{path}"
        headers = self._headers("text/event-stream")
        req = request.Request(url, headers=headers, method="GET")
        return self._urlopen(req, timeout=None)

    def stream_named_events(self, path: str):
        with self.iter_sse(path) as resp:
            event_name = "message"
            event_id: str | None = None
            data_lines: list[str] = []
            for raw_line in resp:
                line = raw_line.decode("utf-8").rstrip("\r\n")
                if line == "":
                    if not data_lines:
                        event_id = None
                        continue
                    payload = json.loads("\n".join(data_lines))
                    data_lines = []
                    yield event_name, event_id, payload
                    event_name = "message"
                    event_id = None
                    continue
                if line.startswith("event:"):
                    event_name = line.split(":", 1)[1].strip()
                elif line.startswith("id:"):
                    event_id = line.split(":", 1)[1].strip()
                elif line.startswith("data:"):
                    data_lines.append(line.split(":", 1)[1].lstrip())

    def stream_management(self):
        path = f"/v1/plugin/management/events?account_id={_DEFAULT_ACCOUNT_ID}"
        for event_name, _event_id, payload in self.stream_named_events(path):
            if event_name == "management_request":
                yield payload

    def stream_inbound(self) -> Any:
        for event_name, event_id, payload in self.stream_named_events(
            "/v1/inbound/events"
        ):
            parsed = parse_inbound_event(event_name, event_id, payload)
            if parsed is not None:
                yield parsed

    def stream_approval_events(self):
        for event_name, _event_id, payload in self.stream_named_events(
            "/v1/approvals/plugin/events"
        ):
            if not isinstance(payload, dict):
                continue
            if payload.get("type") in ("approval_decided", "approval_resolved"):
                yield payload
            elif event_name in ("approval_decided", "approval_resolved"):
                payload = dict(payload)
                payload.setdefault("type", event_name)
                yield payload

    def _json_request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        data = None if body is None else json.dumps(body).encode("utf-8")
        req = request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers=headers or self._headers(),
            method=method,
        )
        with self._urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
            if not raw:
                return {}
            return json.loads(raw)


def parse_inbound_event(
    event_name: str,
    event_id: str | None,
    payload: dict[str, Any],
) -> InboundEvent | None:
    if not isinstance(payload, dict):
        return None
    if event_name in ("user_message", "message"):
        if event_name == "message" and payload.get("upload_id"):
            return None
        if "text" not in payload:
            return None
        if event_name == "message" and not (
            payload.get("conversation_id")
            and payload.get("message_id")
            and isinstance(payload.get("text"), str)
            and payload.get("received_at")
        ):
            return None
        return InboundUserMessage(
            conversation_id=str(payload["conversation_id"]),
            message_id=str(payload["message_id"]),
            text=str(payload.get("text") or ""),
            received_at=str(payload.get("received_at") or ""),
            seq=event_id,
            parts=list(payload.get("parts") or []),
        )
    if event_name in (
        "attachment_available",
        "attachment_ready",
        "attachment_failed",
    ):
        kind = event_name.removeprefix("attachment_")
        upload_id = payload.get("upload_id")
        conversation_id = payload.get("conversation_id")
        if not upload_id or not conversation_id:
            return None
        message_id = payload.get("message_id")
        return InboundAttachmentEvent(
            kind=kind,
            conversation_id=str(conversation_id),
            upload_id=str(upload_id),
            message_id=None if message_id is None else str(message_id),
            seq=event_id,
            payload=payload,
        )
    return None


def _loopback_insecure_ssl_context(base_url: str) -> ssl.SSLContext | None:
    if not base_url.startswith("https://"):
        return None
    host = urlparse(base_url).hostname or ""
    if host not in ("127.0.0.1", "localhost"):
        return None
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx
