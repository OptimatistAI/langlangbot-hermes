"""Inbound attachment delay + terminal dispatch for Hermes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:
    from .sidecar import (
        InboundAttachmentEvent,
        InboundUserMessage,
        any_pending_attachments,
        local_paths_from_parts,
    )
except ImportError:
    from sidecar import (
        InboundAttachmentEvent,
        InboundUserMessage,
        any_pending_attachments,
        local_paths_from_parts,
    )


def message_key(conversation_id: str, message_id: str) -> str:
    return f"{conversation_id}:{message_id}"


def upload_key(conversation_id: str, upload_id: str) -> str:
    return f"{conversation_id}:{upload_id}"


def remember_ack_seq(seqs: list[str], seq: str | None) -> None:
    trimmed = (seq or "").strip()
    if trimmed and trimmed not in seqs:
        seqs.append(trimmed)


def replace_attachment_part(
    parts: list[dict[str, Any]],
    upload_id: str,
    replacement: dict[str, Any],
) -> list[dict[str, Any]]:
    updated: list[dict[str, Any]] = []
    replaced = False
    for part in parts:
        if part.get("type") == "attachment" and part.get("upload_id") == upload_id:
            updated.append(replacement)
            replaced = True
        else:
            updated.append(part)
    if not replaced:
        updated.append(replacement)
    return updated


def attachment_part_from_event(event: InboundAttachmentEvent) -> dict[str, Any]:
    payload = event.payload
    status = {
        "available": "uploading",
        "ready": "ready",
        "failed": "failed",
    }.get(event.kind, "processing")
    part: dict[str, Any] = {
        "type": "attachment",
        "upload_id": event.upload_id,
        "status": status,
        "kind": payload.get("kind") or "file",
        "mime": payload.get("mime") or "application/octet-stream",
        "filename": payload.get("filename") or "attachment",
        "size": payload.get("size") or payload.get("bytes_available") or 0,
    }
    if payload.get("attachment_id"):
        part["attachment_id"] = payload["attachment_id"]
    if payload.get("download_url"):
        part["download_url"] = payload["download_url"]
    if payload.get("local_path"):
        part["local_path"] = payload["local_path"]
    if payload.get("reason"):
        part["failure_reason"] = payload["reason"]
    return part


def format_terminal_attachments_text(
    caption: str,
    parts: list[dict[str, Any]],
) -> str:
    attachments = [part for part in parts if part.get("type") == "attachment"]
    ready = [part for part in attachments if part.get("status") == "ready"]
    failed = [part for part in attachments if part.get("status") == "failed"]
    lines: list[str] = []
    trimmed = caption.strip()
    if trimmed:
        lines.append(trimmed)
    if not ready and failed:
        lines.append(
            f"All {len(failed)} attachment(s) failed. Explain each failure "
            "to the user; there is nothing to analyze."
        )
        lines.extend(_part_lines(failed))
        return "\n".join(lines)
    if ready:
        if failed:
            lines.append(
                f"{len(ready)} attachment(s) ready for analysis; {len(failed)} failed."
            )
        else:
            lines.append(f"{len(ready)} attachment(s) ready for analysis.")
        lines.append(
            "Analyze every ready attachment (use image/file tools with "
            "path= or url= as needed)."
        )
        lines.extend(_part_lines(ready))
    if failed:
        lines.append(
            "Failed attachment(s) — do not analyze these; briefly tell the "
            "user each reason."
        )
        lines.extend(_part_lines(failed))
    return "\n".join(lines).strip()


def _part_lines(parts: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for part in parts:
        filename = part.get("filename") or "attachment"
        status = part.get("status") or "unknown"
        extra = ""
        if part.get("local_path"):
            extra += f" path={part['local_path']}"
        if part.get("download_url"):
            extra += f" url={part['download_url']}"
        if part.get("failure_reason"):
            extra += f" reason={part['failure_reason']}"
        lines.append(f"[attachment {filename} ({status}){extra}]")
    return lines


@dataclass
class PendingInbound:
    conversation_id: str
    message_id: str
    text: str
    parts: list[dict[str, Any]]
    pending_ack_seqs: list[str] = field(default_factory=list)


@dataclass
class DispatchReady:
    conversation_id: str
    message_id: str
    text: str
    parts: list[dict[str, Any]]
    media_urls: list[str]
    ack_seqs: list[str]


class InboundAttachmentBuffer:
    """Hold user_message until every attachment part is terminal."""

    def __init__(self) -> None:
        self._pending: dict[str, PendingInbound] = {}
        self._by_upload: dict[str, str] = {}
        self._processed_uploads: set[str] = set()
        self._dispatched: set[str] = set()

    def on_user_message(self, inbound: InboundUserMessage) -> DispatchReady | None:
        key = message_key(inbound.conversation_id, inbound.message_id)
        if inbound.awaiting_attachments():
            handle = PendingInbound(
                conversation_id=inbound.conversation_id,
                message_id=inbound.message_id,
                text=inbound.text,
                parts=list(inbound.parts or []),
            )
            remember_ack_seq(handle.pending_ack_seqs, inbound.seq)
            self._pending[key] = handle
            self._remember_uploads(key, handle.parts)
            return None
        if key in self._dispatched:
            return DispatchReady(
                conversation_id=inbound.conversation_id,
                message_id=inbound.message_id,
                text=inbound.text,
                parts=list(inbound.parts or []),
                media_urls=local_paths_from_parts(inbound.parts),
                ack_seqs=[inbound.seq] if inbound.seq else [],
            )
        return self._dispatch(
            PendingInbound(
                conversation_id=inbound.conversation_id,
                message_id=inbound.message_id,
                text=inbound.text,
                parts=list(inbound.parts or []),
                pending_ack_seqs=[inbound.seq] if inbound.seq else [],
            )
        )

    def on_attachment(self, event: InboundAttachmentEvent) -> DispatchReady | None:
        if event.kind == "available":
            self._apply_part(event)
            return None
        terminal = upload_key(event.conversation_id, event.upload_id)
        if terminal in self._processed_uploads:
            return DispatchReady(
                conversation_id=event.conversation_id,
                message_id=event.message_id or event.upload_id,
                text="",
                parts=[],
                media_urls=[],
                ack_seqs=[event.seq] if event.seq else [],
            )
        self._processed_uploads.add(terminal)
        handle = self._apply_part(event)
        if handle is None:
            if event.kind != "ready" and event.kind != "failed":
                return None
            parts = [attachment_part_from_event(event)]
            return self._dispatch(
                PendingInbound(
                    conversation_id=event.conversation_id,
                    message_id=event.message_id or event.upload_id,
                    text="",
                    parts=parts,
                    pending_ack_seqs=[event.seq] if event.seq else [],
                )
            )
        if any_pending_attachments(handle.parts):
            return None
        return self._dispatch(handle)

    def _apply_part(self, event: InboundAttachmentEvent) -> PendingInbound | None:
        key = self._resolve_key(event)
        handle = self._pending.get(key) if key else None
        if handle is None:
            return None
        remember_ack_seq(handle.pending_ack_seqs, event.seq)
        handle.parts = replace_attachment_part(
            handle.parts,
            event.upload_id,
            attachment_part_from_event(event),
        )
        if key:
            self._by_upload[event.upload_id] = key
        return handle

    def _resolve_key(self, event: InboundAttachmentEvent) -> str | None:
        if event.message_id:
            key = message_key(event.conversation_id, event.message_id)
            if key in self._pending:
                return key
        return self._by_upload.get(event.upload_id)

    def _remember_uploads(self, key: str, parts: list[dict[str, Any]]) -> None:
        for part in parts:
            upload_id = part.get("upload_id")
            if part.get("type") == "attachment" and isinstance(upload_id, str):
                self._by_upload[upload_id] = key

    def _dispatch(self, handle: PendingInbound) -> DispatchReady:
        key = message_key(handle.conversation_id, handle.message_id)
        self._dispatched.add(key)
        self._pending.pop(key, None)
        for part in handle.parts:
            upload_id = part.get("upload_id")
            if isinstance(upload_id, str):
                self._by_upload.pop(upload_id, None)
        text = handle.text
        if any(part.get("type") == "attachment" for part in handle.parts):
            text = format_terminal_attachments_text(handle.text, handle.parts)
        return DispatchReady(
            conversation_id=handle.conversation_id,
            message_id=handle.message_id,
            text=text,
            parts=handle.parts,
            media_urls=local_paths_from_parts(handle.parts),
            ack_seqs=list(handle.pending_ack_seqs),
        )


def claim_turn_decision(payload: dict[str, Any]) -> str:
    """Classify a durable accept response.

    ``duplicate`` means the message was already claimed, including when the
    adaptor crashed after accept and before the model ran. That message is not
    replayed; the user sends it again.
    """
    if payload.get("duplicate") is True:
        return "duplicate"
    if payload.get("accepted") is True:
        return "accepted"
    return "rejected"
