"""Stage local files into the sidecar media root and send outbound parts."""

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path
from typing import Any

try:
    from .sidecar import LanglangbotSidecarClient
except ImportError:
    from sidecar import LanglangbotSidecarClient


def media_root() -> Path:
    raw = os.getenv("LANGLANGBOT_MEDIA_ROOT", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return (Path.home() / ".openclaw" / "media" / "langlangbot").resolve()


def is_within_media_root(path: Path) -> bool:
    root = media_root()
    try:
        path.resolve().relative_to(root)
        return True
    except ValueError:
        return False


def is_remote_url(value: str) -> bool:
    lowered = value.strip().lower()
    return lowered.startswith(("http://", "https://", "data:"))


def visible_name(path: Path) -> str:
    name = path.name or "attachment.bin"
    # OpenClaw stages as <name>---<uuid><ext>; strip that suffix if present.
    stem = path.stem
    if "---" in stem:
        prefix, maybe_uuid = stem.rsplit("---", 1)
        if len(maybe_uuid) == 36:
            return f"{prefix}{path.suffix}"
    return name


def stage_into_media_root(local_path: Path, filename: str) -> tuple[Path, Path | None]:
    """Return (path_to_register, cleanup_dir_or_None)."""
    resolved = local_path.expanduser().resolve()
    if is_within_media_root(resolved):
        return resolved, None
    staged_dir = media_root() / "outbound" / str(uuid.uuid4())
    staged_dir.mkdir(parents=True, exist_ok=True)
    staged = staged_dir / filename
    try:
        os.link(resolved, staged)
    except OSError:
        shutil.copy2(resolved, staged)
    return staged, staged_dir


def register_local_file(
    client: LanglangbotSidecarClient,
    conversation_id: str,
    local_path: str,
) -> dict[str, Any]:
    source = Path(local_path)
    filename = visible_name(source)
    staged, cleanup_dir = stage_into_media_root(source, filename)
    try:
        registered = client.register_outbound_attachment(
            conversation_id,
            local_path=str(staged),
            filename=filename,
            status="ready",
        )
    finally:
        if cleanup_dir is not None:
            shutil.rmtree(cleanup_dir, ignore_errors=True)
    return {
        "type": "attachment",
        "upload_id": registered.get("upload_id"),
        "attachment_id": registered.get("attachment_id"),
        "status": registered.get("status") or "ready",
        "kind": registered.get("kind") or "file",
        "mime": registered.get("mime") or "application/octet-stream",
        "filename": filename,
        "size": registered.get("size") or 0,
        "download_url": registered.get("download_url"),
    }


def resolve_caption(caption: str | None, local_paths: list[str]) -> str:
    text = (caption or "").strip()
    if not text:
        return ""
    names = set(local_paths)
    for path in local_paths:
        names.add(Path(path).name)
        names.add(visible_name(Path(path)))
    return "" if text in names else text


def send_outbound_files(
    client: LanglangbotSidecarClient,
    conversation_id: str,
    local_paths: list[str],
    caption: str | None = None,
) -> dict[str, Any]:
    parts: list[dict[str, Any]] = []
    for local_path in local_paths:
        if is_remote_url(local_path):
            raise ValueError(
                f"remote media is not supported ({local_path}); "
                "write a local file under the sidecar media root first"
            )
        parts.append(register_local_file(client, conversation_id, local_path))
    result = client.send_message(
        conversation_id,
        resolve_caption(caption, local_paths),
        parts=parts,
    )
    return {"message_id": result.get("message_id", ""), "parts": parts}
