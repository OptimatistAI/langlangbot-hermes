"""Hermes platform adapter for LangLangBot."""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from typing import Any
from uuid import uuid4

from gateway.config import PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    Platform,
    SendResult,
)

try:
    from .approvals import (
        apply_plugin_event,
        deliver_pending,
        register_notify,
        unregister_notify,
    )
    from .config import (
        AGENT_RUNTIME_NAME,
        PLATFORM_NAME,
        adapter_version,
        apply_yaml_config,
        check_requirements,
        env_enablement,
        host_version,
        resolve_sidecar_settings,
        validate_config,
    )
    from .hooks import bind_adapter, register_hooks
    from .inbound import InboundAttachmentBuffer, claim_turn_decision
    from .management import (
        ManagementError,
        build_status,
        context_usage_with_cache,
        live_context_usage,
        list_hermes_models,
        model_selector_provider,
        session_model_command,
    )
    from .outbound import is_remote_url, send_outbound_files
    from .sidecar import (
        AGENT_RUNTIME_KIND,
        InboundAttachmentEvent,
        InboundUserMessage,
        LanglangbotSidecarClient,
        RuntimeMismatchError,
    )
except ImportError:
    from approvals import (  # type: ignore
        apply_plugin_event,
        deliver_pending,
        register_notify,
        unregister_notify,
    )
    from config import (  # type: ignore
        AGENT_RUNTIME_NAME,
        PLATFORM_NAME,
        adapter_version,
        apply_yaml_config,
        check_requirements,
        env_enablement,
        host_version,
        resolve_sidecar_settings,
        validate_config,
    )
    from hooks import bind_adapter, register_hooks  # type: ignore
    from inbound import InboundAttachmentBuffer, claim_turn_decision  # type: ignore
    from management import (  # type: ignore
        ManagementError,
        build_status,
        context_usage_with_cache,
        live_context_usage,
        list_hermes_models,
        model_selector_provider,
        session_model_command,
    )
    from outbound import is_remote_url, send_outbound_files  # type: ignore
    from sidecar import (  # type: ignore
        AGENT_RUNTIME_KIND,
        InboundAttachmentEvent,
        InboundUserMessage,
        LanglangbotSidecarClient,
        RuntimeMismatchError,
    )


logger = logging.getLogger("langlangbot.hermes")


class LanglangbotAdapter(BasePlatformAdapter):
    def __init__(self, config: PlatformConfig) -> None:
        super().__init__(config, Platform(PLATFORM_NAME))
        sidecar_url, plugin_token = resolve_sidecar_settings(config)
        self.sidecar_url = sidecar_url
        self.plugin_token = plugin_token
        self._client = LanglangbotSidecarClient(
            sidecar_url,
            plugin_token,
            runtime_kind=AGENT_RUNTIME_KIND,
        )
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task: asyncio.Task[Any] | None = None
        self._management_task: asyncio.Task[Any] | None = None
        self._approval_task: asyncio.Task[Any] | None = None
        self._inbound = InboundAttachmentBuffer()
        self._pending_user_messages: dict[str, deque[str]] = {}
        self._dispatched_message_ids: set[str] = set()
        self._once_phases: dict[str, set[str]] = {}
        self._turn_failed: set[str] = set()
        self._session_to_conversation: dict[str, str] = {}
        self._conversation_to_session: dict[str, str] = {}
        self._conversation_to_gateway_session: dict[str, str] = {}
        self._session_model: dict[str, tuple[str | None, str | None]] = {}
        self._context_usage: dict[str, dict[str, Any]] = {}
        self._approval_sessions: set[str] = set()
        self._pending_approvals: dict[str, tuple[str, str]] = {}
        self._last_conversation: str | None = None
        bind_adapter(self)

    def supports_draft_streaming(
        self,
        chat_type: str | None = None,
        metadata: dict[str, Any] | None = None,
        chat_id: str | None = None,
    ) -> bool:
        return True

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        self._loop = asyncio.get_running_loop()
        bind_adapter(self)
        health = await asyncio.to_thread(self._client.health)
        # Pairing-time exclusivity: refuse (permanently) when the sidecar was
        # re-paired to another runtime; entering poll loops would only 409.
        paired = health.get("paired_runtime_kind") if isinstance(health, dict) else None
        if isinstance(paired, str) and paired.strip() and paired.strip().lower() != AGENT_RUNTIME_KIND:
            reason = (
                f"sidecar is paired with {paired.strip()}; "
                f"re-pair with --runtime {AGENT_RUNTIME_KIND}"
            )
            logger.error("langlangbot connect refused: %s", reason)
            # Best-effort only: when the mismatch is on this sidecar, the
            # status PUT itself 409s. The reason stays visible above in the
            # adapter logs, which is the reliable surface.
            await self._report_runtime_status(
                connected=False,
                ready=False,
                reason=reason,
            )
            self._mark_disconnected()
            return False
        await self._report_runtime_status(connected=True, ready=True)
        if is_reconnect and self._tasks_running():
            self._mark_connected()
            return True
        if is_reconnect:
            await self._stop_background_tasks()
        self._task = asyncio.create_task(
            self._poll_forever(self._client.stream_inbound, self._handle_inbound_event, "inbound")
        )
        self._management_task = asyncio.create_task(
            self._poll_forever(
                self._client.stream_management,
                self._handle_management_request,
                "management",
            )
        )
        self._approval_task = asyncio.create_task(
            self._poll_forever(
                self._client.stream_approval_events,
                self._handle_approval_event,
                "approval",
            )
        )
        self._mark_connected()
        return True

    async def disconnect(self) -> None:
        for session_key in list(self._approval_sessions):
            unregister_notify(session_key)
        self._approval_sessions.clear()
        await self._stop_background_tasks()
        await self._report_runtime_status(
            connected=False,
            ready=False,
            reason="hermes adapter disconnected",
        )
        self._mark_disconnected()

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SendResult:
        conversation_id = _conversation_id(chat_id)
        is_interim = bool(metadata and metadata.get("_interim_send"))
        user_message_id = (
            None if is_interim else self._pop_pending_message(conversation_id)
        )
        try:
            result = await asyncio.to_thread(
                self._client.send_message,
                conversation_id,
                content,
            )
        except Exception as err:
            if is_interim:
                logger.warning("interim message delivery failed: %s", err)
            else:
                await self._fail_turn(conversation_id, user_message_id, err)
            return SendResult(success=False, message_id="")
        if not is_interim:
            self.capture_context_usage_for_conversation(conversation_id)
            if user_message_id:
                await self._report_phase(conversation_id, user_message_id, "idle")
        return SendResult(success=True, message_id=result.get("message_id", ""))

    async def send_draft(
        self,
        chat_id: str,
        draft_id: int,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> SendResult:
        conversation_id = _conversation_id(chat_id)
        user_message_id = self._peek_pending_message(conversation_id)
        try:
            await asyncio.to_thread(self._client.send_delta, conversation_id, content)
        except Exception as err:
            logger.warning(
                "draft delivery failed; Hermes may fall back to final send: %s",
                err,
            )
            return SendResult(success=False, message_id="")
        if user_message_id:
            await self._report_phase(conversation_id, user_message_id, "streaming")
        return SendResult(success=True, message_id="")

    async def send_exec_approval(
        self,
        chat_id: str,
        command: str,
        session_key: str,
        description: str = "dangerous command",
        metadata: dict[str, Any] | None = None,
        allow_permanent: bool = True,
        allow_session: bool = True,
        smart_denied: bool = False,
    ) -> SendResult:
        del metadata
        conversation_id = _conversation_id(chat_id)
        approval_id = str(uuid4())
        data = {
            "approval_id": approval_id,
            "command": command,
            "description": description,
            "session_key": session_key,
            "allow_permanent": allow_permanent,
            "allow_session": allow_session,
            "smart_denied": smart_denied,
            "timeout_ms": 300_000,
        }
        # Arm the resolver before publishing so an immediate Operator response
        # cannot race ahead of the local correlation entry.
        self._pending_approvals[approval_id] = (session_key, "")
        try:
            await asyncio.to_thread(
                deliver_pending,
                self._client,
                conversation_id,
                data,
            )
        except Exception as err:
            self._pending_approvals.pop(approval_id, None)
            logger.warning("register approval pending failed: %s", err)
            return SendResult(success=False, message_id="", error=str(err))
        return SendResult(success=True, message_id=approval_id)

    async def send_image(self, chat_id: str, image_url: str, caption: str | None = None, **kwargs: Any) -> SendResult:
        return await self._send_media(chat_id, image_url, caption)

    async def send_image_file(self, chat_id: str, file_path: str, caption: str | None = None, **kwargs: Any) -> SendResult:
        return await self._send_media(chat_id, file_path, caption)

    async def send_document(self, chat_id: str, file_path: str, caption: str | None = None, **kwargs: Any) -> SendResult:
        return await self._send_media(chat_id, file_path, caption)

    async def send_voice(self, chat_id: str, file_path: str, caption: str | None = None, **kwargs: Any) -> SendResult:
        return await self._send_media(chat_id, file_path, caption)

    async def send_video(self, chat_id: str, file_path: str, caption: str | None = None, **kwargs: Any) -> SendResult:
        return await self._send_media(chat_id, file_path, caption)

    async def get_chat_info(self, chat_id: str) -> dict[str, Any]:
        return {"name": chat_id, "type": "dm", "chat_id": chat_id}

    def remember_session(
        self,
        session_id: str,
        *,
        model: str | None = None,
        provider: str | None = None,
    ) -> None:
        if not session_id:
            return
        conversation_id = self._session_to_conversation.get(session_id) or self._last_conversation
        if conversation_id:
            previous_session = self._conversation_to_session.get(conversation_id)
            if previous_session and previous_session != session_id:
                self._context_usage.pop(conversation_id, None)
            self._session_to_conversation[session_id] = conversation_id
            self._conversation_to_session[conversation_id] = session_id
            self._bind_approval_session(session_id, conversation_id)
        if model or provider:
            current = self._session_model.get(session_id, (None, None))
            self._session_model[session_id] = (
                model or current[0],
                provider or current[1],
            )

    def session_is_ours(self, session_id: str | None) -> bool:
        return bool(session_id and session_id in self._session_to_conversation)

    def conversation_for_session(self, session_id: str | None = None) -> str | None:
        if session_id:
            conversation_id = self._session_to_conversation.get(session_id)
            if conversation_id:
                return conversation_id
        return self._last_conversation

    def report_hook_phase(
        self,
        session_id: str,
        phase: str,
        detail: str | None = None,
    ) -> None:
        conversation_id = self._session_to_conversation.get(session_id)
        if not conversation_id:
            return
        message_id = self._peek_pending_message(conversation_id)
        if not message_id:
            return
        if not self._claim_once_phase(conversation_id, message_id, phase):
            return
        self.schedule(self._report_phase(conversation_id, message_id, phase, detail))

    def note_dispatch_error(self, error: str) -> None:
        self.schedule(
            self._report_runtime_status(
                connected=True,
                ready=True,
                last_dispatch_error=error,
            )
        )

    def capture_context_usage_for_conversation(self, conversation_id: str) -> None:
        context_usage_with_cache(
            self._context_usage,
            conversation_id,
            live_context_usage(
                self.gateway_runner,
                self._conversation_to_gateway_session.get(conversation_id),
                conversation_id,
            ),
        )

    def schedule(self, coro: Any) -> None:
        loop = self._loop
        if loop is None or not loop.is_running():
            return
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            loop.create_task(coro)
            return
        asyncio.run_coroutine_threadsafe(coro, loop)

    async def _send_media(
        self,
        chat_id: str,
        path_or_url: str,
        caption: str | None,
    ) -> SendResult:
        conversation_id = _conversation_id(chat_id)
        user_message_id = self._pop_pending_message(conversation_id)
        local_path = (path_or_url or "").strip()
        if not local_path:
            return SendResult(success=False, message_id="")
        if is_remote_url(local_path):
            error = f"remote media is not supported: {local_path}"
            await self._fail_turn(conversation_id, user_message_id, error)
            return SendResult(success=False, message_id="")
        try:
            result = await asyncio.to_thread(
                send_outbound_files,
                self._client,
                conversation_id,
                [local_path],
                caption,
            )
        except Exception as err:
            await self._fail_turn(conversation_id, user_message_id, err)
            return SendResult(success=False, message_id="")
        if user_message_id:
            await self._report_phase(conversation_id, user_message_id, "idle")
        return SendResult(success=True, message_id=result.get("message_id", ""))

    def _peek_pending_message(self, conversation_id: str) -> str | None:
        pending = self._pending_user_messages.get(conversation_id)
        return pending[0] if pending else None

    def _pop_pending_message(self, conversation_id: str) -> str | None:
        pending = self._pending_user_messages.get(conversation_id)
        if not pending:
            return None
        user_message_id = pending.popleft()
        if not pending:
            self._pending_user_messages.pop(conversation_id, None)
        return user_message_id

    def _tasks_running(self) -> bool:
        tasks = (self._task, self._management_task, self._approval_task)
        return all(task is not None and not task.done() for task in tasks)

    async def _stop_background_tasks(self) -> None:
        tasks = [task for task in (self._task, self._management_task, self._approval_task) if task]
        for task in tasks:
            task.cancel()
        for task in tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._task = None
        self._management_task = None
        self._approval_task = None

    async def _fail_turn(
        self,
        conversation_id: str,
        message_id: str | None,
        err: Exception | str,
    ) -> None:
        detail = str(err)
        if message_id:
            self._turn_failed.add(f"{conversation_id}:{message_id}")
            await self._report_phase(conversation_id, message_id, "failed", detail)
        await self._report_runtime_status(
            connected=True,
            ready=True,
            last_dispatch_error=detail,
        )

    async def _report_phase(
        self,
        conversation_id: str,
        message_id: str,
        phase: str,
        detail: str | None = None,
    ) -> None:
        key = f"{conversation_id}:{message_id}"
        if phase == "failed":
            self._turn_failed.add(key)
        elif phase == "idle":
            self._once_phases.pop(key, None)
            if key in self._turn_failed:
                return
            self._turn_failed.discard(key)
        try:
            await asyncio.to_thread(
                self._client.report_turn_phase,
                conversation_id,
                message_id,
                phase,
                detail,
            )
        except Exception as err:
            logger.warning("report_turn_phase failed: %s", err)

    async def _report_runtime_status(
        self,
        *,
        connected: bool,
        ready: bool,
        reason: str | None = None,
        last_dispatch_error: str | None = None,
    ) -> None:
        try:
            await asyncio.to_thread(
                self._client.update_runtime_status,
                connected=connected,
                agent_runtime_ready=ready,
                reason=reason,
                last_dispatch_error=last_dispatch_error if last_dispatch_error is not None else (
                    None if ready else reason
                ),
                host_version=host_version(),
                adapter_version=adapter_version(),
            )
        except Exception as err:
            logger.warning("runtime status update failed: %s", err)

    def _claim_once_phase(
        self,
        conversation_id: str,
        message_id: str,
        phase: str,
    ) -> bool:
        if phase not in ("streaming", "thinking"):
            return True
        seen = self._once_phases.setdefault(f"{conversation_id}:{message_id}", set())
        if phase in seen:
            return False
        seen.add(phase)
        return True

    async def _poll_forever(self, factory, handler, name: str) -> None:
        delay = 2.0
        while True:
            try:
                async for event in _async_iter(factory):
                    delay = 2.0
                    await handler(event)
                # Clean stream EOF (no exception): the sidecar restarted with
                # an empty stream. Sleep before reopening so we do not
                # hot-loop on a stream that ends immediately again.
                logger.info("%s stream ended cleanly; reopening shortly", name)
                await asyncio.sleep(delay)
                delay = min(delay * 2.0, 30.0)
            except asyncio.CancelledError:
                raise
            except RuntimeMismatchError as err:
                # Pair switched to another runtime under us: the sidecar 409s
                # every claim. Retrying is pointless; surface once and stop.
                logger.error("%s poll stopped: %s", name, err)
                # Best-effort only: when the mismatch is on this sidecar, the
                # status PUT itself 409s. The reason stays visible here in the
                # adapter logs, which is the reliable surface.
                await self._report_runtime_status(
                    connected=False,
                    ready=False,
                    reason=str(err),
                )
                return
            except Exception as err:
                logger.warning("%s poll error: %s", name, err)
                await asyncio.sleep(delay)
                delay = min(delay * 2.0, 30.0)

    async def _handle_management_request(self, event: dict[str, Any]) -> None:
        request_id = event.get("request_id")
        operation = event.get("operation")
        if not request_id or not operation:
            return
        runtime_name = event.get("runtime_name")
        if runtime_name not in (None, "", AGENT_RUNTIME_NAME):
            return
        conversation_id = str(event.get("conversation_id") or "")
        try:
            if operation == "status":
                await asyncio.to_thread(self._sync_status_result, request_id, conversation_id)
                return
            if operation == "models":
                await asyncio.to_thread(self._sync_models_result, request_id)
                return
            if operation == "set_model":
                model = str(event.get("model") or "").strip()
                await self._set_model_result(
                    request_id,
                    conversation_id,
                    model,
                )
                return
            await asyncio.to_thread(
                self._client.post_management_result,
                request_id,
                ok=False,
                error={
                    "code": "invalid_request",
                    "message": f"unknown management operation: {operation}",
                },
            )
        except ManagementError as err:
            await asyncio.to_thread(
                self._client.post_management_result,
                request_id,
                ok=False,
                error={"code": err.code, "message": err.message},
            )
        except Exception as err:
            logger.warning("management %s failed: %s", operation, err)
            try:
                await asyncio.to_thread(
                    self._client.post_management_result,
                    request_id,
                    ok=False,
                    error={"code": "hermes_error", "message": str(err)},
                )
            except Exception as post_err:
                logger.warning("management result post failed: %s", post_err)

    def _status_payload(self, conversation_id: str) -> dict[str, Any]:
        session_key = self._conversation_to_session.get(conversation_id)
        model, provider = (None, None)
        if session_key:
            model, provider = self._session_model.get(session_key, (None, None))
        status = build_status(
            conversation_id=conversation_id,
            session_key=session_key,
            session_model=model,
            session_provider=provider,
            host_version=host_version(),
            adapter_version=adapter_version(),
        )
        status.update(
            context_usage_with_cache(
                self._context_usage,
                conversation_id,
                live_context_usage(
                    self.gateway_runner,
                    self._conversation_to_gateway_session.get(conversation_id),
                    conversation_id,
                ),
            ),
        )
        return status

    def _sync_status_result(self, request_id: str, conversation_id: str) -> None:
        payload = self._status_payload(conversation_id)
        logger.debug(
            "management status conversation=%s session=%s gateway_session=%s "
            "context_used=%s context_window=%s context_percent=%s",
            conversation_id,
            payload.get("session_key"),
            self._conversation_to_gateway_session.get(conversation_id),
            payload.get("context_used_tokens"),
            payload.get("context_window_tokens"),
            payload.get("context_usage_percent"),
        )
        self._client.post_management_result(
            request_id, ok=True, result=payload
        )

    def _sync_models_result(self, request_id: str) -> None:
        result = list_hermes_models()
        providers = sorted(
            {
                str(model.get("provider"))
                for model in result.get("models", [])
                if isinstance(model, dict) and model.get("provider")
            }
        )
        logger.debug(
            "management models count=%s providers=%s",
            len(result.get("models", [])),
            ",".join(providers),
        )
        self._client.post_management_result(
            request_id, ok=True, result=result
        )

    async def _set_model_result(
        self, request_id: str, conversation_id: str, model: str
    ) -> None:
        command = session_model_command(model)
        logger.info(
            "management set_model requested conversation=%s selector=%s",
            conversation_id,
            model,
        )
        gateway_session_key = self._conversation_to_gateway_session.get(conversation_id)
        if not gateway_session_key:
            raise ManagementError(
                "invalid_request",
                "no Hermes session is bound to this conversation yet; send a message first",
            )
        handler = getattr(self.gateway_runner, "_handle_model_command", None)
        if not callable(handler):
            raise ManagementError(
                "runtime_unavailable",
                "this Hermes version does not expose the gateway model switch",
            )
        event = _message_event(
            adapter=self,
            conversation_id=conversation_id,
            message_id=request_id,
            text=command,
            media_urls=[],
        )
        overrides = getattr(self.gateway_runner, "_session_model_overrides", {}) or {}
        previous = overrides.get(gateway_session_key)
        previous = dict(previous) if isinstance(previous, dict) else None
        reply = await handler(event)
        overrides = getattr(self.gateway_runner, "_session_model_overrides", {}) or {}
        switched = overrides.get(gateway_session_key)
        if not isinstance(switched, dict):
            raise ManagementError(
                "hermes_error",
                str(reply or "Hermes did not apply the model switch"),
            )

        selected = str(switched.get("model") or "").strip()
        provider = str(switched.get("provider") or "").strip()
        if not selected:
            raise ManagementError(
                "hermes_error",
                str(reply or "Hermes returned an empty model after switching"),
            )
        if str(reply or "").lstrip().startswith("❌"):
            raise ManagementError("hermes_error", str(reply))
        requested = model.strip()
        expected_provider = model_selector_provider(requested)
        if expected_provider and provider != expected_provider:
            raise ManagementError(
                "hermes_error",
                "Hermes applied the model on an unexpected provider "
                f"(expected {expected_provider}, got {provider or 'unknown'})",
            )
        selected_ids = {selected, f"{provider}:{selected}" if provider else selected}
        if previous == switched and requested not in selected_ids:
            raise ManagementError(
                "hermes_error",
                str(reply or "Hermes did not apply the requested model switch"),
            )
        agent_session_key = self._conversation_to_session.get(conversation_id)
        if agent_session_key:
            self._session_model[agent_session_key] = (selected, provider or None)
        result = {
            "session_key": agent_session_key or gateway_session_key,
            "model": selected,
            "model_provider": provider or None,
            "scope": "session",
        }
        logger.info(
            "management set_model applied conversation=%s provider=%s model=%s",
            conversation_id,
            provider or "-",
            selected,
        )
        await asyncio.to_thread(
            self._client.post_management_result,
            request_id,
            ok=True,
            result={**result, **self._status_payload(conversation_id)},
        )

    async def _handle_approval_event(self, event: dict[str, Any]) -> None:
        apply_plugin_event(
            event,
            self._pending_approvals,
            mark_resolved=self._client.mark_approval_resolved,
        )

    def _bind_approval_session(self, session_key: str, conversation_id: str) -> None:
        if not session_key or session_key in self._approval_sessions:
            return

        def _notify(approval_data: dict[str, Any]) -> None:
            self.schedule(self._deliver_approval(conversation_id, session_key, approval_data))

        if register_notify(session_key, _notify):
            self._approval_sessions.add(session_key)

    async def _deliver_approval(
        self,
        conversation_id: str,
        session_key: str,
        approval_data: dict[str, Any],
    ) -> None:
        try:
            approval_id = await asyncio.to_thread(
                deliver_pending,
                self._client,
                conversation_id,
                approval_data,
            )
        except Exception as err:
            logger.warning("register approval pending failed: %s", err)
            return
        request_id = str(
            approval_data.get("request_id")
            or approval_data.get("approval_id")
            or approval_id
        )
        self._pending_approvals[approval_id] = (session_key, request_id)
        try:
            poll = await asyncio.to_thread(
                self._client.get_approval_decision,
                approval_id,
            )
        except Exception as err:
            logger.warning("approval decision poll failed: %s", err)
            return
        if poll.get("status") == "decided":
            apply_plugin_event(
                {
                    "type": "approval_decided",
                    "approval_id": approval_id,
                    "decision": poll.get("decision"),
                },
                self._pending_approvals,
                mark_resolved=self._client.mark_approval_resolved,
            )

    async def _handle_inbound_event(self, event: Any) -> None:
        if isinstance(event, InboundUserMessage):
            if event.message_id in self._dispatched_message_ids:
                await self._claim_turn(
                    event.conversation_id,
                    event.message_id,
                    [event.seq] if event.seq else [],
                )
                return
            ready = self._inbound.on_user_message(event)
            if ready is None:
                # Hold the user_message seq until attachments are terminal.
                # Sidecar already emitted waiting_attachments; do not cover with working.
                return
            await self._dispatch_ready(ready)
            return
        if isinstance(event, InboundAttachmentEvent):
            ready = self._inbound.on_attachment(event)
            if ready is None:
                # Keep the user_message seq until every attachment is terminal,
                # then accept that whole seq set in one claim.
                return
            if not ready.parts and ready.ack_seqs and not ready.text:
                if ready.message_id in self._dispatched_message_ids:
                    await self._claim_turn(
                        ready.conversation_id,
                        ready.message_id,
                        ready.ack_seqs,
                    )
                return
            await self._dispatch_ready(ready)

    async def _dispatch_ready(self, ready: Any) -> None:
        if ready.message_id in self._dispatched_message_ids:
            await self._claim_turn(
                ready.conversation_id,
                ready.message_id,
                ready.ack_seqs,
            )
            return
        decision = await self._claim_turn(
            ready.conversation_id,
            ready.message_id,
            ready.ack_seqs,
        )
        if decision == "duplicate":
            self._dispatched_message_ids.add(ready.message_id)
            return
        if decision != "accepted":
            return
        self._dispatched_message_ids.add(ready.message_id)
        pending = self._pending_user_messages.setdefault(ready.conversation_id, deque())
        pending.append(ready.message_id)
        self._last_conversation = ready.conversation_id
        await self._report_phase(ready.conversation_id, ready.message_id, "working")
        event = _message_event(
            adapter=self,
            conversation_id=ready.conversation_id,
            message_id=ready.message_id,
            text=ready.text,
            media_urls=ready.media_urls,
        )
        self._remember_gateway_session(ready.conversation_id, event)
        try:
            await self.handle_message(event)
        except Exception as err:
            await self._fail_turn(ready.conversation_id, ready.message_id, err)
            return

    async def _claim_turn(
        self,
        conversation_id: str,
        message_id: str,
        seqs: list[str],
    ) -> str:
        try:
            payload = await asyncio.to_thread(
                self._client.accept_inbound,
                conversation_id,
                message_id,
                seqs,
            )
        except Exception as err:
            # Fail closed on every accept failure, including a 404 from a
            # sidecar without the endpoint: do not start the turn; the
            # message stays queued and replays on reconnect.
            logger.warning("inbound accept failed: %s", err)
            return "rejected"
        return claim_turn_decision(payload if isinstance(payload, dict) else {})

    def _remember_gateway_session(
        self,
        conversation_id: str,
        event: MessageEvent,
    ) -> None:
        resolver = getattr(self.gateway_runner, "_session_key_for_source", None)
        if not callable(resolver) or event.source is None:
            return
        try:
            session_key = resolver(event.source)
        except Exception:
            return
        if session_key:
            self._conversation_to_gateway_session[conversation_id] = str(session_key)


def _conversation_id(chat_id: str) -> str:
    return chat_id.removeprefix("conversation:")


def _message_event(
    *,
    adapter: LanglangbotAdapter,
    conversation_id: str,
    message_id: str,
    text: str,
    media_urls: list[str],
) -> MessageEvent:
    kwargs: dict[str, Any] = {
        "text": text,
        "source": adapter.build_source(
            chat_id=f"conversation:{conversation_id}",
            chat_name=conversation_id,
            chat_type="dm",
            user_id="ios-operator",
            user_name="ios-operator",
            message_id=message_id,
        ),
        "user_id": "ios-operator",
        "user_name": "ios-operator",
        "message_id": message_id,
        "message_type": MessageType.TEXT,
        "metadata": {"conversation_id": conversation_id},
    }
    if media_urls:
        try:
            return MessageEvent(**kwargs, media_urls=media_urls)
        except TypeError:
            kwargs["metadata"] = {**kwargs["metadata"], "media_urls": media_urls}
    return MessageEvent(**kwargs)


async def _async_iter(factory):
    iterator = factory()
    if hasattr(iterator, "__aiter__"):
        async for item in iterator:
            yield item
        return
    exhausted = object()
    while True:
        item = await asyncio.to_thread(next, iterator, exhausted)
        if item is exhausted:
            return
        yield item


def register(ctx) -> None:
    ctx.register_platform(
        name=PLATFORM_NAME,
        label="LangLangBot",
        adapter_factory=lambda cfg: LanglangbotAdapter(cfg),
        check_fn=check_requirements,
        validate_config=validate_config,
        env_enablement_fn=env_enablement,
        apply_yaml_config_fn=apply_yaml_config,
        emoji="📱",
        platform_hint=(
            "You are chatting with a LangLang Operator via LangLangBot. "
            "Keep replies concise and actionable. Operator only renders reply "
            "text; restate any tool output the user asked for."
        ),
        max_message_length=8000,
    )
    register_hooks(ctx)
