"""Contract tests that do not require a Hermes install."""

from __future__ import annotations

import asyncio
import dataclasses
import io
import json
import os
import sys
import types
import unittest
from dataclasses import make_dataclass
from pathlib import Path
from unittest import mock

from urllib import error

from approvals import (
    apply_plugin_event,
    approval_actions,
    map_operator_decision,
    pending_payload,
)
from config import apply_yaml_config, check_requirements, env_enablement
from inbound import InboundAttachmentBuffer, claim_turn_decision
from management import (
    ManagementError,
    context_usage_with_cache,
    list_hermes_models,
    live_context_usage,
    model_selector_provider,
    parse_model_selector,
    session_model_command,
)
from sidecar import (
    AGENT_RUNTIME_KIND,
    InboundAttachmentEvent,
    InboundUserMessage,
    LanglangbotSidecarClient,
    RuntimeMismatchError,
    parse_inbound_event,
)
import hooks
import tools

def _install_gateway_stubs() -> None:
    """Minimal gateway.* stubs so adapter tests run without a Hermes install."""
    if "gateway.config" in sys.modules:
        return
    gateway = types.ModuleType("gateway")
    gateway_config = types.ModuleType("gateway.config")

    class PlatformConfig:  # noqa: D401 - test stub
        pass

    gateway_config.PlatformConfig = PlatformConfig

    gateway_platforms = types.ModuleType("gateway.platforms")
    base = types.ModuleType("gateway.platforms.base")

    class BasePlatformAdapter:  # noqa: D401 - test stub
        def __init__(self, config: object, platform: object) -> None:
            self.config = config
            self.platform = platform

        def _mark_connected(self) -> None:
            self.connected = True

        def _mark_disconnected(self) -> None:
            self.connected = False

    class MessageType:
        TEXT = "text"

    @dataclasses.dataclass
    class MessageEvent:
        text: str = ""
        source: object = None
        user_id: str = ""
        user_name: str = ""
        message_id: str = ""
        message_type: object = None
        metadata: dict = dataclasses.field(default_factory=dict)
        media_urls: list = dataclasses.field(default_factory=list)

    class Platform:
        def __init__(self, name: str) -> None:
            self.name = name

    @dataclasses.dataclass
    class SendResult:
        success: bool
        message_id: str
        error: str | None = None

    base.BasePlatformAdapter = BasePlatformAdapter
    base.MessageEvent = MessageEvent
    base.MessageType = MessageType
    base.Platform = Platform
    base.SendResult = SendResult

    sys.modules.setdefault("gateway", gateway)
    sys.modules["gateway.config"] = gateway_config
    sys.modules["gateway.platforms"] = gateway_platforms
    sys.modules["gateway.platforms.base"] = base


try:
    from adapter import LanglangbotAdapter
except ImportError:
    _install_gateway_stubs()
    from adapter import LanglangbotAdapter  # type: ignore[no-redef]


class ConfigTests(unittest.TestCase):
    def test_check_requirements_false_without_url(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LANGLANGBOT_SIDECAR_URL", None)
            with mock.patch("config.load_hermes_config", return_value={}):
                self.assertFalse(check_requirements())

    def test_check_requirements_true_from_env(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"LANGLANGBOT_SIDECAR_URL": "https://127.0.0.1:9528"},
        ):
            self.assertTrue(check_requirements())

    def test_apply_yaml_config_maps_sidecar_and_token(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LANGLANGBOT_SIDECAR_URL", None)
            os.environ.pop("LANGLANGBOT_PLUGIN_TOKEN", None)
            extras = apply_yaml_config(
                {},
                {
                    "enabled": True,
                    "sidecar_url": "https://127.0.0.1:9528",
                    "plugin_token": "secret",
                },
            )
        self.assertEqual(
            extras,
            {
                "sidecar_url": "https://127.0.0.1:9528",
                "plugin_token": "secret",
            },
        )

    def test_env_wins_over_yaml(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "LANGLANGBOT_SIDECAR_URL": "https://env.example:1",
                "LANGLANGBOT_PLUGIN_TOKEN": "env-token",
            },
        ):
            extras = apply_yaml_config(
                {},
                {"sidecar_url": "https://yaml.example:1", "plugin_token": "yaml"},
            )
        self.assertIsNone(extras)

    def test_env_enablement_requires_url(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LANGLANGBOT_SIDECAR_URL", None)
            self.assertIsNone(env_enablement())


class InboundTests(unittest.TestCase):
    def test_parse_user_and_attachment_events(self) -> None:
        user = parse_inbound_event(
            "user_message",
            "1",
            {
                "conversation_id": "c1",
                "message_id": "m1",
                "text": "hi",
                "received_at": "2026-01-01T00:00:00Z",
                "parts": [
                    {
                        "type": "attachment",
                        "upload_id": "u1",
                        "status": "uploading",
                    }
                ],
            },
        )
        self.assertIsInstance(user, InboundUserMessage)
        assert isinstance(user, InboundUserMessage)
        self.assertTrue(user.awaiting_attachments())
        ready = parse_inbound_event(
            "attachment_ready",
            "2",
            {
                "conversation_id": "c1",
                "message_id": "m1",
                "upload_id": "u1",
                "local_path": "/tmp/a.png",
                "filename": "a.png",
                "status": "ready",
            },
        )
        self.assertIsInstance(ready, InboundAttachmentEvent)

    def test_delay_until_terminal_then_media_urls(self) -> None:
        buf = InboundAttachmentBuffer()
        delayed = buf.on_user_message(
            InboundUserMessage(
                conversation_id="c1",
                message_id="m1",
                text="see this",
                received_at="",
                seq="1",
                parts=[
                    {
                        "type": "attachment",
                        "upload_id": "u1",
                        "status": "uploading",
                    }
                ],
            )
        )
        self.assertIsNone(delayed)
        ready = buf.on_attachment(
            InboundAttachmentEvent(
                kind="ready",
                conversation_id="c1",
                upload_id="u1",
                message_id="m1",
                seq="2",
                payload={"local_path": "/tmp/a.png", "filename": "a.png"},
            )
        )
        self.assertIsNotNone(ready)
        assert ready is not None
        self.assertEqual(ready.media_urls, ["/tmp/a.png"])
        self.assertIn("ready for analysis", ready.text)
        self.assertEqual(ready.ack_seqs, ["1", "2"])

    def test_accept_claim_runs_a_turn_once(self) -> None:
        self.assertEqual(
            claim_turn_decision({"accepted": True, "duplicate": False}),
            "accepted",
        )
        self.assertEqual(
            claim_turn_decision({"accepted": False, "duplicate": True}),
            "duplicate",
        )
        self.assertEqual(claim_turn_decision({}), "rejected")

    def test_available_attachment_keeps_user_seq_until_terminal(self) -> None:
        buf = InboundAttachmentBuffer()
        self.assertIsNone(
            buf.on_user_message(
                InboundUserMessage(
                    conversation_id="c1",
                    message_id="m1",
                    text="see this",
                    received_at="",
                    seq="1",
                    parts=[
                        {
                            "type": "attachment",
                            "upload_id": "u1",
                            "status": "uploading",
                        }
                    ],
                )
            )
        )
        self.assertIsNone(
            buf.on_attachment(
                InboundAttachmentEvent(
                    kind="available",
                    conversation_id="c1",
                    upload_id="u1",
                    message_id="m1",
                    seq="2",
                    payload={"filename": "a.png"},
                )
            )
        )
        pending = buf._pending["c1:m1"]
        self.assertEqual(pending.pending_ack_seqs, ["1", "2"])

    def test_accept_duplicate_does_not_start_another_turn(self) -> None:
        self.assertEqual(
            claim_turn_decision({"accepted": True, "duplicate": False}),
            "accepted",
        )
        self.assertEqual(
            claim_turn_decision({"accepted": False, "duplicate": True}),
            "duplicate",
        )

    def test_attachment_batch_accepts_the_whole_seq_set(self) -> None:
        captured: dict[str, object] = {}

        class _Response:
            def __enter__(self) -> "_Response":
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def read(self) -> bytes:
                return b'{"accepted": true, "duplicate": false}'

        def _urlopen(req: object, timeout: float | None = None) -> _Response:
            captured["url"] = req.full_url  # type: ignore[attr-defined]
            captured["body"] = json.loads(req.data.decode("utf-8"))  # type: ignore[attr-defined]
            return _Response()

        client = LanglangbotSidecarClient("https://127.0.0.1:9538", "token")
        with mock.patch.object(client, "_urlopen", _urlopen):
            payload = client.accept_inbound("conv-1", "msg-1", ["1", "4", "5"])
        self.assertEqual(payload["accepted"], True)
        self.assertEqual(payload["duplicate"], False)
        self.assertTrue(str(captured["url"]).endswith("/v1/inbound/accept"))
        self.assertEqual(
            captured["body"],
            {
                "conversation_id": "conv-1",
                "message_id": "msg-1",
                "seqs": ["1", "4", "5"],
            },
        )
        self.assertEqual(claim_turn_decision(payload), "accepted")


class RuntimeKindHeaderTests(unittest.TestCase):
    """The adapter must identify itself on every sidecar request."""

    def _captured_request(self, **client_kwargs: object) -> object:
        captured: dict[str, object] = {}

        class _Response:
            def __enter__(self) -> "_Response":
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def read(self) -> bytes:
                return b'{"status": "ok"}'

        def _urlopen(req: object, timeout: float | None = None) -> _Response:
            captured["req"] = req
            return _Response()

        client = LanglangbotSidecarClient(
            "https://127.0.0.1:9538", "token", **client_kwargs
        )
        with mock.patch.object(client, "_urlopen", _urlopen):
            client.health()
        return captured["req"]

    def test_headers_carry_hermes_runtime_kind_when_set(self) -> None:
        req = self._captured_request(runtime_kind=AGENT_RUNTIME_KIND)
        self.assertEqual(
            req.headers.get("X-langlangbot-runtime-kind"),  # type: ignore[attr-defined]
            "hermes",
        )

    def test_headers_omit_runtime_kind_when_unset(self) -> None:
        req = self._captured_request()
        self.assertNotIn("X-langlangbot-runtime-kind", req.headers)  # type: ignore[attr-defined]

    def test_blank_runtime_kind_is_treated_as_unset(self) -> None:
        req = self._captured_request(runtime_kind="   ")
        self.assertNotIn("X-langlangbot-runtime-kind", req.headers)  # type: ignore[attr-defined]


class RuntimeMismatchTests(unittest.TestCase):
    """409 runtime_mismatch is a permanent rejection, never a retry."""

    def _http_error(self, payload: bytes) -> error.HTTPError:
        return error.HTTPError(
            url="https://127.0.0.1:9528/v1/inbound/accept",
            code=409,
            msg="Conflict",
            hdrs={},
            fp=io.BytesIO(payload),
        )

    def test_json_request_raises_typed_runtime_mismatch(self) -> None:
        client = LanglangbotSidecarClient("https://127.0.0.1:9538", "token")
        err = self._http_error(
            b'{"error": "runtime_mismatch", "paired_runtime": "openclaw", '
            b'"received_runtime": "hermes"}'
        )
        with mock.patch.object(client, "_urlopen", side_effect=err):
            with self.assertRaises(RuntimeMismatchError) as ctx:
                client.accept_inbound("c1", "m1", ["1"])
        self.assertEqual(ctx.exception.paired_runtime, "openclaw")
        self.assertEqual(ctx.exception.received_runtime, "hermes")
        self.assertIn("paired with openclaw", str(ctx.exception))

    def test_json_request_raises_typed_instance_mismatch(self) -> None:
        client = LanglangbotSidecarClient("https://127.0.0.1:9538", "token")
        err = self._http_error(
            b'{"error": "runtime_instance_mismatch", "paired_runtime": "hermes", '
            b'"active_pid": 4242, "received_pid": 9999}'
        )
        with mock.patch.object(client, "_urlopen", side_effect=err):
            with self.assertRaises(RuntimeMismatchError) as ctx:
                client.accept_inbound("c1", "m1", ["1"])
        self.assertEqual(ctx.exception.reason, "runtime_instance_mismatch")
        self.assertEqual(ctx.exception.active_pid, 4242)
        self.assertIn("4242", str(ctx.exception))

    def test_other_409_bodies_are_not_mismatch_errors(self) -> None:
        client = LanglangbotSidecarClient("https://127.0.0.1:9538", "token")
        err = self._http_error(b'{"error": "agent_turn_terminal"}')
        with mock.patch.object(client, "_urlopen", side_effect=err):
            with self.assertRaises(error.HTTPError) as ctx:
                client.accept_inbound("c1", "m1", ["1"])
        self.assertNotIsInstance(ctx.exception, RuntimeMismatchError)

    def test_sse_stream_raises_typed_runtime_mismatch(self) -> None:
        """The SSE endpoint 409s before upgrading; it must surface as permanent."""
        client = LanglangbotSidecarClient("https://127.0.0.1:9538", "token")
        err = self._http_error(
            b'{"error": "runtime_mismatch", "paired_runtime": "openclaw", '
            b'"received_runtime": "hermes"}'
        )
        with mock.patch.object(client, "_urlopen", side_effect=err):
            with self.assertRaises(RuntimeMismatchError) as ctx:
                list(client.stream_inbound())
        self.assertEqual(ctx.exception.paired_runtime, "openclaw")

    def test_sse_stream_other_409_bodies_are_not_mismatch_errors(self) -> None:
        client = LanglangbotSidecarClient("https://127.0.0.1:9538", "token")
        err = self._http_error(b'{"error": "internal"}')
        with mock.patch.object(client, "_urlopen", side_effect=err):
            with self.assertRaises(error.HTTPError) as ctx:
                list(client.stream_inbound())
        self.assertNotIsInstance(ctx.exception, RuntimeMismatchError)


class ConnectGateTests(unittest.IsolatedAsyncioTestCase):
    """connect() refuses a sidecar paired with another runtime."""

    def _adapter(self, health_payload: dict) -> tuple[object, mock.Mock]:
        adapter = object.__new__(LanglangbotAdapter)
        adapter._client = mock.Mock()
        adapter._client.health.return_value = health_payload
        adapter._task = None
        adapter._management_task = None
        adapter._approval_task = None
        adapter._tasks_running = mock.Mock(return_value=False)
        adapter._stop_background_tasks = mock.AsyncMock()
        adapter._report_runtime_status = mock.AsyncMock()
        adapter._mark_connected = mock.Mock()
        adapter._mark_disconnected = mock.Mock()
        return adapter, adapter._report_runtime_status

    async def test_connect_refuses_mismatched_paired_runtime(self) -> None:
        adapter, report = self._adapter({"status": "ok", "paired_runtime_kind": "openclaw"})
        with mock.patch("adapter.asyncio.get_running_loop", create=True):
            ok = await adapter.connect()
        self.assertFalse(ok)
        adapter._mark_disconnected.assert_called_once()
        adapter._mark_connected.assert_not_called()
        self.assertIsNone(adapter._task)
        # The refusal is reported so the Operator runtime bar shows it.
        report.assert_awaited_once()
        self.assertEqual(
            report.await_args.kwargs.get("reason"),
            "sidecar is paired with openclaw; re-pair with --runtime hermes",
        )

    async def test_connect_enters_poll_loops_on_matching_paired_runtime(self) -> None:
        adapter, _ = self._adapter({"status": "ok", "paired_runtime_kind": "hermes"})
        # _poll_forever is stubbed to a plain (non-coroutine) Mock so
        # create_task receives nothing to leak a RuntimeWarning about.
        with (
            mock.patch(
                "adapter.LanglangbotAdapter._poll_forever",
                mock.Mock(return_value=None),
            ),
            mock.patch(
                "adapter.asyncio.create_task",
                return_value=mock.Mock(done=lambda: False),
            ) as create_task,
        ):
            ok = await adapter.connect()
        self.assertTrue(ok)
        adapter._mark_connected.assert_called_once()
        self.assertEqual(create_task.call_count, 3)

    async def test_connect_keeps_legacy_behavior_without_paired_kind(self) -> None:
        adapter, _ = self._adapter({"status": "ok"})
        with (
            mock.patch(
                "adapter.LanglangbotAdapter._poll_forever",
                mock.Mock(return_value=None),
            ),
            mock.patch(
                "adapter.asyncio.create_task",
                return_value=mock.Mock(done=lambda: False),
            ),
        ):
            ok = await adapter.connect()
        self.assertTrue(ok)
        adapter._mark_connected.assert_called_once()


class PollStopTests(unittest.IsolatedAsyncioTestCase):
    """A mid-session 409 stops the poll loop instead of retrying forever."""

    def _adapter(self) -> object:
        return object.__new__(LanglangbotAdapter)

    async def test_poll_forever_stops_on_runtime_mismatch(self) -> None:
        adapter = self._adapter()
        adapter._report_runtime_status = mock.AsyncMock()
        seen: list[str] = []

        async def handler(_event: object) -> None:
            seen.append("event")

        def factory() -> object:
            raise RuntimeMismatchError("openclaw", AGENT_RUNTIME_KIND)

        with mock.patch("adapter.asyncio.sleep", new=_no_sleep):
            await asyncio.wait_for(
                adapter._poll_forever(factory, handler, "inbound"), timeout=2
            )
        self.assertEqual(seen, [])
        adapter._report_runtime_status.assert_awaited_once()
        self.assertIn(
            "paired with openclaw",
            adapter._report_runtime_status.await_args.kwargs.get("reason", ""),
        )

    async def test_poll_forever_still_retries_transient_errors(self) -> None:
        adapter = self._adapter()
        adapter._report_runtime_status = mock.AsyncMock()
        attempts = 0

        async def handler(_event: object) -> None:
            nonlocal attempts
            attempts += 1
            if attempts < 2:
                raise ConnectionError("transient")
            # Second success: end the loop through its own cancellation path.
            raise asyncio.CancelledError

        async def _fake_iter(_factory):  # noqa: ANN001
            yield object()

        _real_sleep = asyncio.sleep

        async def _quick_sleep(_delay: float) -> None:
            # Keep the retry yield real so CancelledError can surface.
            await _real_sleep(0.001)

        with (
            mock.patch("adapter.asyncio.sleep", new=_quick_sleep),
            mock.patch("adapter._async_iter", _fake_iter),
        ):
            # The loop re-raises CancelledError (that is its shutdown path);
            # the handler's second-attempt raise just ends the loop.
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(
                    adapter._poll_forever(lambda: None, handler, "inbound"), timeout=5
                )
        self.assertGreaterEqual(attempts, 2)


async def _no_sleep(_delay: float) -> None:
    return None


class ApprovalTests(unittest.TestCase):
    def test_map_operator_decisions(self) -> None:
        self.assertEqual(map_operator_decision("allow-once"), "once")
        self.assertEqual(map_operator_decision("allow-always"), "always")
        self.assertEqual(map_operator_decision("deny"), "deny")
        self.assertEqual(map_operator_decision("session"), "session")
        self.assertIsNone(map_operator_decision("nope"))

    def test_pending_payload_uses_hermes_kind(self) -> None:
        payload = pending_payload(
            {"request_id": "r1", "command": "rm -rf /"},
            "conv",
        )
        self.assertEqual(payload["kind"], "hermes.exec")
        self.assertEqual(payload["approval_id"], "r1")
        self.assertEqual(payload["conversation_id"], "conv")

    def test_approval_actions_follow_hermes_scope_flags(self) -> None:
        self.assertEqual(
            [action["decision"] for action in approval_actions({})],
            ["allow-once", "session", "allow-always", "deny"],
        )
        self.assertEqual(
            [
                action["decision"]
                for action in approval_actions(
                    {"smart_denied": True, "allow_permanent": True}
                )
            ],
            ["allow-once", "deny"],
        )

    def test_operator_decision_resolves_gateway_approval(self) -> None:
        pending = {"approval-1": ("gateway-session", "")}
        mark_resolved = mock.Mock()
        with mock.patch(
            "approvals.resolve_hermes_approval",
            return_value=1,
        ) as resolve:
            apply_plugin_event(
                {
                    "type": "approval_decided",
                    "approval_id": "approval-1",
                    "decision": "allow-once",
                },
                pending,
                mark_resolved=mark_resolved,
            )
        resolve.assert_called_once_with(
            "gateway-session",
            "once",
            request_id=None,
        )
        mark_resolved.assert_called_once_with("approval-1", "allow-once")
        self.assertNotIn("approval-1", pending)

    def test_unmatched_operator_decision_is_logged_and_ignored(self) -> None:
        with (
            self.assertLogs(
                "langlangbot.hermes.approvals",
                level="INFO",
            ) as logs,
            mock.patch("approvals.resolve_hermes_approval") as resolve,
        ):
            apply_plugin_event(
                {
                    "type": "approval_decided",
                    "approval_id": "synthetic:probe-1",
                    "decision": "allow-once",
                },
                {},
            )
        resolve.assert_not_called()
        self.assertIn(
            "ignoring unmatched approval decision id=synthetic:probe-1",
            "\n".join(logs.output),
        )


class HookTests(unittest.TestCase):
    def test_recoverable_stream_error_is_not_reported_as_failed(self) -> None:
        adapter = mock.Mock()
        hooks.bind_adapter(adapter)
        try:
            hooks.on_stream_end(
                session_id="session-1",
                turn_id="background-review",
                finished=False,
                error="[Errno 32] Broken pipe",
                surface="langlangbot",
            )
        finally:
            hooks.bind_adapter(None)
        adapter.report_hook_phase.assert_not_called()
        adapter.note_dispatch_error.assert_not_called()


class ToolTests(unittest.TestCase):
    def test_connection_tool_uses_active_conversation_and_returns_json(self) -> None:
        client = mock.Mock()
        client.get_plugin_connection_current.return_value = {"connected": True}
        with (
            mock.patch("tools.active_conversation_id", return_value="active-conversation"),
            mock.patch("tools._client", return_value=client),
        ):
            result = tools.langlangbot_connection_current({}, session_id="session-1")
        self.assertEqual(json.loads(result), {"connected": True})
        client.get_plugin_connection_current.assert_called_once_with("active-conversation")

    def test_runtime_status_tool_returns_json(self) -> None:
        with (
            mock.patch("tools.active_conversation_id", return_value="active-conversation"),
            mock.patch("tools.default_model_from_config", return_value=("model-1", "provider-1")),
            mock.patch(
                "tools.build_status",
                return_value={
                    "session_key": "session-1",
                    "model": "model-1",
                    "model_provider": "provider-1",
                    "measurement": "exact",
                },
            ),
        ):
            result = tools.langlangbot_operator_runtime_status(
                {},
                session_id="session-1",
            )
        self.assertEqual(json.loads(result)["model"], "model-1")


class ManagementTests(unittest.TestCase):
    def test_model_catalog_flattens_provider_rows_with_qualified_ids(self) -> None:
        inventory = types.ModuleType("hermes_cli.inventory")
        picker_context = make_dataclass(
            "PickerContext",
            [("excluded_providers", list)],
            frozen=True,
        )([])
        inventory.load_picker_context = mock.Mock(return_value=picker_context)
        inventory.build_model_options_payload = mock.Mock(
            return_value={
                "providers": [
                    {
                        "slug": "deepseek",
                        "name": "DeepSeek",
                        "models": ["shared-model", "deepseek-v4-flash"],
                    },
                    {
                        "slug": "kimi-coding-cn",
                        "name": "Kimi China",
                        "models": ["shared-model", "kimi-k3"],
                    },
                    {
                        "slug": "copilot",
                        "name": "GitHub Copilot",
                        "models": ["gpt-5"],
                    },
                    {
                        "slug": "opencode-free",
                        "name": "OpenCode Free",
                        "models": ["free-model"],
                    },
                ]
            }
        )
        hermes_cli = types.ModuleType("hermes_cli")
        hermes_cli.__path__ = []
        hermes_config = types.ModuleType("hermes_cli.config")
        hermes_config.load_config = mock.Mock(
            return_value={
                "model": {"provider": "deepseek"},
            }
        )
        hermes_config.load_env = mock.Mock(
            return_value={
                "DEEPSEEK_API_KEY": "configured",
                "KIMI_CN_API_KEY": "configured",
            }
        )
        hermes_auth = types.ModuleType("hermes_cli.auth")
        hermes_auth.PROVIDER_REGISTRY = {
            "deepseek": types.SimpleNamespace(
                api_key_env_vars=("DEEPSEEK_API_KEY",),
                auth_type="api_key",
            ),
            "kimi-coding-cn": types.SimpleNamespace(
                api_key_env_vars=("KIMI_CN_API_KEY",),
                auth_type="api_key",
            ),
            "copilot": types.SimpleNamespace(
                api_key_env_vars=("COPILOT_GITHUB_TOKEN",),
                auth_type="api_key",
            ),
            "opencode-free": types.SimpleNamespace(
                api_key_env_vars=(),
                auth_type="api_key",
            ),
        }
        hermes_auth.get_auth_status = mock.Mock(return_value={"logged_in": False})
        with (
            mock.patch.dict(
                sys.modules,
                {
                    "hermes_cli": hermes_cli,
                    "hermes_cli.auth": hermes_auth,
                    "hermes_cli.config": hermes_config,
                    "hermes_cli.inventory": inventory,
                },
            ),
            mock.patch(
                "management.default_model_from_config",
                return_value=("deepseek-v4-flash", "deepseek"),
            ),
        ):
            result = list_hermes_models()
        self.assertEqual(
            [model["id"] for model in result["models"]],
            [
                "deepseek:shared-model",
                "deepseek:deepseek-v4-flash",
                "kimi-coding-cn:shared-model",
                "kimi-coding-cn:kimi-k3",
            ],
        )
        self.assertEqual(result["models"][2]["provider"], "kimi-coding-cn")

    def test_session_model_command_is_explicitly_session_scoped(self) -> None:
        self.assertEqual(
            session_model_command("kimi-coding-cn:kimi-k3"),
            "/model --session --provider kimi-coding-cn kimi-k3",
        )
        self.assertEqual(
            session_model_command("custom:local:qwen3:32b"),
            "/model --session custom:local:qwen3:32b",
        )
        self.assertEqual(
            parse_model_selector("kimi-coding-cn:kimi-k3"),
            ("kimi-coding-cn", "kimi-k3"),
        )
        self.assertEqual(
            model_selector_provider("kimi-coding-cn:kimi-k3"),
            "kimi-coding-cn",
        )
        self.assertIsNone(model_selector_provider("custom:local:qwen3:32b"))
        with self.assertRaises(ManagementError):
            session_model_command(" ")

    def test_live_context_usage_reads_current_prompt_not_session_total(self) -> None:
        compressor = mock.Mock(
            last_prompt_tokens=60_000,
            context_length=120_000,
        )
        agent = mock.Mock(
            context_compressor=compressor,
            session_total_tokens=1_900_000,
        )
        runner = mock.Mock()
        runner._resident_agent_for.return_value = agent
        usage = live_context_usage(runner, "gateway-session")
        self.assertEqual(usage["context_used_tokens"], 60_000)
        self.assertEqual(usage["context_window_tokens"], 120_000)
        self.assertEqual(usage["context_usage_percent"], 50)
        runner._resident_agent_for.assert_called_once_with("gateway-session")

    def test_live_context_usage_omits_unknown_occupancy(self) -> None:
        runner = mock.Mock()
        runner._resident_agent_for.return_value = mock.Mock(
            context_compressor=mock.Mock(
                last_prompt_tokens=0,
                context_length=120_000,
            )
        )
        self.assertEqual(live_context_usage(runner, "gateway-session"), {})

    def test_live_context_usage_finds_conversation_in_agent_cache(self) -> None:
        compressor = mock.Mock(
            last_prompt_tokens=75_000,
            context_length=150_000,
        )
        agent = mock.Mock(context_compressor=compressor)
        runner = mock.Mock()
        runner._resident_agent_for.return_value = None
        runner._running_agents = {}
        runner._agent_cache_lock = None
        runner._agent_cache = {
            "agent:main:langlangbot:dm:conversation:conversation-1": (
                agent,
                "signature",
            )
        }
        usage = live_context_usage(
            runner,
            "wrong-key",
            "conversation-1",
        )
        self.assertEqual(usage["context_used_tokens"], 75_000)
        self.assertEqual(usage["context_usage_percent"], 50)

    def test_context_usage_cache_survives_resident_agent_eviction(self) -> None:
        cache: dict[str, dict] = {}
        live = {
            "context_used_tokens": 93_306,
            "context_window_tokens": 1_048_576,
            "context_usage_percent": 9,
            "context_measurement": "provider_usage",
        }
        self.assertEqual(
            context_usage_with_cache(cache, "conversation-1", live),
            live,
        )
        self.assertEqual(
            context_usage_with_cache(cache, "conversation-1", {}),
            live,
        )
        self.assertEqual(
            context_usage_with_cache(cache, "conversation-2", {}),
            {},
        )


class PackageLayoutTests(unittest.TestCase):
    def test_plugin_yaml_exists(self) -> None:
        root = Path(__file__).resolve().parent
        self.assertTrue((root / "plugin.yaml").is_file())
        self.assertTrue((root / "__init__.py").is_file())


if __name__ == "__main__":
    unittest.main()
