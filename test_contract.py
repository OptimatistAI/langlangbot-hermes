"""Contract tests that do not require a Hermes install."""

from __future__ import annotations

import json
import os
import sys
import types
import unittest
from dataclasses import make_dataclass
from pathlib import Path
from unittest import mock

from approvals import (
    apply_plugin_event,
    approval_actions,
    map_operator_decision,
    pending_payload,
)
from config import apply_yaml_config, check_requirements, env_enablement
from inbound import InboundAttachmentBuffer
from management import (
    ManagementError,
    context_usage_with_cache,
    list_hermes_models,
    live_context_usage,
    model_selector_provider,
    parse_model_selector,
    session_model_command,
)
from sidecar import InboundAttachmentEvent, InboundUserMessage, parse_inbound_event
import hooks
import tools


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
