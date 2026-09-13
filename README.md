# LangLangBot Hermes adapter

Hermes **platform plugin** for LangLangBot. Hermes consumes inbound Operator
chat and writes outbound replies; it is not an API Server, TUI gateway, or ACP
client.

## Install (production)

Sidecar + adapter (picks Hermes when `hermes` is on PATH and `openclaw` is not):

```bash
curl -fsSL https://optimatist.ai/langlangbot/install.sh | bash -s -- --runtime hermes
langlangbot pair ABC2-T9K4
```

Plugin only, once the sidecar is already installed:

```bash
hermes plugins install OptimatistAI/langlangbot-hermes --enable
hermes gateway restart
```

That clones the public plugin repo to `~/.hermes/plugins/langlangbot/`
(`plugin.yaml` `name: langlangbot`). Do not copy into `plugins/platforms/` —
that is Hermes' bundled layout, and `hermes plugins enable langlangbot` looks
for the flat directory.

## Config

```yaml
gateway:
  platforms:
    langlangbot:
      enabled: true
      sidecar_url: "https://127.0.0.1:9528"
      plugin_token: "optional-shared-token"
```

Or via environment (`env` wins over YAML):

```bash
export LANGLANGBOT_SIDECAR_URL=https://127.0.0.1:9528
export LANGLANGBOT_PLUGIN_TOKEN=optional-shared-token
```

Then `hermes gateway status` should list LangLangBot only when a sidecar URL is
configured. Pair the sidecar first with the short code from Operator
(`langlangbot pair ABC2-T9K4`); this plugin does not start it.

## Install (development)

From a langlangbot checkout, match the published layout:

```bash
mkdir -p ~/.hermes/plugins/langlangbot
cp -R packages/hermes/. ~/.hermes/plugins/langlangbot/
hermes plugins enable langlangbot
```

LangLangBot serves HTTPS by default. Loopback (`127.0.0.1` / `localhost`) skips
TLS certificate verification in this adapter (same idea as OpenClaw
`sidecarInsecureTls`). Use `http://` only when langlangbot runs with
`LANGLANGBOT_INSECURE_HTTP=1`.

## Flow

```
Operator app -> LangLangBot -> Hermes LanglangbotAdapter (inbound SSE)
Hermes AIAgent -> send / send_draft / media -> LangLangBot outbound -> Operator SSE
```

### Runtime status

On connect, the adapter reports `PUT /v1/plugin/runtime/status` with:

- `kind`: `hermes`
- `runtime_name`: `Hermes`
- `adapter_version` / best-effort `host_version`
- `connected` / `agent_runtime_ready`

### Management

The plugin answers the sidecar management bus in-process:

- `status` — `~/.hermes/config.yaml` `model.provider` / `model.default` plus the
  session model from `on_session_start`
- `models` — in-process Hermes model picker (not an empty HTTP `/v1/models`)
- `set_model` — session-scoped `/model <id>` (never `--global`)

### Agent turn phases

Hooks report `working` / `thinking` (reasoning deltas, requires
`plugins.stream_reasoning_deltas: true`) / `streaming` / `tool` / `idle` /
`failed`. Inbound messages with non-terminal attachments are held until
`attachment_ready` / `attachment_failed`; the sidecar `waiting_attachments`
phase is left in place.

### Approvals

Dangerous Hermes commands notify Operator via `POST /v1/approvals/pending`
(`kind=hermes.exec`). Operator `allow-once` / `allow-always` / `deny` map to
Hermes `once` / `always` / `deny`.
