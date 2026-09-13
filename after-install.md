# LangLangBot Hermes plugin

This plugin is a Hermes **platform adapter**. After install, pair the LangLangBot
sidecar with the short code from Operator (`langlangbot pair ABC2-T9K4`) if you
have not already.

Enablement writes `langlangbot` to `plugins.enabled`. Confirm the platform block:

```yaml
gateway:
  platforms:
    langlangbot:
      enabled: true
      sidecar_url: "https://127.0.0.1:9528"
```

Or:

```bash
export LANGLANGBOT_SIDECAR_URL=https://127.0.0.1:9528
```

Then:

```bash
hermes gateway restart
hermes gateway status
```
