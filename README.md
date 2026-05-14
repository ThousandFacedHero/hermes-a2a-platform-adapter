# hermes-a2a ☤

A2A v1 platform adapter for [Hermes Agent](https://github.com/NousResearch/hermes-agent) — makes your Hermes instance discoverable and callable by other agents via the [A2A protocol](https://github.com/a2aproject/A2A). No forking, no core changes, just drop it in.

Full A2A v1 spec coverage: Agent Card, SendMessage, streaming, task management. 135 tests.

## Quick Start

```bash
cp -r hermes-a2a ~/.hermes/plugins/a2a-platform
```

Set `A2A_ENABLED=true` in `~/.hermes/.env` and restart the gateway. Your agent card is now at `/.well-known/agent-card.json`.

For Docker, overlay onto the stock image:

```dockerfile
FROM nousresearch/hermes-agent:latest
COPY hermes-a2a/ /opt/hermes/plugins/platforms/a2a/
```

`aiohttp` ships with the stock image — no extra installs needed.

## Configuration

Environment variables (or `~/.hermes/.env`):

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `A2A_ENABLED` | yes | — | `true` to activate |
| `A2A_PORT` | no | `8645` | HTTP listen port |
| `A2A_AUTH_TOKEN` | no | — | Bearer token for authentication |
| `A2A_AGENT_NAME` | no | `Hermes Agent` | Display name in Agent Card |
| `A2A_AGENT_DESCRIPTION` | no | — | Description in Agent Card |
| `A2A_GATEWAY_URL` | no | `http://localhost:{port}` | External URL for Agent Card |
| `A2A_TASK_TTL` | no | `3600` | Task expiry in seconds |

## How It Works

This is a standard Hermes [platform adapter](https://github.com/NousResearch/hermes-agent/blob/main/gateway/platforms/ADDING_A_PLATFORM.md) — A2A sits alongside Telegram, Discord, Slack as another way to reach your agent.

- `A2AAdapter` extends `BasePlatformAdapter`, runs an aiohttp server
- Inbound A2A messages route through the gateway's message handler like any other platform
- Hermes tools auto-populate as A2A skills in the Agent Card

```
Other agent → GET /.well-known/agent-card.json → discovers skills
Other agent → POST /a2a { "method": "SendMessage", ... } → Hermes processes → response
```

## Endpoints

| Path | Method | Description |
|------|--------|-------------|
| `/.well-known/agent-card.json` | GET | Agent Card discovery |
| `/a2a` | POST | JSON-RPC 2.0 dispatch |
| `/health` | GET | Health check |

## Supported Methods

| Method | HTTP Alias | Description |
|--------|------------|-------------|
| `SendMessage` | `message/send` | Synchronous message → task response |
| `SendStreamingMessage` | `message/stream` | SSE streaming response |
| `GetTask` | `tasks/get` | Retrieve task by ID |
| `ListTasks` | `tasks/list` | Filter and paginate tasks |
| `CancelTask` | `tasks/cancel` | Cancel an in-progress task |
| `SubscribeToTask` | `tasks/subscribe` | SSE stream of status updates |

## Testing

```bash
pip install pytest aiohttp
pytest tests/ -v
```

## Spec Compliance

Audited against `a2a.proto` (the normative A2A v1 spec):

- All 8 task states (SUBMITTED → WORKING → COMPLETED/FAILED/CANCELED/REJECTED + INPUT_REQUIRED, AUTH_REQUIRED)
- `supported_interfaces` with protocol binding and version (not the legacy `url` field)
- Messages use `message_id`, `ROLE_USER`/`ROLE_AGENT`, Part OneOf
- Artifacts include `artifact_id`, SSE events include timestamps
- JSON-RPC error codes: -32700, -32600, -32601, -32602

## License

MIT
