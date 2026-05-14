"""
A2A Platform Adapter for Hermes Agent.

A plugin-based gateway adapter that exposes Hermes as a discoverable
A2A (Agent-to-Agent) agent via JSON-RPC 2.0 over HTTP.

Routes:
    GET  /health                       → health check
    GET  /.well-known/agent-card.json  → A2A agent card
    POST /a2a                          → JSON-RPC dispatch

Configuration via environment variables (config.extra fallback):
    A2A_ENABLED    — "true" / "1" / "yes" to activate
    A2A_PORT       — HTTP port (default 8645)
    A2A_AUTH_TOKEN — optional Bearer token for request auth
    A2A_AGENT_NAME — agent display name (default "Hermes Agent")
    A2A_AGENT_DESCRIPTION — agent description
    A2A_GATEWAY_URL — base URL for the agent card
    A2A_TASK_TTL   — task TTL in seconds (default 3600)
"""

import asyncio
import json
import logging
import os
import socket as _socket
import uuid
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Optional dependency — aiohttp may not be installed
try:
    from aiohttp import web
    AIOHTTP_AVAILABLE = True
except ImportError:
    web = None  # type: ignore[assignment]
    AIOHTTP_AVAILABLE = False

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)

try:
    from .agent_card import build_agent_card
    from .handlers import handle_cancel_task, handle_get_task, handle_list_tasks, handle_send_message
    from .jsonrpc import extract_user_text, make_error_response, parse_request, resolve_method
    from .task_store import TaskStore, TERMINAL_STATES
except ImportError:
    from agent_card import build_agent_card
    from handlers import handle_cancel_task, handle_get_task, handle_list_tasks, handle_send_message
    from jsonrpc import extract_user_text, make_error_response, parse_request, resolve_method
    from task_store import TaskStore, TERMINAL_STATES


# ---------------------------------------------------------------------------
# Module-level helpers for plugin registration
# ---------------------------------------------------------------------------

def check_requirements() -> bool:
    """Return True if aiohttp is available."""
    return AIOHTTP_AVAILABLE


def validate_config(config) -> bool:
    """Config is always valid — no required fields beyond A2A_ENABLED."""
    return True


def is_connected(config) -> bool:
    """Check whether A2A is enabled via environment."""
    return os.getenv("A2A_ENABLED", "").lower() in ("true", "1", "yes")


# ---------------------------------------------------------------------------
# A2A Adapter
# ---------------------------------------------------------------------------

class A2AAdapter(BasePlatformAdapter):
    """Async A2A adapter implementing the BasePlatformAdapter interface.

    Runs an aiohttp HTTP server with JSON-RPC 2.0 dispatch, streaming
    support (SSE), and an agent card endpoint.
    """

    def __init__(self, config, **kwargs):
        platform = Platform("a2a")
        super().__init__(config=config, platform=platform)

        extra = getattr(config, "extra", {}) or {}

        # Connection / identity settings (env vars override config.extra)
        self.port = int(
            os.getenv("A2A_PORT") or extra.get("port", 8645)
        )
        self.auth_token = (
            os.getenv("A2A_AUTH_TOKEN") or extra.get("auth_token", "")
        )
        self.agent_name = (
            os.getenv("A2A_AGENT_NAME")
            or extra.get("agent_name", "Hermes Agent")
        )
        self.agent_description = (
            os.getenv("A2A_AGENT_DESCRIPTION")
            or extra.get("agent_description", "")
        )
        self.gateway_url = (
            os.getenv("A2A_GATEWAY_URL")
            or extra.get("gateway_url", f"http://localhost:{self.port}")
        )
        task_ttl = int(
            os.getenv("A2A_TASK_TTL") or extra.get("task_ttl", 3600)
        )

        # Task store
        self.task_store = TaskStore(ttl_seconds=task_ttl)

        # Runtime state
        self._app: Optional[Any] = None
        self._runner: Optional[Any] = None
        self._site: Optional[Any] = None

    # ── Auth ─────────────────────────────────────────────────────────────

    def _validate_auth(self, request) -> bool:
        """Check Bearer token if A2A_AUTH_TOKEN is configured.

        Returns True if auth passes (or no token is required).
        """
        if not self.auth_token:
            return True
        auth_header = request.headers.get("Authorization", "")
        return auth_header == f"Bearer {self.auth_token}"

    # ── Connection lifecycle ─────────────────────────────────────────────

    async def connect(self) -> bool:
        """Start the aiohttp HTTP server with A2A routes."""
        if not AIOHTTP_AVAILABLE:
            logger.error("A2A: aiohttp is not installed")
            self._set_fatal_error(
                "missing_dependency",
                "aiohttp is required: pip install aiohttp",
                retryable=False,
            )
            return False

        # Check port availability
        sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        try:
            sock.bind(("0.0.0.0", self.port))
        except OSError as e:
            logger.error("A2A: port %d is already in use — %s", self.port, e)
            self._set_fatal_error(
                "port_in_use",
                f"Port {self.port} is already in use",
                retryable=False,
            )
            return False
        finally:
            sock.close()

        # Build aiohttp app
        self._app = web.Application()
        self._app.router.add_get("/health", self._handle_health)
        self._app.router.add_get(
            "/.well-known/agent-card.json", self._handle_agent_card
        )
        self._app.router.add_post("/a2a", self._handle_jsonrpc)

        # Start server
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, "0.0.0.0", self.port)
        await self._site.start()

        self._mark_connected()
        logger.info(
            "A2A: listening on port %d (agent: %s)",
            self.port,
            self.agent_name,
        )
        return True

    async def disconnect(self) -> None:
        """Stop the aiohttp server and clean up."""
        self._mark_disconnected()
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
        self._app = None
        self._site = None

    # ── Sending (no-op for A2A) ──────────────────────────────────────────

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """A2A does not send outbound messages — always succeeds."""
        return SendResult(success=True)

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Return minimal chat info for A2A sessions."""
        return {"name": chat_id, "type": "a2a"}

    # ── Agent invocation ─────────────────────────────────────────────────

    async def _invoke_agent(
        self, user_text: str, context_id: str
    ) -> Optional[str]:
        """Invoke the Hermes agent via the gateway message handler.

        Returns the agent's text response, or None on failure.
        """
        if not self._message_handler:
            logger.warning("A2A: no message handler set")
            return None

        source = self.build_source(
            chat_id=f"a2a:{context_id}",
            chat_name="a2a",
            chat_type="a2a",
            user_id=f"a2a:{context_id}",
            user_name="a2a-client",
        )

        event = MessageEvent(
            text=user_text,
            message_type=MessageType.TEXT,
            source=source,
            internal=True,
        )

        return await self._message_handler(event)

    # ── HTTP handlers ────────────────────────────────────────────────────

    async def _handle_health(self, request) -> Any:
        """GET /health — simple health check."""
        return web.json_response({"status": "ok"})

    async def _handle_agent_card(self, request) -> Any:
        """GET /.well-known/agent-card.json — return the A2A agent card."""
        if not self._validate_auth(request):
            return web.json_response(
                {"error": "Unauthorized"}, status=401
            )

        # Build tool definitions from the Hermes model tools
        tool_definitions = []
        try:
            import model_tools
            tool_definitions = model_tools.get_tool_definitions(quiet_mode=True)
        except Exception as e:
            logger.debug("A2A: could not load tool definitions: %s", e)

        card = build_agent_card(
            agent_name=self.agent_name,
            agent_description=self.agent_description,
            gateway_url=self.gateway_url,
            tool_definitions=tool_definitions,
        )
        return web.json_response(card)

    async def _handle_jsonrpc(self, request) -> Any:
        """POST /a2a — JSON-RPC 2.0 dispatch."""
        if not self._validate_auth(request):
            return web.json_response(
                {"error": "Unauthorized"}, status=401
            )

        try:
            body = await request.json()
        except Exception:
            return web.json_response(
                make_error_response(None, -32700, "Parse error"),
                status=200,
            )

        parsed = parse_request(body)
        req_id = parsed["id"]

        if parsed["error"] is not None:
            return web.json_response(
                make_error_response(
                    req_id,
                    parsed["error"]["code"],
                    parsed["error"]["message"],
                ),
                status=200,
            )

        method = resolve_method(parsed["method"])
        params = parsed["params"]

        if method is None:
            return web.json_response(
                make_error_response(req_id, -32601, "Method not found"),
                status=200,
            )

        # Dispatch to handlers
        if method == "SendMessage":
            result = await handle_send_message(
                req_id, params, self.task_store, self._invoke_agent
            )
            return web.json_response(result)

        if method == "SendStreamingMessage":
            return await self._handle_streaming(req_id, params, request)

        if method == "GetTask":
            result = handle_get_task(req_id, params, self.task_store)
            return web.json_response(result)

        if method == "ListTasks":
            result = handle_list_tasks(req_id, params, self.task_store)
            return web.json_response(result)

        if method == "CancelTask":
            result = handle_cancel_task(req_id, params, self.task_store)
            return web.json_response(result)

        if method == "SubscribeToTask":
            return await self._handle_subscribe(req_id, params, request)

        # Should not reach here — resolve_method already filters
        return web.json_response(
            make_error_response(req_id, -32601, "Method not found"),
            status=200,
        )

    # ── Streaming ────────────────────────────────────────────────────────

    async def _handle_streaming(self, req_id, params, request) -> Any:
        """Handle SendStreamingMessage — SSE response with status + artifact events."""
        user_text, context_id, _task_id = extract_user_text(params)

        if not user_text:
            return web.json_response(
                make_error_response(req_id, -32602, "Invalid params: no user text"),
            )

        if not context_id:
            context_id = str(uuid.uuid4())

        task = self.task_store.create(context_id, user_text)
        task_id = task["id"]

        # Prepare SSE response
        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
        await response.prepare(request)

        async def _send_event(event_data: dict) -> None:
            line = f"data: {json.dumps(event_data)}\n\n"
            await response.write(line.encode("utf-8"))

        # Send WORKING status update
        working_task = self.task_store.update_status(task_id, "TASK_STATE_WORKING")
        await _send_event({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "statusUpdate": {
                    "task_id": task_id,
                    "context_id": context_id,
                    "status": working_task["status"],
                },
            },
        })

        # Invoke agent
        try:
            response_text = await self._invoke_agent(user_text, context_id)
        except Exception as exc:
            failed_task = self.task_store.update_status(task_id, "TASK_STATE_FAILED", str(exc))
            await _send_event({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "statusUpdate": {
                        "task_id": task_id,
                        "context_id": context_id,
                        "status": failed_task["status"],
                    },
                },
            })
            await response.write_eof()
            return response

        if not response_text:
            failed_task = self.task_store.update_status(task_id, "TASK_STATE_FAILED")
            await _send_event({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "statusUpdate": {
                        "task_id": task_id,
                        "context_id": context_id,
                        "status": failed_task["status"],
                    },
                },
            })
            await response.write_eof()
            return response

        # Send artifact update
        completed_task = self.task_store.update_status(task_id, "TASK_STATE_COMPLETED", response_text)
        artifact = completed_task["artifacts"][-1] if completed_task["artifacts"] else {}
        artifact_id = artifact.get("artifact_id", str(uuid.uuid4()))
        await _send_event({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "artifactUpdate": {
                    "task_id": task_id,
                    "context_id": context_id,
                    "artifact": {
                        "artifact_id": artifact_id,
                        "parts": [{"text": response_text}],
                    },
                    "append": False,
                    "last_chunk": True,
                },
            },
        })

        # Send COMPLETED status update
        await _send_event({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "statusUpdate": {
                    "task_id": task_id,
                    "context_id": context_id,
                    "status": completed_task["status"],
                },
            },
        })

        await response.write_eof()
        return response

    async def _handle_subscribe(self, req_id, params, request) -> Any:
        """Handle SubscribeToTask — SSE stream of status updates for a task."""
        task_id = params.get("id")
        if not task_id:
            return web.json_response(
                make_error_response(req_id, -32602, "Invalid params: missing id"),
            )

        task = self.task_store.get(task_id)
        if task is None:
            return web.json_response(
                make_error_response(req_id, -32602, f"Task not found: {task_id}"),
            )

        # Check if already terminal
        if task["status"]["state"] in TERMINAL_STATES:
            return web.json_response(
                make_error_response(
                    req_id,
                    -32602,
                    f"Task {task_id} is already in terminal state "
                    f"{task['status']['state']}",
                ),
            )

        # Prepare SSE response
        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
        await response.prepare(request)

        # Subscribe and stream events
        try:
            async for event in self.task_store.subscribe(task_id):
                sse_data = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "statusUpdate": {
                            "task_id": event["task_id"],
                            "context_id": event["context_id"],
                            "status": event["status"],
                        },
                    },
                }
                line = f"data: {json.dumps(sse_data)}\n\n"
                await response.write(line.encode("utf-8"))

                # Stop streaming on terminal state
                if event["status"]["state"] in TERMINAL_STATES:
                    break
        except (KeyError, ValueError) as exc:
            error_data = {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32602, "message": str(exc)},
            }
            line = f"data: {json.dumps(error_data)}\n\n"
            await response.write(line.encode("utf-8"))

        await response.write_eof()
        return response


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------

def register(ctx):
    """Plugin entry point — called by the Hermes plugin system."""
    ctx.register_platform(
        name="a2a",
        label="A2A Protocol",
        adapter_factory=lambda cfg: A2AAdapter(cfg),
        check_fn=check_requirements,
        validate_config=validate_config,
        is_connected=is_connected,
        required_env=["A2A_ENABLED"],
        install_hint="pip install aiohttp",
        allowed_users_env="A2A_ALLOWED_USERS",
        allow_all_env="A2A_ALLOW_ALL_USERS",
        emoji="\U0001f517",
        pii_safe=True,
        allow_update_command=False,
        platform_hint=(
            "You are communicating via the A2A (Agent-to-Agent) protocol. "
            "The caller is another AI agent. Respond in plain text. "
            "Be precise and structured — the response will be parsed programmatically."
        ),
    )
