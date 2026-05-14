"""
Tests for handlers.py — A2A JSON-RPC request handlers.
Run from plugins/hermes-a2a/ directory:
    python -m pytest tests/test_handlers.py -v
"""

import asyncio
import pytest

from task_store import (
    TaskStore,
    TASK_STATE_WORKING,
    TASK_STATE_COMPLETED,
    TASK_STATE_FAILED,
    TASK_STATE_SUBMITTED,
)
from handlers import (
    handle_send_message,
    handle_get_task,
    handle_list_tasks,
    handle_cancel_task,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_store() -> TaskStore:
    return TaskStore()


async def agent_ok(user_text: str, context_id: str):
    """Mock invoke_agent that returns a fixed response."""
    return "Hello from agent"


async def agent_none(user_text: str, context_id: str):
    """Mock invoke_agent that returns None (empty response)."""
    return None


async def agent_error(user_text: str, context_id: str):
    """Mock invoke_agent that raises an exception."""
    raise RuntimeError("agent exploded")


def send_params(text: str, context_id: str = None) -> dict:
    """Build params for a SendMessage call."""
    msg = {"parts": [{"text": text}]}
    if context_id is not None:
        msg["context_id"] = context_id
    return {"message": msg}


# ---------------------------------------------------------------------------
# TestHandleSendMessage
# ---------------------------------------------------------------------------

class TestHandleSendMessage:
    @pytest.mark.asyncio
    async def test_creates_task_and_invokes_agent_returns_completed_with_artifacts(self):
        store = make_store()
        params = send_params("ping")

        resp = await handle_send_message("req-1", params, store, agent_ok)

        assert resp["jsonrpc"] == "2.0"
        assert resp["id"] == "req-1"
        task = resp["result"]["task"]
        assert task["status"]["state"] == TASK_STATE_COMPLETED
        assert len(task["artifacts"]) == 1
        assert task["artifacts"][0]["parts"][0]["text"] == "Hello from agent"

    @pytest.mark.asyncio
    async def test_returns_32602_error_on_empty_text(self):
        store = make_store()
        params = {"message": {"parts": []}}

        resp = await handle_send_message("req-2", params, store, agent_ok)

        assert "error" in resp
        assert resp["error"]["code"] == -32602

    @pytest.mark.asyncio
    async def test_handles_agent_failure_returns_failed(self):
        store = make_store()
        params = send_params("boom")

        resp = await handle_send_message("req-3", params, store, agent_error)

        assert resp["jsonrpc"] == "2.0"
        task = resp["result"]["task"]
        assert task["status"]["state"] == TASK_STATE_FAILED

    @pytest.mark.asyncio
    async def test_uses_provided_context_id(self):
        store = make_store()
        params = send_params("hello", context_id="my-ctx-123")

        resp = await handle_send_message("req-4", params, store, agent_ok)

        task = resp["result"]["task"]
        assert task["context_id"] == "my-ctx-123"

    @pytest.mark.asyncio
    async def test_generates_context_id_when_not_provided(self):
        store = make_store()
        params = send_params("hello")

        resp = await handle_send_message("req-5", params, store, agent_ok)

        task = resp["result"]["task"]
        assert task["context_id"] is not None
        assert len(task["context_id"]) == 36  # UUID4

    @pytest.mark.asyncio
    async def test_none_response_returns_failed(self):
        store = make_store()
        params = send_params("hi")

        resp = await handle_send_message("req-6", params, store, agent_none)

        task = resp["result"]["task"]
        assert task["status"]["state"] == TASK_STATE_FAILED


# ---------------------------------------------------------------------------
# TestHandleGetTask
# ---------------------------------------------------------------------------

class TestHandleGetTask:
    def test_returns_existing_task(self):
        store = make_store()
        task = store.create("ctx-a", "hello")
        task_id = task["id"]

        resp = handle_get_task("req-10", {"id": task_id}, store)

        assert resp["jsonrpc"] == "2.0"
        assert resp["result"]["id"] == task_id

    def test_returns_error_for_missing_task(self):
        store = make_store()

        resp = handle_get_task("req-11", {"id": "does-not-exist"}, store)

        assert "error" in resp
        assert resp["error"]["code"] == -32602

    def test_returns_error_for_missing_id_param(self):
        store = make_store()

        resp = handle_get_task("req-12", {}, store)

        assert "error" in resp
        assert resp["error"]["code"] == -32602


# ---------------------------------------------------------------------------
# TestHandleListTasks
# ---------------------------------------------------------------------------

class TestHandleListTasks:
    def test_returns_all_tasks(self):
        store = make_store()
        store.create("ctx-b", "task one")
        store.create("ctx-b", "task two")

        resp = handle_list_tasks("req-20", {}, store)

        assert resp["jsonrpc"] == "2.0"
        assert len(resp["result"]["tasks"]) == 2

    def test_filters_by_context_id(self):
        store = make_store()
        store.create("ctx-x", "in context")
        store.create("ctx-y", "other context")

        resp = handle_list_tasks("req-21", {"context_id": "ctx-x"}, store)

        tasks = resp["result"]["tasks"]
        assert len(tasks) == 1
        assert tasks[0]["context_id"] == "ctx-x"


# ---------------------------------------------------------------------------
# TestHandleCancelTask
# ---------------------------------------------------------------------------

class TestHandleCancelTask:
    def test_cancels_working_task(self):
        store = make_store()
        task = store.create("ctx-c", "work")
        task_id = task["id"]
        store.update_status(task_id, TASK_STATE_WORKING)

        resp = handle_cancel_task("req-30", {"id": task_id}, store)

        assert resp["jsonrpc"] == "2.0"
        assert resp["result"]["status"]["state"] == "TASK_STATE_CANCELED"

    def test_cancel_terminal_task_returns_error(self):
        store = make_store()
        task = store.create("ctx-d", "done")
        task_id = task["id"]
        store.update_status(task_id, TASK_STATE_COMPLETED, "done")

        resp = handle_cancel_task("req-31", {"id": task_id}, store)

        assert "error" in resp
        assert resp["error"]["code"] == -32602

    def test_cancel_missing_id_param_returns_error(self):
        store = make_store()

        resp = handle_cancel_task("req-32", {}, store)

        assert "error" in resp
        assert resp["error"]["code"] == -32602
