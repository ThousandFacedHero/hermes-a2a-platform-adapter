"""
In-memory A2A task store with TTL-based expiry.

Provides TaskStore for creating, retrieving, updating, canceling, listing,
and subscribing to A2A protocol tasks.
"""

import asyncio
import time
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from typing import AsyncIterator, Optional

# Task state constants
TASK_STATE_SUBMITTED = "TASK_STATE_SUBMITTED"
TASK_STATE_WORKING = "TASK_STATE_WORKING"
TASK_STATE_COMPLETED = "TASK_STATE_COMPLETED"
TASK_STATE_FAILED = "TASK_STATE_FAILED"
TASK_STATE_CANCELED = "TASK_STATE_CANCELED"
TASK_STATE_INPUT_REQUIRED = "TASK_STATE_INPUT_REQUIRED"
TASK_STATE_REJECTED = "TASK_STATE_REJECTED"
TASK_STATE_AUTH_REQUIRED = "TASK_STATE_AUTH_REQUIRED"

# Role constants
ROLE_USER = "ROLE_USER"
ROLE_AGENT = "ROLE_AGENT"

# Terminal states — no further transitions allowed
TERMINAL_STATES = {
    TASK_STATE_COMPLETED,
    TASK_STATE_FAILED,
    TASK_STATE_CANCELED,
    TASK_STATE_REJECTED,
}


def _now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _new_uuid() -> str:
    """Return a new UUID4 string."""
    return str(uuid.uuid4())


def _make_message(role: str, text: str) -> dict:
    return {
        "message_id": _new_uuid(),
        "role": role,
        "parts": [{"text": text}],
    }


def _make_artifact(text: str) -> dict:
    return {
        "artifact_id": _new_uuid(),
        "parts": [{"text": text}],
    }


class TaskStore:
    """
    In-memory store for A2A protocol tasks.

    Parameters
    ----------
    ttl_seconds:
        How long (in seconds) a task lives after creation. Tasks older than
        this are treated as expired. Default: 3600.
    max_entries:
        Maximum number of tasks held. When at capacity the oldest task is
        evicted before adding a new one. Default: 1000.
    """

    def __init__(self, ttl_seconds: int = 3600, max_entries: int = 1000) -> None:
        self._ttl = ttl_seconds
        self._max = max_entries
        # Maps task_id → {"task": dict, "created_at": float}
        self._store: OrderedDict[str, dict] = OrderedDict()
        # Maps task_id → list[asyncio.Queue]
        self._subscribers: dict[str, list[asyncio.Queue]] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_expired(self, entry: dict) -> bool:
        return (time.monotonic() - entry["created_at"]) > self._ttl

    def _get_entry(self, task_id: str) -> Optional[dict]:
        """Return the internal entry dict if present and not expired."""
        entry = self._store.get(task_id)
        if entry is None:
            return None
        if self._is_expired(entry):
            return None
        return entry

    def _notify_subscribers(self, task: dict) -> None:
        """Push a status-update event to all queued subscribers."""
        task_id = task["id"]
        queues = self._subscribers.get(task_id, [])
        event = {
            "task_id": task_id,
            "context_id": task["context_id"],
            "status": dict(task["status"]),
        }
        for q in queues:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create(self, context_id: str, message_text: str) -> dict:
        """
        Create a new task.

        Returns a Task dict with TASK_STATE_SUBMITTED status and the user
        message stored in history.
        """
        # Evict oldest entry if at capacity
        if len(self._store) >= self._max:
            oldest_id = next(iter(self._store))
            del self._store[oldest_id]
            self._subscribers.pop(oldest_id, None)

        task_id = _new_uuid()
        task: dict = {
            "id": task_id,
            "context_id": context_id,
            "status": {
                "state": TASK_STATE_SUBMITTED,
                "timestamp": _now_iso(),
            },
            "artifacts": [],
            "history": [_make_message(ROLE_USER, message_text)],
            "metadata": {},
        }
        self._store[task_id] = {
            "task": task,
            "created_at": time.monotonic(),
        }
        return task

    def get(self, task_id: str) -> Optional[dict]:
        """Return the task or None if not found / expired."""
        entry = self._get_entry(task_id)
        if entry is None:
            return None
        return entry["task"]

    def update_status(
        self,
        task_id: str,
        state: str,
        agent_message: Optional[str] = None,
    ) -> dict:
        """
        Update the task state.

        Optionally appends an agent message to history. If the new state is
        TASK_STATE_COMPLETED and agent_message is provided, also appends an
        artifact.

        Raises KeyError if the task is not found (or expired).
        """
        entry = self._get_entry(task_id)
        if entry is None:
            raise KeyError(task_id)

        task = entry["task"]
        task["status"] = {
            "state": state,
            "timestamp": _now_iso(),
        }

        if agent_message is not None:
            task["history"].append(_make_message(ROLE_AGENT, agent_message))
            if state == TASK_STATE_COMPLETED:
                task["artifacts"].append(_make_artifact(agent_message))

        self._notify_subscribers(task)
        return task

    def cancel(self, task_id: str) -> dict:
        """
        Cancel a task.

        Raises KeyError if not found. Raises ValueError if already in a
        terminal state.
        """
        entry = self._get_entry(task_id)
        if entry is None:
            raise KeyError(task_id)

        task = entry["task"]
        if task["status"]["state"] in TERMINAL_STATES:
            raise ValueError(
                f"Task {task_id} is already in terminal state "
                f"{task['status']['state']!r}"
            )

        task["status"] = {
            "state": TASK_STATE_CANCELED,
            "timestamp": _now_iso(),
        }
        self._notify_subscribers(task)
        return task

    def list(
        self,
        context_id: Optional[str] = None,
        status: Optional[str] = None,
        page_size: int = 50,
        page_token: str = "",
    ) -> dict:
        """
        Return a page of tasks.

        Parameters
        ----------
        context_id:
            Filter to tasks with this context_id.
        status:
            Filter to tasks with this status state.
        page_size:
            Maximum number of tasks to return per page.
        page_token:
            Task ID to start from (exclusive). Empty string means start from
            the beginning.

        Returns
        -------
        dict with keys:
            "tasks": list of Task dicts
            "next_page_token": task ID for the next page, or "" if done
        """
        # Build ordered list of non-expired tasks
        all_tasks = []
        for task_id, entry in self._store.items():
            if self._is_expired(entry):
                continue
            task = entry["task"]
            if context_id is not None and task["context_id"] != context_id:
                continue
            if status is not None and task["status"]["state"] != status:
                continue
            all_tasks.append(task)

        # Apply page_token cursor (start after the token task ID)
        if page_token:
            start_idx = None
            for i, task in enumerate(all_tasks):
                if task["id"] == page_token:
                    start_idx = i + 1
                    break
            if start_idx is None:
                start_idx = 0
            all_tasks = all_tasks[start_idx:]

        page = all_tasks[:page_size]
        remaining = all_tasks[page_size:]

        next_page_token = page[-1]["id"] if remaining else ""

        return {
            "tasks": page,
            "next_page_token": next_page_token,
        }

    async def subscribe(self, task_id: str) -> AsyncIterator[dict]:
        """
        Async generator that yields status-update events for a task.

        Raises ValueError if the task is already in a terminal state.
        Events have keys: task_id, context_id, status.
        The generator ends after yielding a terminal-state event.
        """
        # Validate task exists and is not terminal
        entry = self._get_entry(task_id)
        if entry is None:
            raise KeyError(task_id)

        task = entry["task"]
        if task["status"]["state"] in TERMINAL_STATES:
            raise ValueError(
                f"Task {task_id} is already in terminal state "
                f"{task['status']['state']!r}"
            )

        q: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._subscribers.setdefault(task_id, []).append(q)

        try:
            while True:
                event = await q.get()
                yield event
                if event["status"]["state"] in TERMINAL_STATES:
                    break
        finally:
            subs = self._subscribers.get(task_id, [])
            if q in subs:
                subs.remove(q)
            if not subs:
                self._subscribers.pop(task_id, None)

    def cleanup(self) -> None:
        """Remove all expired entries from the store."""
        expired_ids = [
            task_id
            for task_id, entry in list(self._store.items())
            if self._is_expired(entry)
        ]
        for task_id in expired_ids:
            del self._store[task_id]
            self._subscribers.pop(task_id, None)
