"""
Tests for task_store.py — in-memory A2A task store with TTL expiry.
Run from plugins/hermes-a2a/ directory:
    python -m pytest tests/test_task_store.py -v
"""

import asyncio
import time
import pytest
import pytest_asyncio

from task_store import (
    TaskStore,
    TASK_STATE_SUBMITTED,
    TASK_STATE_WORKING,
    TASK_STATE_COMPLETED,
    TASK_STATE_FAILED,
    TASK_STATE_CANCELED,
    TASK_STATE_REJECTED,
    ROLE_USER,
    ROLE_AGENT,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_store(**kwargs) -> TaskStore:
    return TaskStore(**kwargs)


# ---------------------------------------------------------------------------
# TestTaskStoreCreate
# ---------------------------------------------------------------------------

class TestTaskStoreCreate:
    def test_returns_task_dict(self):
        store = make_store()
        task = store.create("ctx-1", "hello")
        assert isinstance(task, dict)

    def test_has_uuid_id(self):
        store = make_store()
        task = store.create("ctx-1", "hello")
        assert "id" in task
        assert isinstance(task["id"], str)
        assert len(task["id"]) == 36  # UUID4 canonical form

    def test_context_id_stored(self):
        store = make_store()
        task = store.create("ctx-abc", "hello")
        assert task["context_id"] == "ctx-abc"

    def test_initial_state_submitted(self):
        store = make_store()
        task = store.create("ctx-1", "hello")
        assert task["status"]["state"] == TASK_STATE_SUBMITTED

    def test_status_has_timestamp(self):
        store = make_store()
        task = store.create("ctx-1", "hello")
        ts = task["status"]["timestamp"]
        assert isinstance(ts, str)
        assert "T" in ts  # ISO 8601

    def test_artifacts_starts_empty(self):
        store = make_store()
        task = store.create("ctx-1", "hello")
        assert task["artifacts"] == []

    def test_metadata_starts_empty(self):
        store = make_store()
        task = store.create("ctx-1", "hello")
        assert task["metadata"] == {}

    def test_history_has_user_message(self):
        store = make_store()
        task = store.create("ctx-1", "hello world")
        assert len(task["history"]) == 1
        msg = task["history"][0]
        assert msg["role"] == ROLE_USER
        assert msg["parts"] == [{"text": "hello world"}]

    def test_history_message_has_message_id(self):
        store = make_store()
        task = store.create("ctx-1", "hello")
        msg = task["history"][0]
        assert "message_id" in msg
        assert isinstance(msg["message_id"], str)
        assert len(msg["message_id"]) == 36

    def test_two_tasks_have_different_ids(self):
        store = make_store()
        t1 = store.create("ctx-1", "a")
        t2 = store.create("ctx-1", "b")
        assert t1["id"] != t2["id"]

    def test_task_retrievable_after_create(self):
        store = make_store()
        task = store.create("ctx-1", "hello")
        retrieved = store.get(task["id"])
        assert retrieved is not None
        assert retrieved["id"] == task["id"]


# ---------------------------------------------------------------------------
# TestTaskStoreGet
# ---------------------------------------------------------------------------

class TestTaskStoreGet:
    def test_returns_existing_task(self):
        store = make_store()
        task = store.create("ctx-1", "hi")
        result = store.get(task["id"])
        assert result is not None
        assert result["id"] == task["id"]

    def test_returns_none_for_nonexistent(self):
        store = make_store()
        result = store.get("no-such-id")
        assert result is None

    def test_returns_none_for_expired_task(self):
        store = make_store(ttl_seconds=0)
        task = store.create("ctx-1", "hi")
        # TTL=0 means expired immediately
        result = store.get(task["id"])
        assert result is None

    def test_returns_task_within_ttl(self):
        store = make_store(ttl_seconds=3600)
        task = store.create("ctx-1", "hi")
        result = store.get(task["id"])
        assert result is not None


# ---------------------------------------------------------------------------
# TestTaskStoreUpdateStatus
# ---------------------------------------------------------------------------

class TestTaskStoreUpdateStatus:
    def test_changes_state(self):
        store = make_store()
        task = store.create("ctx-1", "hi")
        updated = store.update_status(task["id"], TASK_STATE_WORKING)
        assert updated["status"]["state"] == TASK_STATE_WORKING

    def test_updates_timestamp(self):
        store = make_store()
        task = store.create("ctx-1", "hi")
        old_ts = task["status"]["timestamp"]
        time.sleep(0.01)
        updated = store.update_status(task["id"], TASK_STATE_WORKING)
        assert updated["status"]["timestamp"] >= old_ts

    def test_adds_agent_message_to_history(self):
        store = make_store()
        task = store.create("ctx-1", "hi")
        updated = store.update_status(task["id"], TASK_STATE_WORKING, agent_message="working on it")
        agent_msgs = [m for m in updated["history"] if m["role"] == ROLE_AGENT]
        assert len(agent_msgs) == 1
        assert agent_msgs[0]["parts"] == [{"text": "working on it"}]

    def test_agent_message_has_message_id(self):
        store = make_store()
        task = store.create("ctx-1", "hi")
        updated = store.update_status(task["id"], TASK_STATE_WORKING, agent_message="ok")
        agent_msgs = [m for m in updated["history"] if m["role"] == ROLE_AGENT]
        assert "message_id" in agent_msgs[0]
        assert len(agent_msgs[0]["message_id"]) == 36

    def test_no_agent_message_no_history_addition(self):
        store = make_store()
        task = store.create("ctx-1", "hi")
        updated = store.update_status(task["id"], TASK_STATE_WORKING)
        assert len(updated["history"]) == 1  # only the original user message

    def test_adds_artifact_on_completed_with_message(self):
        store = make_store()
        task = store.create("ctx-1", "hi")
        updated = store.update_status(task["id"], TASK_STATE_COMPLETED, agent_message="done!")
        assert len(updated["artifacts"]) == 1
        artifact = updated["artifacts"][0]
        assert "artifact_id" in artifact
        assert len(artifact["artifact_id"]) == 36
        assert artifact["parts"] == [{"text": "done!"}]

    def test_no_artifact_on_completed_without_message(self):
        store = make_store()
        task = store.create("ctx-1", "hi")
        updated = store.update_status(task["id"], TASK_STATE_COMPLETED)
        assert updated["artifacts"] == []

    def test_no_artifact_on_working_with_message(self):
        store = make_store()
        task = store.create("ctx-1", "hi")
        updated = store.update_status(task["id"], TASK_STATE_WORKING, agent_message="in progress")
        assert updated["artifacts"] == []

    def test_raises_keyerror_for_nonexistent(self):
        store = make_store()
        with pytest.raises(KeyError):
            store.update_status("no-such-id", TASK_STATE_WORKING)

    def test_get_reflects_updated_state(self):
        store = make_store()
        task = store.create("ctx-1", "hi")
        store.update_status(task["id"], TASK_STATE_COMPLETED, agent_message="done")
        retrieved = store.get(task["id"])
        assert retrieved["status"]["state"] == TASK_STATE_COMPLETED


# ---------------------------------------------------------------------------
# TestTaskStoreCancel
# ---------------------------------------------------------------------------

class TestTaskStoreCancel:
    def test_sets_canceled_state(self):
        store = make_store()
        task = store.create("ctx-1", "hi")
        result = store.cancel(task["id"])
        assert result["status"]["state"] == TASK_STATE_CANCELED

    def test_raises_keyerror_for_nonexistent(self):
        store = make_store()
        with pytest.raises(KeyError):
            store.cancel("no-such-id")

    def test_raises_valueerror_for_completed(self):
        store = make_store()
        task = store.create("ctx-1", "hi")
        store.update_status(task["id"], TASK_STATE_COMPLETED, agent_message="done")
        with pytest.raises(ValueError):
            store.cancel(task["id"])

    def test_raises_valueerror_for_failed(self):
        store = make_store()
        task = store.create("ctx-1", "hi")
        store.update_status(task["id"], TASK_STATE_FAILED)
        with pytest.raises(ValueError):
            store.cancel(task["id"])

    def test_raises_valueerror_for_rejected(self):
        store = make_store()
        task = store.create("ctx-1", "hi")
        store.update_status(task["id"], TASK_STATE_REJECTED)
        with pytest.raises(ValueError):
            store.cancel(task["id"])

    def test_raises_valueerror_for_already_canceled(self):
        store = make_store()
        task = store.create("ctx-1", "hi")
        store.cancel(task["id"])
        with pytest.raises(ValueError):
            store.cancel(task["id"])

    def test_cancel_reflects_in_get(self):
        store = make_store()
        task = store.create("ctx-1", "hi")
        store.cancel(task["id"])
        retrieved = store.get(task["id"])
        assert retrieved["status"]["state"] == TASK_STATE_CANCELED


# ---------------------------------------------------------------------------
# TestTaskStoreList
# ---------------------------------------------------------------------------

class TestTaskStoreList:
    def test_returns_all_tasks(self):
        store = make_store()
        store.create("ctx-1", "a")
        store.create("ctx-1", "b")
        store.create("ctx-1", "c")
        result = store.list()
        assert len(result["tasks"]) == 3

    def test_returns_dict_with_tasks_and_next_page_token(self):
        store = make_store()
        result = store.list()
        assert "tasks" in result
        assert "next_page_token" in result

    def test_empty_store(self):
        store = make_store()
        result = store.list()
        assert result["tasks"] == []
        assert result["next_page_token"] == ""

    def test_filter_by_context_id(self):
        store = make_store()
        store.create("ctx-A", "a")
        store.create("ctx-A", "b")
        store.create("ctx-B", "c")
        result = store.list(context_id="ctx-A")
        assert len(result["tasks"]) == 2
        for t in result["tasks"]:
            assert t["context_id"] == "ctx-A"

    def test_filter_by_status(self):
        store = make_store()
        t1 = store.create("ctx-1", "a")
        t2 = store.create("ctx-1", "b")
        store.create("ctx-1", "c")
        store.update_status(t1["id"], TASK_STATE_COMPLETED, agent_message="done")
        store.update_status(t2["id"], TASK_STATE_COMPLETED, agent_message="done")
        result = store.list(status=TASK_STATE_COMPLETED)
        assert len(result["tasks"]) == 2
        for t in result["tasks"]:
            assert t["status"]["state"] == TASK_STATE_COMPLETED

    def test_pagination_three_pages(self):
        store = make_store()
        for i in range(5):
            store.create("ctx-1", f"msg-{i}")

        # Page 1
        page1 = store.list(page_size=2)
        assert len(page1["tasks"]) == 2
        assert page1["next_page_token"] != ""

        # Page 2
        page2 = store.list(page_size=2, page_token=page1["next_page_token"])
        assert len(page2["tasks"]) == 2
        assert page2["next_page_token"] != ""

        # Page 3
        page3 = store.list(page_size=2, page_token=page2["next_page_token"])
        assert len(page3["tasks"]) == 1
        assert page3["next_page_token"] == ""

    def test_pagination_no_duplicates(self):
        store = make_store()
        for i in range(5):
            store.create("ctx-1", f"msg-{i}")

        ids_seen = set()
        token = ""
        while True:
            result = store.list(page_size=2, page_token=token)
            for t in result["tasks"]:
                assert t["id"] not in ids_seen
                ids_seen.add(t["id"])
            token = result["next_page_token"]
            if not token:
                break
        assert len(ids_seen) == 5

    def test_default_page_size_50(self):
        store = make_store()
        for i in range(60):
            store.create("ctx-1", f"msg-{i}")
        result = store.list()
        assert len(result["tasks"]) == 50
        assert result["next_page_token"] != ""


# ---------------------------------------------------------------------------
# TestTaskStoreCleanup
# ---------------------------------------------------------------------------

class TestTaskStoreCleanup:
    def test_removes_expired_tasks(self):
        store = make_store(ttl_seconds=0)
        store.create("ctx-1", "a")
        store.create("ctx-1", "b")
        store.cleanup()
        result = store.list()
        assert result["tasks"] == []

    def test_keeps_unexpired_tasks(self):
        store = make_store(ttl_seconds=3600)
        store.create("ctx-1", "a")
        store.create("ctx-1", "b")
        store.cleanup()
        result = store.list()
        assert len(result["tasks"]) == 2

    def test_max_entries_evicts_oldest(self):
        store = make_store(max_entries=3)
        t1 = store.create("ctx-1", "first")
        store.create("ctx-1", "second")
        store.create("ctx-1", "third")
        # Adding a 4th should evict t1 (oldest)
        store.create("ctx-1", "fourth")
        result = store.get(t1["id"])
        assert result is None

    def test_max_entries_newer_tasks_remain(self):
        store = make_store(max_entries=3)
        store.create("ctx-1", "first")
        t2 = store.create("ctx-1", "second")
        t3 = store.create("ctx-1", "third")
        t4 = store.create("ctx-1", "fourth")
        assert store.get(t2["id"]) is not None
        assert store.get(t3["id"]) is not None
        assert store.get(t4["id"]) is not None


# ---------------------------------------------------------------------------
# TestTaskStoreSubscribe
# ---------------------------------------------------------------------------

class TestTaskStoreSubscribe:
    @pytest.mark.asyncio
    async def test_receives_status_update_event(self):
        store = make_store()
        task = store.create("ctx-1", "hi")

        events = []

        async def collect():
            async for event in store.subscribe(task["id"]):
                events.append(event)
                break  # stop after first event

        collect_task = asyncio.create_task(collect())
        await asyncio.sleep(0)  # yield so collect_task starts

        store.update_status(task["id"], TASK_STATE_WORKING)
        await asyncio.sleep(0.05)

        collect_task.cancel()
        try:
            await collect_task
        except asyncio.CancelledError:
            pass

        assert len(events) == 1
        event = events[0]
        assert event["task_id"] == task["id"]
        assert event["context_id"] == task["context_id"]
        assert "status" in event
        assert event["status"]["state"] == TASK_STATE_WORKING

    @pytest.mark.asyncio
    async def test_subscribe_to_terminal_task_raises_valueerror(self):
        store = make_store()
        task = store.create("ctx-1", "hi")
        store.update_status(task["id"], TASK_STATE_COMPLETED, agent_message="done")

        with pytest.raises(ValueError):
            async for _ in store.subscribe(task["id"]):
                pass

    @pytest.mark.asyncio
    async def test_terminal_state_update_ends_subscription(self):
        store = make_store()
        task = store.create("ctx-1", "hi")

        events = []

        async def collect():
            async for event in store.subscribe(task["id"]):
                events.append(event)

        collect_task = asyncio.create_task(collect())
        await asyncio.sleep(0)

        store.update_status(task["id"], TASK_STATE_COMPLETED, agent_message="done")
        await asyncio.sleep(0.05)

        # collect_task should have finished naturally
        assert collect_task.done() or len(events) >= 1

        if not collect_task.done():
            collect_task.cancel()
            try:
                await collect_task
            except asyncio.CancelledError:
                pass

        assert len(events) >= 1
        last_event = events[-1]
        assert last_event["status"]["state"] == TASK_STATE_COMPLETED

    @pytest.mark.asyncio
    async def test_cancel_sends_event_to_subscriber(self):
        store = make_store()
        task = store.create("ctx-1", "hi")

        events = []

        async def collect():
            async for event in store.subscribe(task["id"]):
                events.append(event)

        collect_task = asyncio.create_task(collect())
        await asyncio.sleep(0)

        store.cancel(task["id"])
        await asyncio.sleep(0.05)

        if not collect_task.done():
            collect_task.cancel()
            try:
                await collect_task
            except asyncio.CancelledError:
                pass

        assert len(events) >= 1
        assert events[-1]["status"]["state"] == TASK_STATE_CANCELED
