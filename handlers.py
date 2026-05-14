"""
A2A JSON-RPC request handlers.

Each handler is a standalone function (or coroutine) that receives
(req_id, params, task_store, ...) and returns a JSON-RPC response dict.
"""

import uuid
from typing import Awaitable, Callable, Dict, Optional

try:
    from .jsonrpc import extract_user_text, make_error_response, make_result_response
    from .task_store import TaskStore, TASK_STATE_WORKING, TASK_STATE_COMPLETED, TASK_STATE_FAILED
except ImportError:
    from jsonrpc import extract_user_text, make_error_response, make_result_response
    from task_store import TaskStore, TASK_STATE_WORKING, TASK_STATE_COMPLETED, TASK_STATE_FAILED

# Type alias for the agent invocation callable.
# Takes (user_text, context_id) and returns agent response text or None.
InvokeAgent = Callable[[str, str], Awaitable[Optional[str]]]


async def handle_send_message(
    req_id,
    params: Dict,
    task_store: TaskStore,
    invoke_agent: InvokeAgent,
) -> Dict:
    """Handle a SendMessage / message/send request.

    - Extracts user text from params.
    - Returns -32602 error if no text found.
    - Creates a task, transitions it to WORKING, invokes the agent.
    - On success sets COMPLETED with the agent response.
    - On empty response sets FAILED with "Empty response".
    - On exception sets FAILED with the error message.
    """
    user_text, context_id, _task_id = extract_user_text(params)

    if not user_text:
        return make_error_response(req_id, -32602, "Invalid params: no user text")

    # Generate context_id if not supplied
    if not context_id:
        context_id = str(uuid.uuid4())

    task = task_store.create(context_id, user_text)
    task_id = task["id"]
    task_store.update_status(task_id, TASK_STATE_WORKING)

    try:
        response_text = await invoke_agent(user_text, context_id)
    except Exception as exc:
        task = task_store.update_status(task_id, TASK_STATE_FAILED, str(exc))
        return make_result_response(req_id, {"task": task})

    if not response_text:
        task = task_store.update_status(task_id, TASK_STATE_FAILED, None)
        # Store failure reason in status message without an agent message artifact
        task["status"]["message"] = "Empty response"
        return make_result_response(req_id, {"task": task})

    task = task_store.update_status(task_id, TASK_STATE_COMPLETED, response_text)
    return make_result_response(req_id, {"task": task})


def handle_get_task(req_id, params: Dict, task_store: TaskStore) -> Dict:
    """Handle a GetTask / tasks/get request."""
    task_id = params.get("id")
    if not task_id:
        return make_error_response(req_id, -32602, "Invalid params: missing id")

    task = task_store.get(task_id)
    if task is None:
        return make_error_response(req_id, -32602, f"Task not found: {task_id}")

    return make_result_response(req_id, task)


def handle_list_tasks(req_id, params: Dict, task_store: TaskStore) -> Dict:
    """Handle a ListTasks / tasks/list request."""
    context_id = params.get("context_id")
    status = params.get("status")
    page_size = params.get("page_size", 50)
    page_token = params.get("page_token", "")

    result = task_store.list(
        context_id=context_id,
        status=status,
        page_size=page_size,
        page_token=page_token,
    )
    return make_result_response(req_id, result)


def handle_cancel_task(req_id, params: Dict, task_store: TaskStore) -> Dict:
    """Handle a CancelTask / tasks/cancel request."""
    task_id = params.get("id")
    if not task_id:
        return make_error_response(req_id, -32602, "Invalid params: missing id")

    try:
        task = task_store.cancel(task_id)
    except (KeyError, ValueError) as exc:
        return make_error_response(req_id, -32602, str(exc))

    return make_result_response(req_id, task)
