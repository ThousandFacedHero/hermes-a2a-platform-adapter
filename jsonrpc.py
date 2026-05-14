"""
JSON-RPC 2.0 dispatcher for A2A protocol.
Handles method resolution, request parsing, and response formatting.
"""

from typing import Any, Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

METHOD_ALIASES: Dict[str, str] = {
    "message/send": "SendMessage",
    "message/stream": "SendStreamingMessage",
    "tasks/get": "GetTask",
    "tasks/list": "ListTasks",
    "tasks/cancel": "CancelTask",
    "tasks/subscribe": "SubscribeToTask",
}

KNOWN_METHODS = {
    "SendMessage",
    "SendStreamingMessage",
    "GetTask",
    "ListTasks",
    "CancelTask",
    "SubscribeToTask",
}


# ---------------------------------------------------------------------------
# Request parsing
# ---------------------------------------------------------------------------

def parse_request(body: Dict) -> Dict:
    """Parse a JSON-RPC 2.0 request body.

    Returns a dict with keys:
        method  - the method string (or None on error)
        id      - the request id (may be None)
        params  - the params dict (defaults to {})
        error   - None on success, or {"code": int, "message": str} on failure
    """
    req_id = body.get("id")

    if body.get("jsonrpc") != "2.0":
        return {
            "method": None,
            "id": req_id,
            "params": {},
            "error": {"code": -32600, "message": "Invalid Request: missing jsonrpc 2.0"},
        }

    method = body.get("method")
    if method is None or not isinstance(method, str):
        return {
            "method": None,
            "id": req_id,
            "params": {},
            "error": {"code": -32600, "message": "Invalid Request: missing method"},
        }

    params = body.get("params", {})
    if not isinstance(params, dict):
        params = {}

    return {
        "method": method,
        "id": req_id,
        "params": params,
        "error": None,
    }


# ---------------------------------------------------------------------------
# Method resolution
# ---------------------------------------------------------------------------

def resolve_method(method: str) -> Optional[str]:
    """Resolve a method name to a canonical KNOWN_METHODS entry.

    Returns the canonical name if found, or None if unknown.
    """
    if method in KNOWN_METHODS:
        return method
    if method in METHOD_ALIASES:
        return METHOD_ALIASES[method]
    return None


# ---------------------------------------------------------------------------
# Response builders
# ---------------------------------------------------------------------------

def make_error_response(req_id: Any, code: int, message: str) -> Dict:
    """Build a JSON-RPC 2.0 error response."""
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": code, "message": message},
    }


def make_result_response(req_id: Any, result: Any) -> Dict:
    """Build a JSON-RPC 2.0 success response."""
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "result": result,
    }


# ---------------------------------------------------------------------------
# Parameter extraction
# ---------------------------------------------------------------------------

def extract_user_text(params: Dict) -> Tuple[str, Optional[str], Optional[str]]:
    """Extract user text, context_id, and task_id from A2A message params.

    Returns:
        (user_text, context_id, task_id)
    """
    message: Dict = params.get("message", {})
    if not message:
        return ("", None, None)

    # Extract text from parts
    parts = message.get("parts")
    user_text = ""
    if parts is not None:
        text_pieces = [p["text"] for p in parts if isinstance(p, dict) and "text" in p]
        user_text = "".join(text_pieces)
        if not user_text:
            # Fall back to message-level text field if parts had no text
            user_text = message.get("text", "")
    else:
        # No parts key at all — fall back to text field
        user_text = message.get("text", "")

    # Extract context_id: message field takes priority over top-level params
    context_id: Optional[str] = message.get("context_id") or params.get("context_id")

    # Extract task_id: message field takes priority over top-level params
    task_id: Optional[str] = message.get("task_id") or params.get("task_id")

    return (user_text, context_id, task_id)
