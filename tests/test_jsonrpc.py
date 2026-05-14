"""
Tests for jsonrpc.py — JSON-RPC 2.0 dispatcher with A2A method aliases.
Run from plugins/hermes-a2a/ directory:
    python -m pytest tests/test_jsonrpc.py -v
"""

import pytest

from jsonrpc import (
    parse_request,
    resolve_method,
    make_error_response,
    make_result_response,
    extract_user_text,
    METHOD_ALIASES,
    KNOWN_METHODS,
)


# ---------------------------------------------------------------------------
# TestParseRequest
# ---------------------------------------------------------------------------

class TestParseRequest:
    def test_valid_request(self):
        body = {"jsonrpc": "2.0", "method": "SendMessage", "id": 1, "params": {"foo": "bar"}}
        result = parse_request(body)
        assert result["method"] == "SendMessage"
        assert result["id"] == 1
        assert result["params"] == {"foo": "bar"}
        assert result["error"] is None

    def test_missing_jsonrpc_field(self):
        body = {"method": "SendMessage", "id": 1}
        result = parse_request(body)
        assert result["error"] is not None
        assert result["error"]["code"] == -32600
        assert "jsonrpc 2.0" in result["error"]["message"]

    def test_wrong_jsonrpc_version(self):
        body = {"jsonrpc": "1.0", "method": "SendMessage", "id": 1}
        result = parse_request(body)
        assert result["error"] is not None
        assert result["error"]["code"] == -32600

    def test_missing_method(self):
        body = {"jsonrpc": "2.0", "id": 1}
        result = parse_request(body)
        assert result["error"] is not None
        assert result["error"]["code"] == -32600
        assert "method" in result["error"]["message"]

    def test_method_not_string(self):
        body = {"jsonrpc": "2.0", "method": 42, "id": 1}
        result = parse_request(body)
        assert result["error"] is not None
        assert result["error"]["code"] == -32600

    def test_params_defaults_to_empty_dict(self):
        body = {"jsonrpc": "2.0", "method": "SendMessage", "id": 1}
        result = parse_request(body)
        assert result["error"] is None
        assert result["params"] == {}

    def test_id_preserved(self):
        body = {"jsonrpc": "2.0", "method": "SendMessage", "id": "abc-123"}
        result = parse_request(body)
        assert result["id"] == "abc-123"

    def test_null_id_allowed(self):
        body = {"jsonrpc": "2.0", "method": "SendMessage", "id": None}
        result = parse_request(body)
        assert result["id"] is None
        assert result["error"] is None


# ---------------------------------------------------------------------------
# TestResolveMethod
# ---------------------------------------------------------------------------

class TestResolveMethod:
    def test_primary_names_pass_through(self):
        for name in KNOWN_METHODS:
            assert resolve_method(name) == name

    def test_http_alias_message_send(self):
        assert resolve_method("message/send") == "SendMessage"

    def test_http_alias_message_stream(self):
        assert resolve_method("message/stream") == "SendStreamingMessage"

    def test_http_alias_tasks_get(self):
        assert resolve_method("tasks/get") == "GetTask"

    def test_http_alias_tasks_list(self):
        assert resolve_method("tasks/list") == "ListTasks"

    def test_http_alias_tasks_cancel(self):
        assert resolve_method("tasks/cancel") == "CancelTask"

    def test_http_alias_tasks_subscribe(self):
        assert resolve_method("tasks/subscribe") == "SubscribeToTask"

    def test_unknown_method_returns_none(self):
        assert resolve_method("DoSomethingElse") is None

    def test_empty_string_returns_none(self):
        assert resolve_method("") is None

    def test_all_aliases_map_to_known_methods(self):
        for alias, canonical in METHOD_ALIASES.items():
            assert canonical in KNOWN_METHODS, f"{alias} maps to {canonical!r} which is not in KNOWN_METHODS"


# ---------------------------------------------------------------------------
# TestMakeErrorResponse
# ---------------------------------------------------------------------------

class TestMakeErrorResponse:
    def test_correct_format(self):
        resp = make_error_response(1, -32600, "Invalid Request")
        assert resp == {
            "jsonrpc": "2.0",
            "id": 1,
            "error": {"code": -32600, "message": "Invalid Request"},
        }

    def test_null_id(self):
        resp = make_error_response(None, -32700, "Parse error")
        assert resp["id"] is None
        assert resp["error"]["code"] == -32700

    def test_string_id(self):
        resp = make_error_response("req-42", -32601, "Method not found")
        assert resp["id"] == "req-42"
        assert resp["jsonrpc"] == "2.0"

    def test_no_result_key(self):
        resp = make_error_response(1, -32600, "bad")
        assert "result" not in resp


# ---------------------------------------------------------------------------
# TestMakeResultResponse
# ---------------------------------------------------------------------------

class TestMakeResultResponse:
    def test_correct_format(self):
        resp = make_result_response(1, {"task": "data"})
        assert resp == {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"task": "data"},
        }

    def test_null_id(self):
        resp = make_result_response(None, {})
        assert resp["id"] is None

    def test_string_id(self):
        resp = make_result_response("abc", [1, 2, 3])
        assert resp["id"] == "abc"
        assert resp["result"] == [1, 2, 3]

    def test_no_error_key(self):
        resp = make_result_response(1, {})
        assert "error" not in resp


# ---------------------------------------------------------------------------
# TestExtractUserText
# ---------------------------------------------------------------------------

class TestExtractUserText:
    def test_text_from_parts(self):
        params = {
            "message": {
                "parts": [{"text": "hello world"}]
            }
        }
        text, context_id, task_id = extract_user_text(params)
        assert text == "hello world"
        assert context_id is None
        assert task_id is None

    def test_multiple_parts_joined(self):
        params = {
            "message": {
                "parts": [
                    {"text": "hello"},
                    {"text": " world"},
                ]
            }
        }
        text, _, _ = extract_user_text(params)
        assert text == "hello world"

    def test_parts_without_text_skipped(self):
        params = {
            "message": {
                "parts": [
                    {"type": "file", "data": "..."},
                    {"text": "actual text"},
                ]
            }
        }
        text, _, _ = extract_user_text(params)
        assert text == "actual text"

    def test_context_id_from_message(self):
        params = {
            "message": {
                "parts": [{"text": "hi"}],
                "context_id": "ctx-abc",
            }
        }
        _, context_id, _ = extract_user_text(params)
        assert context_id == "ctx-abc"

    def test_task_id_from_message(self):
        params = {
            "message": {
                "parts": [{"text": "hi"}],
                "task_id": "task-xyz",
            }
        }
        _, _, task_id = extract_user_text(params)
        assert task_id == "task-xyz"

    def test_context_id_from_params_fallback(self):
        params = {
            "context_id": "ctx-from-params",
            "message": {
                "parts": [{"text": "hi"}],
            }
        }
        _, context_id, _ = extract_user_text(params)
        assert context_id == "ctx-from-params"

    def test_task_id_from_params_fallback(self):
        params = {
            "task_id": "task-from-params",
            "message": {
                "parts": [{"text": "hi"}],
            }
        }
        _, _, task_id = extract_user_text(params)
        assert task_id == "task-from-params"

    def test_message_context_id_takes_priority_over_params(self):
        params = {
            "context_id": "ctx-params",
            "message": {
                "parts": [{"text": "hi"}],
                "context_id": "ctx-message",
            }
        }
        _, context_id, _ = extract_user_text(params)
        assert context_id == "ctx-message"

    def test_fallback_to_text_field(self):
        params = {
            "message": {
                "parts": [],
                "text": "fallback text",
            }
        }
        text, _, _ = extract_user_text(params)
        assert text == "fallback text"

    def test_empty_parts_returns_empty_string(self):
        params = {
            "message": {
                "parts": [],
            }
        }
        text, context_id, task_id = extract_user_text(params)
        assert text == ""
        assert context_id is None
        assert task_id is None

    def test_missing_message_returns_empty(self):
        params = {}
        text, context_id, task_id = extract_user_text(params)
        assert text == ""
        assert context_id is None
        assert task_id is None

    def test_missing_parts_key_falls_back_to_text(self):
        params = {
            "message": {
                "text": "only text field",
            }
        }
        text, _, _ = extract_user_text(params)
        assert text == "only text field"
