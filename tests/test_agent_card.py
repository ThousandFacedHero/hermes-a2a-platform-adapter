"""
Tests for agent_card.py — spec-compliant A2A AgentCard builder.
Run from plugins/hermes-a2a/ directory:
    python -m pytest tests/test_agent_card.py -v
"""

import pytest

from agent_card import (
    PLUGIN_VERSION,
    build_agent_skills,
    build_agent_card,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_tool(name: str, description: str = "A tool") -> dict:
    return {"type": "function", "function": {"name": name, "description": description}}


def make_tool_no_name() -> dict:
    return {"type": "function", "function": {}}


# ---------------------------------------------------------------------------
# TestBuildAgentSkills
# ---------------------------------------------------------------------------

class TestBuildAgentSkills:
    def test_maps_tool_definitions_to_skills(self):
        tools = [
            make_tool("search_web", "Searches the web"),
            make_tool("send_email", "Sends an email"),
        ]
        skills = build_agent_skills(tools)
        assert len(skills) == 2

    def test_skill_has_correct_id(self):
        tools = [make_tool("my_tool", "Does a thing")]
        skills = build_agent_skills(tools)
        assert skills[0]["id"] == "my_tool"

    def test_skill_has_correct_name(self):
        tools = [make_tool("my_tool", "Does a thing")]
        skills = build_agent_skills(tools)
        assert skills[0]["name"] == "my_tool"

    def test_skill_has_correct_description(self):
        tools = [make_tool("my_tool", "Does a thing")]
        skills = build_agent_skills(tools)
        assert skills[0]["description"] == "Does a thing"

    def test_skill_has_tags_as_list(self):
        tools = [make_tool("my_tool", "Does a thing")]
        skills = build_agent_skills(tools)
        assert isinstance(skills[0]["tags"], list)
        assert skills[0]["tags"] == []

    def test_skill_has_all_required_fields(self):
        tools = [make_tool("my_tool", "Does a thing")]
        skills = build_agent_skills(tools)
        skill = skills[0]
        assert "id" in skill
        assert "name" in skill
        assert "description" in skill
        assert "tags" in skill

    def test_empty_tool_list_returns_empty(self):
        skills = build_agent_skills([])
        assert skills == []

    def test_skips_tools_without_name(self):
        tools = [
            make_tool_no_name(),
            make_tool("valid_tool", "A valid tool"),
        ]
        skills = build_agent_skills(tools)
        assert len(skills) == 1
        assert skills[0]["id"] == "valid_tool"

    def test_all_skipped_returns_empty(self):
        tools = [make_tool_no_name(), make_tool_no_name()]
        skills = build_agent_skills(tools)
        assert skills == []


# ---------------------------------------------------------------------------
# TestBuildAgentCard
# ---------------------------------------------------------------------------

class TestBuildAgentCard:
    def _make_card(self, gateway_url="https://example.com", tools=None):
        if tools is None:
            tools = [make_tool("echo", "Echoes input")]
        return build_agent_card(
            agent_name="TestAgent",
            agent_description="A test agent",
            gateway_url=gateway_url,
            tool_definitions=tools,
        )

    def test_has_name(self):
        card = self._make_card()
        assert card["name"] == "TestAgent"

    def test_has_description(self):
        card = self._make_card()
        assert card["description"] == "A test agent"

    def test_has_version(self):
        card = self._make_card()
        assert card["version"] == PLUGIN_VERSION

    def test_version_is_plugin_version_constant(self):
        card = self._make_card()
        assert card["version"] == "1.0.0"

    def test_has_supported_interfaces(self):
        card = self._make_card()
        assert "supported_interfaces" in card

    def test_has_capabilities(self):
        card = self._make_card()
        assert "capabilities" in card

    def test_has_default_input_modes(self):
        card = self._make_card()
        assert "default_input_modes" in card

    def test_has_default_output_modes(self):
        card = self._make_card()
        assert "default_output_modes" in card

    def test_has_skills(self):
        card = self._make_card()
        assert "skills" in card

    def test_has_all_required_fields(self):
        card = self._make_card()
        required = [
            "name", "description", "version",
            "supported_interfaces", "capabilities",
            "default_input_modes", "default_output_modes",
            "skills",
        ]
        for field in required:
            assert field in card, f"Missing required field: {field}"

    def test_supported_interfaces_is_list(self):
        card = self._make_card()
        assert isinstance(card["supported_interfaces"], list)

    def test_supported_interfaces_has_one_entry(self):
        card = self._make_card()
        assert len(card["supported_interfaces"]) == 1

    def test_supported_interface_has_url(self):
        card = self._make_card()
        iface = card["supported_interfaces"][0]
        assert "url" in iface

    def test_supported_interface_url_appends_a2a(self):
        card = self._make_card(gateway_url="https://example.com")
        iface = card["supported_interfaces"][0]
        assert iface["url"] == "https://example.com/a2a"

    def test_supported_interface_has_protocol_binding(self):
        card = self._make_card()
        iface = card["supported_interfaces"][0]
        assert "protocol_binding" in iface
        assert iface["protocol_binding"] == "JSONRPC"

    def test_supported_interface_has_protocol_version(self):
        card = self._make_card()
        iface = card["supported_interfaces"][0]
        assert "protocol_version" in iface
        assert iface["protocol_version"] == "1.0"

    def test_strips_trailing_slash_from_gateway_url(self):
        card = self._make_card(gateway_url="https://example.com/")
        iface = card["supported_interfaces"][0]
        assert iface["url"] == "https://example.com/a2a"

    def test_strips_multiple_trailing_slashes(self):
        card = self._make_card(gateway_url="https://example.com///")
        iface = card["supported_interfaces"][0]
        assert iface["url"] == "https://example.com/a2a"

    def test_capabilities_has_streaming_true(self):
        card = self._make_card()
        assert card["capabilities"] == {"streaming": True}

    def test_default_input_modes_is_text_plain(self):
        card = self._make_card()
        assert card["default_input_modes"] == ["text/plain"]

    def test_default_output_modes_is_text_plain(self):
        card = self._make_card()
        assert card["default_output_modes"] == ["text/plain"]

    def test_includes_skills_from_tools(self):
        tools = [
            make_tool("tool_a", "Does A"),
            make_tool("tool_b", "Does B"),
        ]
        card = self._make_card(tools=tools)
        assert len(card["skills"]) == 2
        skill_ids = {s["id"] for s in card["skills"]}
        assert "tool_a" in skill_ids
        assert "tool_b" in skill_ids

    def test_no_top_level_url_field(self):
        card = self._make_card()
        assert "url" not in card

    def test_no_protocol_version_field(self):
        card = self._make_card()
        assert "protocolVersion" not in card

    def test_no_preferred_transport_field(self):
        card = self._make_card()
        assert "preferredTransport" not in card

    def test_no_non_spec_fields(self):
        card = self._make_card()
        non_spec = ["url", "protocolVersion", "preferredTransport"]
        for field in non_spec:
            assert field not in card, f"Non-spec field present: {field}"
