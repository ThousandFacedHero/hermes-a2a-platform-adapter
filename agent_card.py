"""
A2A spec-compliant AgentCard builder for the Hermes plugin.

Provides utilities to convert Hermes (OpenAI-format) tool definitions into
A2A AgentSkill dicts and to assemble a complete A2A AgentCard.
"""

from typing import Dict, List

PLUGIN_VERSION = "1.0.0"


def build_agent_skills(tool_definitions: List[Dict]) -> List[Dict]:
    """
    Convert Hermes tool definitions to A2A AgentSkill dicts.

    Parameters
    ----------
    tool_definitions:
        List of OpenAI-format tool dicts:
        ``{"type": "function", "function": {"name": "...", "description": "..."}}``

    Returns
    -------
    List of AgentSkill dicts:
        ``{"id": name, "name": name, "description": description, "tags": []}``

    Tools with no function name are silently skipped.
    """
    skills = []
    for tool in tool_definitions:
        func = tool.get("function", {})
        name = func.get("name", "")
        if not name:
            continue
        description = func.get("description", "")
        skills.append({
            "id": name,
            "name": name,
            "description": description,
            "tags": [],
        })
    return skills


def build_agent_card(
    agent_name: str,
    agent_description: str,
    gateway_url: str,
    tool_definitions: List[Dict],
) -> Dict:
    """
    Build a spec-compliant A2A AgentCard dict.

    Parameters
    ----------
    agent_name:
        Human-readable name for the agent.
    agent_description:
        Short description of the agent's purpose.
    gateway_url:
        Base URL of the A2A gateway (trailing slashes are stripped).
    tool_definitions:
        List of OpenAI-format tool dicts used to populate ``skills``.

    Returns
    -------
    A2A AgentCard dict conforming to the A2A v1.0 specification.
    """
    base_url = gateway_url.rstrip("/")
    return {
        "name": agent_name,
        "description": agent_description,
        "version": PLUGIN_VERSION,
        "supported_interfaces": [
            {
                "url": f"{base_url}/a2a",
                "protocol_binding": "JSONRPC",
                "protocol_version": "1.0",
            }
        ],
        "capabilities": {"streaming": True},
        "default_input_modes": ["text/plain"],
        "default_output_modes": ["text/plain"],
        "skills": build_agent_skills(tool_definitions),
    }
