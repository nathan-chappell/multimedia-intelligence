from __future__ import annotations

import os

import jmespath
import pytest
from openai import OpenAI

from multimedia_intelligence.files.tools.jmespath_commands import (
    JmesPathValidator,
    jmespath_custom_tool,
)

LIVE_MODEL = "gpt-5.6"

GENERATION_CASES = (
    (
        "CSV rows are a JSON array with timestamp, region, units, and revenue fields. "
        "Generate a JMESPath expression that returns region and revenue for rows where revenue "
        "is at least 100, sorted by revenue."
    ),
    (
        "CSV rows are a JSON array with product and units fields. Generate a JMESPath expression "
        "that computes the average units value."
    ),
    (
        "The JSON document has an events array. Generate a JMESPath expression that selects "
        "events whose type is opened and returns their timestamp and actor fields."
    ),
)


@pytest.mark.live
@pytest.mark.skipif(
    os.getenv("RUN_OPENAI_BEHAVIORAL") != "1",
    reason="Set RUN_OPENAI_BEHAVIORAL=1 to allow OpenAI API calls",
)
@pytest.mark.parametrize("prompt", GENERATION_CASES)
def test_model_generates_valid_jmespath(prompt: str) -> None:
    response = OpenAI().responses.create(
        model=LIVE_MODEL,
        input=prompt,
        tools=[jmespath_custom_tool()],  # type: ignore[list-item]
        tool_choice={"type": "custom", "name": "query_structured_data"},
    )
    calls = [item for item in response.output if item.type == "custom_tool_call"]
    assert len(calls) == 1

    expression = calls[0].input
    JmesPathValidator().validate(expression)
    jmespath.compile(expression)
