import pytest
from lark.exceptions import UnexpectedInput

from multimedia_intelligence.files.tools.json_commands import (
    JsonCommandValidator,
    json_inspection_custom_tool,
)


@pytest.mark.parametrize(
    "command",
    [
        "Chars(0, 4096)",
        "JsonPath($.items[*].name)",
        'JsonPath($["odd-key"]|$.metadata.version)',
    ],
)
def test_json_grammar_accepts_bounded_commands(command: str) -> None:
    JsonCommandValidator().validate(command)


def test_json_grammar_rejects_script_filters() -> None:
    with pytest.raises(UnexpectedInput):
        JsonCommandValidator().validate("JsonPath($.items[?(@.secret)])")


def test_custom_tool_embeds_lark_grammar() -> None:
    tool = json_inspection_custom_tool()
    assert tool["type"] == "custom"
    assert tool["format"] == {
        "type": "grammar",
        "syntax": "lark",
        "definition": tool["format"]["definition"],  # type: ignore[index]
    }
