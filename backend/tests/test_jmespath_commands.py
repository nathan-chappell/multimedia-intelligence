import pytest
from lark import Lark, Tree
from lark.exceptions import UnexpectedInput

from multimedia_intelligence.files.tools.jmespath_commands import (
    JmesPathValidator,
    jmespath_custom_tool,
    load_jmespath_grammar,
)


@pytest.mark.parametrize(
    "expression",
    [
        "foo.bar",
        '"key with spaces"',
        "rows[0].name",
        "rows[-1]",
        "rows[1:5:2]",
        "rows[]",
        "rows[*].name",
        "rows[?revenue >= `100`].{region: region, revenue: revenue}",
        "sort_by(rows, &revenue)[-1]",
        "length(rows) > `0` && rows[0].active == `true`",
        "primary || fallback | [0]",
        "[name, totals.revenue]",
        '{"display name": name, total: sum(values)}',
        "contains(name, 'O\\'Brien')",
        "'newline\n'",
        r'metadata."unicode\u2713"',
        r'`{"enabled": true, "thresholds": [1, 2.5, null]}`.enabled',
        "@",
        "!disabled",
        "*",
        "[]",
    ],
)
def test_jmespath_grammar_accepts_spec_expressions(expression: str) -> None:
    JmesPathValidator().validate(expression)


@pytest.mark.parametrize(
    "expression",
    [
        "",
        ".foo",
        "foo.",
        "foo..bar",
        "1",
        "$",
        "foo[01]",
        "foo[1,2]",
        "foo[?]",
        'foo[ ?bar == `"baz"`]',
        "foo[bar]",
        "foo === bar",
        "foo &&",
        "(foo",
        "{}",
        "{foo}",
        "foo('unterminated)",
        '"bad\\qescape"',
        "`{bad json}`",
    ],
)
def test_jmespath_grammar_rejects_invalid_expressions(expression: str) -> None:
    with pytest.raises(UnexpectedInput):
        JmesPathValidator().validate(expression)


def test_jmespath_binding_order_is_pipe_then_or_then_and() -> None:
    tree = Lark(load_jmespath_grammar(), parser="lalr").parse("a | b || c && d")
    assert tree.data == "pipe_expression"
    or_expression = tree.children[1]
    assert isinstance(or_expression, Tree) and or_expression.data == "or_expression"
    and_expression = or_expression.children[1]
    assert isinstance(and_expression, Tree) and and_expression.data == "and_expression"


def test_custom_tool_embeds_lark_grammar() -> None:
    tool = jmespath_custom_tool()
    assert tool["type"] == "custom"
    assert tool["name"] == "query_structured_data"
    assert tool["format"] == {
        "type": "grammar",
        "syntax": "lark",
        "definition": tool["format"]["definition"],  # type: ignore[index]
    }
