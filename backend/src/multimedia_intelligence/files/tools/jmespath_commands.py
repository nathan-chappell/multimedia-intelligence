from __future__ import annotations

from importlib.resources import files

from lark import Lark

GRAMMAR_RESOURCE = "files/grammars/jmespath.lark"


def load_jmespath_grammar() -> str:
    package_root = files("multimedia_intelligence")
    return package_root.joinpath(GRAMMAR_RESOURCE).read_text(encoding="utf-8")


class JmesPathValidator:
    """Validate JMESPath syntax with the same grammar supplied to OpenAI."""

    def __init__(self, grammar: str | None = None) -> None:
        self._parser = Lark(grammar or load_jmespath_grammar(), parser="lalr")

    def validate(self, expression: str) -> None:
        self._parser.parse(expression)


def jmespath_custom_tool() -> dict[str, object]:
    """Return a grammar-constrained custom tool that generates one JMESPath expression."""

    return {
        "type": "custom",
        "name": "query_structured_data",
        "description": (
            "Write one JMESPath expression to query JSON data. CSV inputs are represented as "
            "an array of JSON objects, with inferred numbers, booleans, nulls, and strings."
        ),
        "format": {
            "type": "grammar",
            "syntax": "lark",
            "definition": load_jmespath_grammar(),
        },
    }
