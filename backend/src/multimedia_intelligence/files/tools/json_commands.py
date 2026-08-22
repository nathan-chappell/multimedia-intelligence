from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files

from lark import Lark

GRAMMAR_RESOURCE = "files/grammars/json_inspection.lark"


@dataclass(frozen=True, slots=True)
class JsonInspectionLimits:
    max_chars_per_call: int = 64 * 1024
    max_queries_per_call: int = 8
    max_results_per_query: int = 100
    max_result_bytes: int = 256 * 1024


def load_json_inspection_grammar() -> str:
    package_root = files("multimedia_intelligence")
    return package_root.joinpath(GRAMMAR_RESOURCE).read_text(encoding="utf-8")


class JsonCommandValidator:
    """Server-side validation for the same Lark grammar sent to OpenAI.

    The grammar intentionally exposes a safe JSONPath subset: property, index,
    wildcard, and quoted-property selectors. Script expressions and recursive
    filters are excluded so a model-generated query cannot become executable JS.
    """

    def __init__(self, grammar: str | None = None) -> None:
        self._parser = Lark(grammar or load_json_inspection_grammar(), parser="lalr")

    def validate(self, command: str) -> None:
        self._parser.parse(command)


def json_inspection_custom_tool() -> dict[str, object]:
    """Return the Responses custom-tool declaration used by the gateway later."""

    return {
        "type": "custom",
        "name": "inspect_json",
        "description": (
            "Read a bounded character range with Chars(start,count), or evaluate one "
            "or more safe JSONPath expressions with JsonPath(query|query)."
        ),
        "format": {
            "type": "grammar",
            "syntax": "lark",
            "definition": load_json_inspection_grammar(),
        },
    }
