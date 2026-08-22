from __future__ import annotations

from collections.abc import Sequence

from chatkit.types import UserMessageItem

ALLOWED_CHAT_MODELS = frozenset(
    {
        "gpt-5.6",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
    }
)


def resolve_chat_model(
    current_item: UserMessageItem | None,
    thread_items: Sequence[object],
) -> str:
    """Resolve a ChatKit-selected model without accepting arbitrary model IDs."""

    candidates = [current_item] if current_item is not None else []
    candidates.extend(item for item in reversed(thread_items) if isinstance(item, UserMessageItem))
    for item in candidates:
        model = item.inference_options.model
        if model is None:
            continue
        if model not in ALLOWED_CHAT_MODELS:
            raise ValueError(f"Unsupported chat model: {model}")
        return model
    raise ValueError("ChatKit must provide a selected model for every conversation turn")
