from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal


@dataclass(frozen=True, slots=True)
class TokenRate:
    input_per_million: Decimal
    cached_input_per_million: Decimal
    output_per_million: Decimal


# Demo estimates are centralized and stamped onto every event. Production
# deployments should update them together with billing_pricing_version.
TOKEN_RATES: dict[str, TokenRate] = {
    "gpt-5.6-luna": TokenRate(Decimal("0.25"), Decimal("0.025"), Decimal("2.00")),
    "gpt-5.6-terra": TokenRate(Decimal("1.00"), Decimal("0.10"), Decimal("8.00")),
    "gpt-5.6": TokenRate(Decimal("2.50"), Decimal("0.25"), Decimal("15.00")),
}
TRANSCRIPTION_RATES_PER_MINUTE: dict[str, Decimal] = {
    "gpt-4o-mini-transcribe": Decimal("0.003"),
    "gpt-4o-transcribe-diarize": Decimal("0.006"),
}


def validate_configured_pricing(
    *, token_models: tuple[str, ...], transcription_models: tuple[str, ...]
) -> None:
    missing = [model for model in token_models if model not in TOKEN_RATES]
    missing.extend(
        model for model in transcription_models if model not in TRANSCRIPTION_RATES_PER_MINUTE
    )
    if missing:
        raise RuntimeError(
            f"Missing billing prices for configured models: {', '.join(sorted(set(missing)))}"
        )


def token_cost_microusd(
    model: str,
    *,
    input_tokens: int,
    cached_input_tokens: int,
    output_tokens: int,
    markup: float,
) -> int:
    rate = TOKEN_RATES.get(model)
    if rate is None:
        raise ValueError(f"No billing price is configured for model {model!r}")
    cached = max(min(cached_input_tokens, input_tokens), 0)
    uncached = max(input_tokens - cached, 0)
    provider_usd = (
        Decimal(uncached) * rate.input_per_million
        + Decimal(cached) * rate.cached_input_per_million
        + Decimal(max(output_tokens, 0)) * rate.output_per_million
    ) / Decimal(1_000_000)
    return _microusd(provider_usd * Decimal(str(markup)))


def transcription_cost_microusd(model: str, *, seconds: float, markup: float) -> int:
    rate = TRANSCRIPTION_RATES_PER_MINUTE.get(model)
    if rate is None:
        raise ValueError(f"No billing price is configured for transcription model {model!r}")
    return _microusd(Decimal(str(max(seconds, 0))) / Decimal(60) * rate * Decimal(str(markup)))


def _microusd(value: Decimal) -> int:
    return max(
        1,
        int((value * Decimal(1_000_000)).quantize(Decimal("1"), rounding=ROUND_HALF_UP)),
    )
