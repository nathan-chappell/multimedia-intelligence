from types import SimpleNamespace
from typing import Any, cast

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from openai import AsyncOpenAI

from multimedia_intelligence.api.billing import build_billing_router
from multimedia_intelligence.auth import (
    AuthenticatedUser,
    ensure_identity_row,
    mint_access_token,
)
from multimedia_intelligence.billing.attribution import OpenAIResponseAttributionGateway
from multimedia_intelligence.billing.service import BillingService
from multimedia_intelligence.db import create_engine_and_session, initialize_schema

from .settings import TEST_SETTINGS


class FakeAttributionGateway:
    def __init__(self) -> None:
        self.response_ids: list[str] = []

    async def retrieve(self, response_id: str) -> dict[str, object]:
        self.response_ids.append(response_id)
        return {
            "id": response_id,
            "status": "completed",
            "model": "gpt-5.6-luna",
            "usage": {"input_tokens": 42, "output_tokens": 11, "total_tokens": 53},
            "output": [{"type": "message", "role": "assistant"}],
        }


async def test_ledger_pages_and_response_attribution_are_user_scoped() -> None:
    engine, sessions = create_engine_and_session("sqlite+aiosqlite:///:memory:")
    await initialize_schema(engine)
    user = AuthenticatedUser(id="user_one", username="one", is_admin=False)
    other = AuthenticatedUser(id="user_two", username="two", is_admin=False)
    admin = AuthenticatedUser(id="admin", username="admin", is_admin=True)
    for identity in (user, other, admin):
        await ensure_identity_row(sessions, identity)

    billing = BillingService(sessions, TEST_SETTINGS)
    for index in range(12):
        await billing.adjust(
            user_id=user.id,
            amount_microusd=index + 1,
            actor_user_id=admin.id,
            description=f"Adjustment {index}",
        )
    attributed = await billing.append_event(
        user_id=user.id,
        amount_microusd=-100,
        event_type="agent_model_usage",
        description="Agent model usage",
        provider_request_id="req_attributed",
        provider_response_id="resp_attributed",
        idempotency_key="openai:req_attributed",
        event_metadata={"model": "gpt-5.6-luna"},
    )

    gateway = FakeAttributionGateway()
    app = FastAPI()
    app.include_router(
        build_billing_router(sessions, TEST_SETTINGS, gateway),
        prefix="/api",
    )
    user_token, _ = mint_access_token(user, TEST_SETTINGS)
    other_token, _ = mint_access_token(other, TEST_SETTINGS)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        page = await client.get(
            "/api/billing/ledger",
            params={"limit": 5, "offset": 5},
            headers={"Authorization": f"Bearer {user_token}"},
        )
        attribution = await client.get(
            f"/api/billing/ledger/{attributed.id}/attribution",
            headers={"Authorization": f"Bearer {user_token}"},
        )
        hidden = await client.get(
            f"/api/billing/ledger/{attributed.id}/attribution",
            headers={"Authorization": f"Bearer {other_token}"},
        )

    assert page.status_code == 200
    assert page.json()["total"] == 13
    assert page.json()["limit"] == 5
    assert page.json()["offset"] == 5
    assert len(page.json()["items"]) == 5
    assert attribution.status_code == 200
    assert attribution.json()["event"]["provider_response_id"] == "resp_attributed"
    assert attribution.json()["provider_response"] == {
        "id": "resp_attributed",
        "status": "completed",
        "model": "gpt-5.6-luna",
        "usage": {"input_tokens": 42, "output_tokens": 11, "total_tokens": 53},
        "output": [{"type": "message", "role": "assistant"}],
    }
    assert gateway.response_ids == ["resp_attributed"]
    assert hidden.status_code == 404
    await engine.dispose()


async def test_openai_attribution_excludes_request_secrets() -> None:
    class Responses:
        async def retrieve(self, response_id: str) -> Any:
            assert response_id == "resp_safe"
            return SimpleNamespace(
                model_dump=lambda **_kwargs: {
                    "id": response_id,
                    "status": "completed",
                    "model": "gpt-5.6-luna",
                    "output": [{"type": "message", "content": []}],
                    "instructions": "private application instructions",
                    "safety_identifier": "hashed-user-identifier",
                    "prompt_cache_key": "private-cache-key",
                }
            )

    client = cast(AsyncOpenAI, SimpleNamespace(responses=Responses()))
    gateway = OpenAIResponseAttributionGateway("test-key", client)

    result = await gateway.retrieve("resp_safe")

    assert result == {
        "id": "resp_safe",
        "status": "completed",
        "model": "gpt-5.6-luna",
        "output": [{"type": "message", "content": []}],
    }
