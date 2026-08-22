from fastapi.testclient import TestClient

from multimedia_intelligence.main import app, settings


def test_health() -> None:
    with TestClient(app) as client:
        response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chatkit_requires_bearer_authentication() -> None:
    with TestClient(app) as client:
        response = client.post("/chatkit", content=b"{}")
    assert response.status_code == 401


def test_regular_api_routes_accept_the_same_bearer_token() -> None:
    token = settings.admin_bearer_token.get_secret_value()
    with TestClient(app) as client:
        response = client.post(
            "/api/ingestion/plans/missing/approve",
            headers={"Authorization": f"Bearer {token}"},
        )
    assert response.status_code == 404
