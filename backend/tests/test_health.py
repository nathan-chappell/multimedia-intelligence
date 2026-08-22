from fastapi.testclient import TestClient

from multimedia_intelligence.main import app


def test_health() -> None:
    with TestClient(app) as client:
        response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chatkit_requires_bearer_authentication() -> None:
    with TestClient(app) as client:
        response = client.post("/chatkit", content=b"{}")
    assert response.status_code == 401
