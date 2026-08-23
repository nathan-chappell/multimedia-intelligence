from httpx import ASGITransport, AsyncClient

from multimedia_intelligence.main import app


async def test_spa_fallback_and_not_found_boundaries() -> None:
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        root = await client.get("/")
        spa_route = await client.get("/some/future/client-route")
        missing_api = await client.get("/api/does-not-exist")
        missing_asset = await client.get("/assets/does-not-exist.js")
        devtools_probe = await client.get(
            "/.well-known/appspecific/com.chrome.devtools.json"
        )

    assert root.status_code == 200
    assert spa_route.status_code == 200
    assert "<div id=\"root\"></div>" in spa_route.text
    assert missing_api.status_code == 404
    assert missing_api.headers["content-type"].startswith("application/json")
    assert missing_asset.status_code == 404
    assert devtools_probe.status_code == 200
    assert devtools_probe.json() == {}
