from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from chatkit.server import StreamingResult
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from multimedia_intelligence.api.assets import build_asset_router
from multimedia_intelligence.api.health import router as health_router
from multimedia_intelligence.api.users import build_user_router
from multimedia_intelligence.auth import authenticate_request, ensure_builtin_admin
from multimedia_intelligence.chat.conversations import OpenAIConversationGateway
from multimedia_intelligence.chat.server import MultimediaChatServer
from multimedia_intelligence.chat.store import SqlAlchemyChatKitStore
from multimedia_intelligence.chat.transcription import OpenAITranscriptionGateway
from multimedia_intelligence.config import get_settings
from multimedia_intelligence.context import ClientInfo, RequestContext
from multimedia_intelligence.db import create_engine_and_session
from multimedia_intelligence.files.access import ScopedAgentDataAccess
from multimedia_intelligence.files.expiration import FileExpirationService
from multimedia_intelligence.files.s3_store import S3BlobStore
from multimedia_intelligence.observability import configure_logging

settings = get_settings()
configure_logging(settings)
engine, sessions = create_engine_and_session(settings.database_url)
conversation_gateway = OpenAIConversationGateway(settings.openai_api_key)
store = SqlAlchemyChatKitStore(
    engine,
    sessions,
    conversation_gateway,
    max_page_size=settings.chatkit_max_page_size,
)
transcription_gateway = OpenAITranscriptionGateway(
    settings.openai_api_key,
    settings.openai_dictation_model,
    max_audio_bytes=settings.max_dictation_bytes,
)
chatkit_server = MultimediaChatServer(
    store=store,
    transcription_gateway=transcription_gateway,
)
blob_store = S3BlobStore.from_settings(settings)
file_expiration = FileExpirationService(sessions, lambda: blob_store)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    await store.initialize()
    await ensure_builtin_admin(sessions, settings)
    expiration_stop = asyncio.Event()
    expiration_task = asyncio.create_task(
        file_expiration.run(expiration_stop, settings.expiration_sweep_seconds)
    )
    yield
    expiration_stop.set()
    await expiration_task
    await engine.dispose()


app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(health_router, prefix="/api")
app.include_router(build_asset_router(sessions, settings, blob_store), prefix="/api")
app.include_router(build_user_router(sessions, settings), prefix="/api")


@app.get("/.well-known/appspecific/com.chrome.devtools.json", include_in_schema=False)
async def chrome_devtools_discovery() -> JSONResponse:
    """Acknowledge Chrome's optional workspace discovery probe."""

    return JSONResponse({})


@app.post("/chatkit")
async def chatkit_endpoint(request: Request) -> Response:
    user = await authenticate_request(request, sessions, settings)
    context = RequestContext(
        client=ClientInfo(user_id=user.id, username=user.username, is_admin=user.is_admin),
        data_access=ScopedAgentDataAccess(sessions, user.id, blob_store),
        request=request,
    )
    result = await chatkit_server.process(await request.body(), context=context)
    if isinstance(result, StreamingResult):
        return StreamingResponse(result, media_type="text/event-stream")
    if hasattr(result, "json"):
        return Response(content=result.json, media_type="application/json")
    return JSONResponse(result)


frontend_dist = Path.cwd() / "frontend" / "dist"
frontend_assets = frontend_dist / "assets"
if frontend_assets.is_dir():
    app.mount("/assets", StaticFiles(directory=frontend_assets), name="frontend-assets")


@app.get("/{frontend_path:path}", include_in_schema=False)
async def serve_frontend(frontend_path: str) -> FileResponse:
    """Serve concrete build files and let the SPA handle extensionless routes."""

    if frontend_path == "chatkit" or frontend_path.startswith(("api/", "chatkit/")):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")
    if not frontend_dist.is_dir():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")
    requested_file = (frontend_dist / frontend_path).resolve()
    if requested_file.is_relative_to(frontend_dist.resolve()) and requested_file.is_file():
        return FileResponse(requested_file)
    if Path(frontend_path).suffix:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")
    return FileResponse(frontend_dist / "index.html")
