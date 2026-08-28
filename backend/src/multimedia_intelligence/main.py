from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from agents import set_default_openai_key
from chatkit.server import StreamingResult
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from multimedia_intelligence.api.assets import build_asset_router
from multimedia_intelligence.api.billing import build_billing_router
from multimedia_intelligence.api.collections import build_collection_router
from multimedia_intelligence.api.health import router as health_router
from multimedia_intelligence.api.threads import build_thread_router
from multimedia_intelligence.api.users import build_user_router
from multimedia_intelligence.auth import authenticate_request
from multimedia_intelligence.billing import BillingService
from multimedia_intelligence.billing.pricing import validate_configured_pricing
from multimedia_intelligence.chat.conversations import OpenAIConversationGateway
from multimedia_intelligence.chat.server import MultimediaChatServer
from multimedia_intelligence.chat.store import SqlAlchemyChatKitStore
from multimedia_intelligence.chat.titles import OpenAITitleSuggestionGateway
from multimedia_intelligence.chat.transcription import OpenAITranscriptionGateway
from multimedia_intelligence.config import get_settings
from multimedia_intelligence.context import ClientInfo, RequestContext
from multimedia_intelligence.db import create_engine_and_session
from multimedia_intelligence.files.access import ScopedAgentDataAccess
from multimedia_intelligence.files.indexing import (
    FileIndexReader,
    FileIndexWriter,
    OpenAIMediaTranscriptionGateway,
    OpenAIVectorStoreGateway,
)
from multimedia_intelligence.files.s3_store import S3BlobStore
from multimedia_intelligence.files.transcripts import AssetTranscriptCache
from multimedia_intelligence.observability import configure_logging

settings = get_settings()
configure_logging(settings)
if settings.openai_api_key:
    # Pydantic reads the project .env file, but the Agents SDK does not. Configure
    # its default provider explicitly so agent turns use the same key as ingestion.
    set_default_openai_key(
        settings.openai_api_key,
        use_for_tracing=settings.openai_tracing_enabled,
    )
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
billing = BillingService(sessions, settings)
title_suggestions = OpenAITitleSuggestionGateway(
    settings.openai_api_key,
    settings.openai_title_model,
    settings=settings,
)
chatkit_server = MultimediaChatServer(
    store=store,
    transcription_gateway=transcription_gateway,
    billing=billing,
)
blob_store = S3BlobStore.from_settings(settings)
vector_store_gateway = (
    OpenAIVectorStoreGateway(settings.openai_api_key, settings) if settings.openai_api_key else None
)
media_transcription_gateway = (
    OpenAIMediaTranscriptionGateway(settings.openai_api_key, settings.openai_diarization_model)
    if settings.openai_api_key
    else None
)
transcript_cache = (
    AssetTranscriptCache(
        sessions,
        blob_store,
        media_transcription_gateway,
        settings,
        billing,
    )
    if media_transcription_gateway is not None
    else None
)
file_index = (
    FileIndexReader(
        sessions,
        blob_store,
        vector_store_gateway,
        max_vision_pdf_bytes=settings.max_vision_pdf_bytes,
    )
    if vector_store_gateway is not None
    else None
)
file_index_writer = (
    FileIndexWriter(
        sessions,
        blob_store,
        vector_store_gateway,
        settings=settings,
        transcription=media_transcription_gateway,
        transcript_cache=transcript_cache,
        billing=billing,
    )
    if vector_store_gateway is not None
    else None
)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    validate_configured_pricing(
        token_models=(
            "gpt-5.6-luna",
            "gpt-5.6-terra",
            "gpt-5.6",
            settings.openai_title_model,
        ),
        transcription_models=(
            settings.openai_dictation_model,
            settings.openai_diarization_model,
        ),
    )
    await store.initialize()
    yield
    await engine.dispose()


app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_origin_regex=settings.effective_cors_origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(health_router, prefix="/api")
app.include_router(build_asset_router(sessions, settings, blob_store), prefix="/api")
app.include_router(build_collection_router(sessions, settings, file_index), prefix="/api")
app.include_router(build_thread_router(store, settings, title_suggestions), prefix="/api")
app.include_router(build_user_router(sessions, settings), prefix="/api")
app.include_router(build_billing_router(sessions, settings), prefix="/api")


@app.get("/.well-known/appspecific/com.chrome.devtools.json", include_in_schema=False)
async def chrome_devtools_discovery() -> JSONResponse:
    """Acknowledge Chrome's optional workspace discovery probe."""

    return JSONResponse({})


@app.post("/chatkit")
async def chatkit_endpoint(request: Request) -> Response:
    user = await authenticate_request(request, sessions, settings)
    context = RequestContext(
        client=ClientInfo(user_id=user.id, username=user.username, is_admin=user.is_admin),
        data_access=ScopedAgentDataAccess(
            sessions,
            user.id,
            blob_store,
            file_index,
            file_index_writer,
            transcript_cache=transcript_cache,
        ),
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
