from __future__ import annotations

import base64
import json
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .session_manager import SessionManager

app = FastAPI(title="Realtime Ditto Backend")
sessions = SessionManager()


class SessionCreateResponse(BaseModel):
    session_id: str


class AudioChunkRequest(BaseModel):
    audio_base64: str


class GenericResponse(BaseModel):
    status: str


@app.post("/sessions", response_model=SessionCreateResponse)
async def create_session() -> SessionCreateResponse:
    try:
        session_id = await sessions.create_session()
    except RuntimeError as exc:
        raise HTTPException(status_code=429, detail=str(exc))
    return SessionCreateResponse(session_id=session_id)


@app.post("/sessions/{session_id}/audio", response_model=GenericResponse)
async def upload_audio_chunk(session_id: str, request: AudioChunkRequest) -> GenericResponse:
    try:
        pcm16 = base64.b64decode(request.audio_base64)
        await sessions.append_audio(session_id, pcm16)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")
    return GenericResponse(status="queued")


@app.post("/sessions/{session_id}/commit", response_model=GenericResponse)
async def commit_session(session_id: str) -> GenericResponse:
    try:
        await sessions.commit(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")
    return GenericResponse(status="committed")


async def _event_stream(session_id: str) -> AsyncGenerator[bytes, None]:
    try:
        async for event in sessions.stream_events(session_id):
            payload = json.dumps(event).encode("utf-8")
            yield b"data: " + payload + b"\n\n"
    except KeyError:
        yield b"event: error\ndata: {\"detail\": \"Session not found\"}\n\n"


@app.get("/sessions/{session_id}/events")
async def session_events(session_id: str) -> StreamingResponse:
    generator = _event_stream(session_id)
    headers = {"Cache-Control": "no-cache"}
    return StreamingResponse(generator, media_type="text/event-stream", headers=headers)


@app.delete("/sessions/{session_id}", response_model=GenericResponse)
async def delete_session(session_id: str) -> GenericResponse:
    await sessions.close_session(session_id)
    return GenericResponse(status="closed")


@app.get("/healthz", response_model=GenericResponse)
async def healthcheck() -> GenericResponse:
    return GenericResponse(status="ok")
