from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import logging
import json
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .session_manager import SessionManager

app = FastAPI(title="Realtime Ditto Backend")
sessions = SessionManager()
logger = logging.getLogger(__name__)
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

if FRONTEND_DIR.exists():
    app.mount("/ui", StaticFiles(directory=FRONTEND_DIR, html=True), name="ui")


class SessionCreateResponse(BaseModel):
    session_id: str


class AudioChunkRequest(BaseModel):
    audio_base64: str


class GenericResponse(BaseModel):
    status: str


@app.get("/", response_class=HTMLResponse)
async def root() -> HTMLResponse:
    if FRONTEND_DIR.exists():
        return RedirectResponse(url="/ui/", status_code=307)
    return HTMLResponse("<h1>Realtime Ditto Backend</h1>")


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


@app.websocket("/ws/realtime")
async def realtime_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        session_id = await sessions.create_session()
    except RuntimeError as exc:
        logger.error("Failed to create realtime session", exc_info=True)
        await websocket.send_json(
            {
                "type": "error",
                "error": {"message": str(exc)},
            }
        )
        await websocket.close(code=1013)
        return
    await websocket.send_json({"type": "session.created", "session_id": session_id})

    forward_task = asyncio.create_task(_forward_events_to_websocket(websocket, session_id))

    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break

            if "bytes" in message and message["bytes"] is not None:
                await sessions.append_audio(session_id, message["bytes"])
                continue

            if "text" in message and message["text"] is not None:
                should_close = await _handle_text_frame(session_id, message["text"], websocket)
                if should_close:
                    break
                continue

    except WebSocketDisconnect:
        pass
    finally:
        forward_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await forward_task
        await sessions.close_session(session_id)


async def _handle_text_frame(session_id: str, payload: str, websocket: WebSocket) -> bool:
    try:
        message = json.loads(payload)
    except json.JSONDecodeError:
        logger.warning("Invalid JSON payload received on realtime websocket", exc_info=True)
        await websocket.send_json(
            {
                "type": "error",
                "error": {"message": "Invalid JSON payload"},
            }
        )
        return False

    msg_type = message.get("type")
    if msg_type == "audio":
        audio_b64 = message.get("audio")
        if not audio_b64:
            logger.warning("Realtime websocket audio message missing 'audio' field")
            await websocket.send_json(
                {
                    "type": "error",
                    "error": {"message": "Missing 'audio' field"},
                }
            )
            return False
        try:
            pcm = base64.b64decode(audio_b64)
        except (binascii.Error, ValueError):
            logger.warning("Realtime websocket audio payload failed to decode", exc_info=True)
            await websocket.send_json(
                {
                    "type": "error",
                    "error": {"message": "Invalid base64 audio payload"},
                }
            )
            return False
        await sessions.append_audio(session_id, pcm)
        return False

    if msg_type == "commit":
        instructions = message.get("instructions")
        await sessions.commit(session_id, instructions=instructions)
        return False

    if msg_type in {"response.create", "request_response"}:
        instructions = message.get("instructions")
        await sessions.request_response(session_id, instructions=instructions)
        return False

    if msg_type == "close":
        return True

    if msg_type == "ping":
        await websocket.send_json({"type": "pong"})
        return False

    logger.warning("Unsupported realtime websocket message type: %s", msg_type)
    await websocket.send_json(
        {
            "type": "error",
            "error": {"message": f"Unsupported message type: {msg_type}"},
        }
    )
    return False


async def _forward_events_to_websocket(websocket: WebSocket, session_id: str) -> None:
    try:
        async for event in sessions.stream_events(session_id):
            try:
                await websocket.send_text(json.dumps(event))
            except WebSocketDisconnect:
                break
    except asyncio.CancelledError:
        raise
