from __future__ import annotations

import asyncio
import uuid
import logging
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any, Dict

from .config import get_settings
from .gpt_client import GPTRealtimeClient
from .avatar_engine import AvatarEngine, BaseAvatarSession


@dataclass
class Session:
    session_id: str
    client: GPTRealtimeClient
    event_task: asyncio.Task[None]
    event_queue: asyncio.Queue[dict[str, Any]] = field(
        default_factory=lambda: asyncio.Queue(maxsize=2000)
    )
    avatar_session: BaseAvatarSession | None = None
    created_at: float = field(default_factory=lambda: asyncio.get_event_loop().time())


class SessionManager:
    def __init__(self) -> None:
        self._sessions: Dict[str, Session] = {}
        self._settings = get_settings()
        self._lock = asyncio.Lock()
        self._avatar_engine = AvatarEngine()
        self._logger = logging.getLogger("ditto.session")

    async def create_session(self) -> str:
        async with self._lock:
            if len(self._sessions) >= self._settings.max_sessions:
                raise RuntimeError("Session limit exceeded")

            if not self._settings.openai_api_key:
                raise RuntimeError("OPENAI_API_KEY is not configured")

            session_id = uuid.uuid4().hex
            self._logger.info("Creating session %s", session_id)
            client = GPTRealtimeClient(
                api_key=self._settings.openai_api_key,
                model=self._settings.openai_realtime_model,
            )
            await client.connect()
            avatar_session = await self._avatar_engine.create_session(session_id)

            event_task = asyncio.create_task(self._pipe_events(session_id, client))
            self._sessions[session_id] = Session(
                session_id=session_id,
                client=client,
                event_task=event_task,
                avatar_session=avatar_session,
            )
            self._logger.info("Session %s created", session_id)
            return session_id

    async def append_audio(self, session_id: str, pcm16: bytes) -> None:
        session = await self._get_session(session_id)
        self._logger.debug("Session %s append_audio len=%d", session_id, len(pcm16))
        await session.client.send_audio_chunk(pcm16)

    async def commit(self, session_id: str, *, instructions: str | None = None) -> None:
        session = await self._get_session(session_id)
        self._logger.info("Session %s commit (instructions=%s)", session_id, instructions)
        committed = await session.client.commit_input()
        if not committed:
            self._logger.info("Session %s commit skipped (no pending audio)", session_id)
            return
        await session.client.request_response(instructions=instructions)

    async def request_response(self, session_id: str, *, instructions: str | None = None) -> None:
        session = await self._get_session(session_id)
        self._logger.info("Session %s response requested (instructions=%s)", session_id, instructions)
        await session.client.request_response(instructions=instructions)

    async def close_session(self, session_id: str) -> None:
        async with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is None:
            return
        self._logger.info("Closing session %s", session_id)
        session.event_task.cancel()
        await session.client.close()
        avatar_session = session.avatar_session
        if avatar_session is not None:
            try:
                await avatar_session.finalize()
            except Exception:
                pass
        try:
            await session.event_task
        except asyncio.CancelledError:
            pass

    async def stream_events(self, session_id: str) -> AsyncGenerator[dict[str, Any], None]:
        session = await self._get_session(session_id)
        self._logger.debug("Streaming events for session %s", session_id)
        while True:
            event = await session.event_queue.get()
            yield event
            session.event_queue.task_done()

    async def _get_session(self, session_id: str) -> Session:
        async with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"Unknown session id: {session_id}")
        return session

    async def _pipe_events(self, session_id: str, client: GPTRealtimeClient) -> None:
        try:
            async for event in client.stream_events():
                session = await self._get_session(session_id)
                self._logger.debug("Session %s upstream event %s", session_id, event.get("type"))
                await session.event_queue.put(event)
        except asyncio.CancelledError:
            self._logger.debug("Event pipe cancelled for session %s", session_id)
            pass
        except KeyError:
            self._logger.debug("Event pipe terminating, session %s missing", session_id)
            pass
