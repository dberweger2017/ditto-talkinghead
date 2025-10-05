from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any, Dict

from .config import get_settings
from .gpt_client import GPTRealtimeClient


@dataclass
class Session:
    session_id: str
    client: GPTRealtimeClient
    event_task: asyncio.Task[None]
    event_queue: asyncio.Queue[dict[str, Any]] = field(default_factory=asyncio.Queue)
    created_at: float = field(default_factory=lambda: asyncio.get_event_loop().time())


class SessionManager:
    def __init__(self) -> None:
        self._sessions: Dict[str, Session] = {}
        self._settings = get_settings()
        self._lock = asyncio.Lock()

    async def create_session(self) -> str:
        async with self._lock:
            if len(self._sessions) >= self._settings.max_sessions:
                raise RuntimeError("Session limit exceeded")

            session_id = uuid.uuid4().hex
            client = GPTRealtimeClient(
                api_key=self._settings.openai_api_key or "",
                model=self._settings.openai_realtime_model,
            )
            await client.connect()

            event_task = asyncio.create_task(self._pipe_events(session_id, client))
            self._sessions[session_id] = Session(
                session_id=session_id,
                client=client,
                event_task=event_task,
            )
            return session_id

    async def append_audio(self, session_id: str, pcm16: bytes) -> None:
        session = await self._get_session(session_id)
        await session.client.send_audio_chunk(pcm16)

    async def commit(self, session_id: str) -> None:
        session = await self._get_session(session_id)
        await session.client.commit_input()
        await session.client.request_response()

    async def close_session(self, session_id: str) -> None:
        async with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is None:
            return
        session.event_task.cancel()
        await session.client.close()
        try:
            await session.event_task
        except asyncio.CancelledError:
            pass

    async def stream_events(self, session_id: str) -> AsyncGenerator[dict[str, Any], None]:
        session = await self._get_session(session_id)
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
                await session.event_queue.put(event)
        except asyncio.CancelledError:
            pass
        except KeyError:
            pass
