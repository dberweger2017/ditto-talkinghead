from __future__ import annotations

import asyncio
import base64
import json
from typing import Any, AsyncGenerator


class GPTRealtimeClient:
    """Async client skeleton for OpenAI Realtime WebSocket."""

    def __init__(self, *, api_key: str, model: str) -> None:
        self.api_key = api_key
        self.model = model
        self._connected = asyncio.Event()
        self._receive_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._sender_lock = asyncio.Lock()
        self._mock_task: asyncio.Task[None] | None = None

    async def __aenter__(self) -> "GPTRealtimeClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def connect(self) -> None:
        if self._connected.is_set():
            return
        # TODO: replace with real websocket connection; mocked for now.
        self._mock_task = asyncio.create_task(self._mock_stream())
        self._connected.set()

    async def close(self) -> None:
        if not self._connected.is_set():
            return
        self._connected.clear()
        if self._mock_task:
            self._mock_task.cancel()
            try:
                await self._mock_task
            except asyncio.CancelledError:
                pass
            self._mock_task = None

    async def send_audio_chunk(self, pcm16: bytes, chunk_id: str | None = None) -> None:
        """Placeholder for streaming audio data."""
        await self._connected.wait()
        payload = {
            "type": "input_audio_buffer.append",
            "chunk_id": chunk_id,
            "audio_base64": base64.b64encode(pcm16).decode("ascii"),
        }
        await self._emit_local_event({"event": "audio_chunk_sent", "payload": payload})

    async def commit_input(self) -> None:
        await self._connected.wait()
        await self._emit_local_event({"event": "input_committed"})

    async def request_response(self, *, instructions: str | None = None) -> None:
        await self._connected.wait()
        payload: dict[str, Any] = {
            "type": "response.create",
            "response": {
                "modalities": ["text", "audio"],
            },
        }
        if instructions:
            payload["response"]["instructions"] = instructions
        await self._emit_local_event({"event": "response_requested", "payload": payload})

    async def stream_events(self) -> AsyncGenerator[dict[str, Any], None]:
        """Yield events emitted by the remote model (mocked for now)."""
        while True:
            event = await self._receive_queue.get()
            yield event
            self._receive_queue.task_done()

    async def _emit_local_event(self, event: dict[str, Any]) -> None:
        await self._receive_queue.put({"type": "local.debug", "data": event})

    async def _mock_stream(self) -> None:
        """Mock server events to unblock early integration testing."""
        try:
            await asyncio.sleep(0.1)
            await self._receive_queue.put(
                {
                    "type": "output_text.delta",
                    "content": "[mock] Hello there!",
                }
            )
            await asyncio.sleep(0.1)
            silent_audio = base64.b64encode(b"\x00\x00" * 640).decode("ascii")
            await self._receive_queue.put(
                {
                    "type": "output_audio.delta",
                    "audio_base64": silent_audio,
                }
            )
            await asyncio.sleep(0.05)
            await self._receive_queue.put(
                {
                    "type": "response.completed",
                }
            )
        except asyncio.CancelledError:
            pass
