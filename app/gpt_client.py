from __future__ import annotations

import asyncio
import base64
import inspect
import json
import logging
from typing import Any, AsyncGenerator

import websockets
from websockets.client import WebSocketClientProtocol
from websockets.exceptions import ConnectionClosed


class GPTRealtimeClient:
    """Async client skeleton for OpenAI Realtime WebSocket."""

    def __init__(self, *, api_key: str, model: str) -> None:
        self.api_key = api_key
        self.model = model
        self._connected = asyncio.Event()
        self._receive_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._sender_lock = asyncio.Lock()
        self._ws: WebSocketClientProtocol | None = None
        self._receiver_task: asyncio.Task[None] | None = None
        self._logger = logging.getLogger(__name__)

    async def __aenter__(self) -> "GPTRealtimeClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def connect(self) -> None:
        if self._connected.is_set():
            return
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for realtime streaming")

        url = f"wss://api.openai.com/v1/realtime?model={self.model}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "realtime=v1",
        }

        connect_signature = inspect.signature(websockets.connect)
        header_kwarg = "extra_headers" if "extra_headers" in connect_signature.parameters else "additional_headers"

        try:
            self._ws = await websockets.connect(
                url,
                max_size=16 * 1024 * 1024,
                ping_interval=20,
                ping_timeout=20,
                **{header_kwarg: headers},
            )
        except Exception as exc:  # pragma: no cover - network boundary
            raise RuntimeError("Failed to connect to OpenAI realtime endpoint") from exc

        self._receiver_task = asyncio.create_task(self._receiver_loop())
        self._connected.set()
        await self._initialize_session()

    async def close(self) -> None:
        if not self._connected.is_set():
            return
        self._connected.clear()
        if self._receiver_task:
            self._receiver_task.cancel()
            try:
                await self._receiver_task
            except asyncio.CancelledError:
                pass
            self._receiver_task = None
        if self._ws is not None:
            try:
                await self._ws.close(code=1000)
            except ConnectionClosed:
                pass
            self._ws = None

    async def send_audio_chunk(self, pcm16: bytes, chunk_id: str | None = None) -> None:
        """Stream a PCM16 audio chunk to the realtime session."""
        await self._connected.wait()
        payload = {
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(pcm16).decode("ascii"),
        }
        if chunk_id is not None:
            payload["chunk_id"] = chunk_id
        await self._send_json(payload)
        await self._emit_local_event({"event": "audio_chunk_sent", "chunk_id": chunk_id})

    async def commit_input(self) -> None:
        await self._connected.wait()
        await self._send_json({"type": "input_audio_buffer.commit"})
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
        await self._send_json(payload)
        await self._emit_local_event({"event": "response_requested", "payload": payload})

    async def stream_events(self) -> AsyncGenerator[dict[str, Any], None]:
        """Yield events emitted by the remote model."""
        while True:
            event = await self._receive_queue.get()
            yield event
            self._receive_queue.task_done()

    async def _emit_local_event(self, event: dict[str, Any]) -> None:
        await self._receive_queue.put({"type": "local.debug", "data": event})

    async def _initialize_session(self) -> None:
        await self._send_json(
            {
                "type": "session.update",
                "session": {
                    "modalities": ["text", "audio"],
                    "input_audio_format": "pcm16",
                    "input_audio_sample_rate": 16000,
                    "voice": "verse",
                },
            }
        )

    async def _send_json(self, payload: dict[str, Any]) -> None:
        if self._ws is None:
            raise RuntimeError("Realtime client is not connected")
        message = json.dumps(payload)
        async with self._sender_lock:
            await self._ws.send(message)

    async def _receiver_loop(self) -> None:
        assert self._ws is not None
        try:
            async for message in self._ws:
                if isinstance(message, bytes):
                    await self._handle_binary_message(message)
                else:
                    await self._handle_text_message(message)
        except asyncio.CancelledError:
            raise
        except ConnectionClosed as exc:  # pragma: no cover - network
            await self._emit_remote_event(
                {
                    "type": "connection.closed",
                    "code": exc.code,
                    "reason": exc.reason,
                }
            )
        except Exception as exc:  # pragma: no cover - network boundary
            self._logger.exception("Realtime receiver loop crashed")
            await self._emit_remote_event(
                {
                    "type": "error",
                    "error": {
                        "message": str(exc),
                        "kind": exc.__class__.__name__,
                    },
                }
            )

    async def _handle_text_message(self, message: str) -> None:
        try:
            event = json.loads(message)
        except json.JSONDecodeError:
            await self._emit_remote_event(
                {
                    "type": "error",
                    "error": {
                        "message": "Failed to decode JSON from realtime stream",
                    },
                    "raw": message,
                }
            )
            return
        await self._emit_remote_event(event)

    async def _handle_binary_message(self, payload: bytes) -> None:
        await self._emit_remote_event(
            {
                "type": "binary.delta",
                "data": base64.b64encode(payload).decode("ascii"),
            }
        )

    async def _emit_remote_event(self, event: dict[str, Any]) -> None:
        await self._receive_queue.put(event)
