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
        self._logger = logging.getLogger("ditto.gpt")
        self._has_pending_audio = False

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
        self._logger.info("Connecting to OpenAI realtime endpoint %s", url)
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
            self._logger.error("Realtime connection handshake failed", exc_info=True)
            raise RuntimeError("Failed to connect to OpenAI realtime endpoint") from exc

        self._receiver_task = asyncio.create_task(self._receiver_loop())
        self._connected.set()
        self._logger.info("Realtime websocket connected; sending session.update")
        await self._initialize_session()

    async def close(self) -> None:
        if not self._connected.is_set():
            return
        self._logger.info("Closing realtime websocket")
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
        self._logger.debug("Sending audio chunk bytes=%d chunk_id=%s", len(pcm16), chunk_id)
        await self._send_json(payload)
        await self._emit_local_event({"event": "audio_chunk_sent", "chunk_id": chunk_id})
        self._has_pending_audio = True

    async def commit_input(self) -> bool:
        await self._connected.wait()
        if not self._has_pending_audio:
            self._logger.info("No pending audio to commit; skipping")
            await self._emit_local_event({"event": "input_commit_skipped"})
            return False
        self._logger.info("Committing input audio buffer")
        await self._send_json({"type": "input_audio_buffer.commit"})
        await self._emit_local_event({"event": "input_committed"})
        self._has_pending_audio = False
        return True

    async def request_response(self, *, instructions: str | None = None) -> None:
        await self._connected.wait()
        payload: dict[str, Any] = {
            "type": "response.create",
            "response": {
                "modalities": ["text", "audio"],
                "audio": {"voice": "verse"},
            },
        }
        if instructions:
            payload["response"]["instructions"] = instructions
        self._logger.info("Requesting response (instructions=%s)", instructions)
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
                    "turn_detection": None,
                },
            }
        )
        self._logger.debug("session.update dispatched")

    async def _send_json(self, payload: dict[str, Any]) -> None:
        if self._ws is None:
            raise RuntimeError("Realtime client is not connected")
        message = json.dumps(payload)
        async with self._sender_lock:
            self._logger.debug("Sending JSON payload type=%s", payload.get("type"))
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
            self._logger.info("Realtime websocket closed by server code=%s reason=%s", exc.code, exc.reason)
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
        self._logger.debug("Received realtime JSON event type=%s", event.get("type"))
        if event.get("type") == "input_audio_buffer.committed":
            self._logger.debug("Upstream acknowledged audio commit; clearing pending flag")
            self._has_pending_audio = False
        if event.get("type") == "response.audio.delta" and "delta" in event:
            self._logger.debug("Decoding JSON audio delta")
            await self._emit_remote_event(
                {
                    "type": "audio.chunk",
                    "data": event["delta"],
                }
            )
        await self._emit_remote_event(event)

    async def _handle_binary_message(self, payload: bytes) -> None:
        self._logger.debug("Received realtime binary payload len=%d", len(payload))
        await self._emit_remote_event(
            {
                "type": "binary.delta",
                "data": base64.b64encode(payload).decode("ascii"),
            }
        )

    async def _emit_remote_event(self, event: dict[str, Any]) -> None:
        self._logger.debug("Queueing remote event type=%s", event.get("type"))
        await self._receive_queue.put(event)
