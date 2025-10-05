from __future__ import annotations

import asyncio
import contextlib
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncGenerator, Iterable, Optional

import numpy as np

from .config import get_settings

try:
    from stream_pipeline_online import StreamSDK as OnlineStreamSDK  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    OnlineStreamSDK = None  # type: ignore


FrameArray = np.ndarray


@dataclass
class AvatarFrame:
    index: int
    data: FrameArray


class BaseAvatarSession:
    def __init__(self, session_id: str, loop: asyncio.AbstractEventLoop | None = None) -> None:
        self.session_id = session_id
        self._loop = loop or asyncio.get_event_loop()
        self._ready_event = asyncio.Event()
        self._closed = False
        self._error: Exception | None = None

    async def wait_until_ready(self) -> None:
        await self._ready_event.wait()
        if self._error:
            raise self._error

    async def enqueue_audio_chunk(self, audio: bytes | np.ndarray | Iterable[float]) -> None:
        raise NotImplementedError

    async def finalize(self) -> None:
        raise NotImplementedError

    async def frame_stream(self) -> AsyncGenerator[AvatarFrame, None]:
        raise NotImplementedError

    def set_error(self, exc: Exception) -> None:
        self._error = exc
        self._loop.call_soon_threadsafe(self._ready_event.set)

    def mark_ready(self) -> None:
        self._loop.call_soon_threadsafe(self._ready_event.set)

    @property
    def error(self) -> Exception | None:
        return self._error


class MockAvatarSession(BaseAvatarSession):
    def __init__(self, session_id: str, loop: asyncio.AbstractEventLoop) -> None:
        super().__init__(session_id, loop)
        self._frame_queue: asyncio.Queue[AvatarFrame | None] = asyncio.Queue()
        self._sample_count = 0
        self.mark_ready()

    async def enqueue_audio_chunk(self, audio: bytes | np.ndarray | Iterable[float]) -> None:
        if self._closed:
            return
        arr = _to_float_mono(audio)
        self._sample_count += len(arr)

    async def finalize(self) -> None:
        if self._closed:
            return
        self._closed = True
        frame = np.zeros((256, 256, 3), dtype=np.uint8)
        text = f"Mock frame {self._sample_count // 16000}s"
        _draw_text(frame, text)
        await self._frame_queue.put(AvatarFrame(index=0, data=frame))
        await self._frame_queue.put(None)

    async def frame_stream(self) -> AsyncGenerator[AvatarFrame, None]:
        while True:
            frame = await self._frame_queue.get()
            if frame is None:
                break
            yield frame


class DittoAvatarSession(BaseAvatarSession):
    def __init__(
        self,
        session_id: str,
        *,
        loop: asyncio.AbstractEventLoop,
        cfg_pkl: Path,
        data_root: Path,
        source_path: Path,
        output_dir: Path,
        frame_queue_size: int,
        chunksize: tuple[int, int, int],
        setup_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(session_id, loop)
        if OnlineStreamSDK is None:
            raise RuntimeError("Ditto StreamSDK is unavailable (import failed)")
        self._cfg_pkl = cfg_pkl
        self._data_root = data_root
        self._source_path = source_path
        self._output_dir = output_dir
        self._frame_queue: asyncio.Queue[AvatarFrame | None] = asyncio.Queue(maxsize=frame_queue_size)
        self._audio_queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=frame_queue_size * 4 or 8)
        self._chunksize = chunksize
        self._setup_kwargs = setup_kwargs or {}
        self._thread: threading.Thread | None = None
        self._sdk: OnlineStreamSDK | None = None  # type: ignore
        self._stop_event = threading.Event()
        self._ready_flag = threading.Event()
        self._frame_index = 0
        self._buffer = np.zeros((0,), dtype=np.float32)
        self._prepad = np.zeros((self._chunksize[0] * 640,), dtype=np.float32)
        self._chunk_stride = self._chunksize[1] * 640
        self._split_len = int(sum(self._chunksize) * 0.04 * 16000) + 80
        self._ended = False

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name=f"ditto-session-{self.session_id}", daemon=True)
        self._thread.start()

    async def enqueue_audio_chunk(self, audio: bytes | np.ndarray | Iterable[float]) -> None:
        await self.wait_until_ready()
        if self._closed:
            return
        chunk = _to_float_mono(audio)
        await asyncio.to_thread(self._audio_queue.put, chunk)

    async def finalize(self) -> None:
        if self._closed:
            return
        if not self._ready_event.is_set():
            try:
                await self.wait_until_ready()
            except Exception:
                pass
        self._closed = True
        await asyncio.to_thread(self._audio_queue.put, None)
        if self._thread:
            await asyncio.to_thread(self._thread.join)

    async def frame_stream(self) -> AsyncGenerator[AvatarFrame, None]:
        await self.wait_until_ready()
        while True:
            frame = await self._frame_queue.get()
            if frame is None:
                if self._error:
                    raise self._error
                break
            yield frame

    def _run(self) -> None:
        try:
            self._output_dir.mkdir(parents=True, exist_ok=True)
            tmp_output = self._output_dir / f"{self.session_id}.mp4"
            sdk = OnlineStreamSDK(str(self._cfg_pkl), str(self._data_root))  # type: ignore
            setup_kwargs = dict(self._setup_kwargs)
            setup_kwargs.setdefault("online_mode", True)
            setup_kwargs.setdefault("N_d", -1)
            with self._patched_writer(sdk):
                sdk.setup(str(self._source_path), str(tmp_output), **setup_kwargs)
            sdk.setup_Nd(N_d=-1, ctrl_info={})
            self._sdk = sdk
            self._buffer = self._prepad.copy()
            self._notify_ready()
            self._process_audio_loop()
            sdk.close()
        except Exception as exc:  # pragma: no cover - heavy pipeline
            self._set_error(exc)
        finally:
            self._emit_end()

    def _process_audio_loop(self) -> None:
        assert self._sdk is not None
        while not self._stop_event.is_set():
            try:
                item = self._audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                self._flush_buffer(final=True)
                break
            self._buffer = np.concatenate([self._buffer, item])
            self._flush_buffer(final=False)
        self._stop_event.set()

    def _flush_buffer(self, *, final: bool) -> None:
        assert self._sdk is not None
        while len(self._buffer) >= self._split_len:
            chunk = self._buffer[: self._split_len]
            self._run_chunk(chunk)
            self._buffer = self._buffer[self._chunk_stride :]
        if final and len(self._buffer) > 0:
            pad_len = max(self._split_len - len(self._buffer), 0)
            if pad_len > 0:
                chunk = np.pad(self._buffer, (0, pad_len), mode="constant")
            else:
                chunk = self._buffer[: self._split_len]
            self._run_chunk(chunk)
            self._buffer = np.zeros((0,), dtype=np.float32)

    def _run_chunk(self, chunk: np.ndarray) -> None:
        assert self._sdk is not None
        chunk = chunk.astype(np.float32, copy=False)
        self._sdk.run_chunk(chunk, self._chunksize)

    def _notify_ready(self) -> None:
        self._ready_flag.set()
        self._loop.call_soon_threadsafe(self._ready_event.set)

    def _emit_frame(self, frame: np.ndarray, fmt: str = "rgb") -> None:
        if self._ended:
            return
        if fmt == "bgr":
            frame = frame[..., ::-1]
        frame = np.ascontiguousarray(frame)
        avatar_frame = AvatarFrame(index=self._frame_index, data=frame)
        self._frame_index += 1

        def _put() -> None:
            if self._ended:
                return
            try:
                if self._frame_queue.full():
                    self._frame_queue.get_nowait()
                self._frame_queue.put_nowait(avatar_frame)
            except asyncio.QueueFull:
                try:
                    self._frame_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                self._frame_queue.put_nowait(avatar_frame)
            except asyncio.QueueEmpty:
                self._frame_queue.put_nowait(avatar_frame)

        self._loop.call_soon_threadsafe(_put)

    def _emit_end(self) -> None:
        def _put_end() -> None:
            if self._ended:
                return
            self._ended = True
            try:
                self._frame_queue.put_nowait(None)
            except asyncio.QueueFull:
                try:
                    self._frame_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                self._frame_queue.put_nowait(None)

        self._loop.call_soon_threadsafe(_put_end)

    def _set_error(self, exc: Exception) -> None:
        self._error = exc
        self._loop.call_soon_threadsafe(self._ready_event.set)

    @contextlib.contextmanager
    def _patched_writer(self, sdk: OnlineStreamSDK):  # type: ignore[override]
        import stream_pipeline_online as spo

        original_writer = spo.VideoWriterByImageIO
        session = self

        class StreamingVideoWriter:
            def __init__(self, *_args: Any, **_kwargs: Any) -> None:
                self._session = session

            def __call__(self, img: np.ndarray, fmt: str = "rgb") -> None:
                self._session._emit_frame(img, fmt=fmt)

            def close(self) -> None:
                pass

        try:
            spo.VideoWriterByImageIO = StreamingVideoWriter
            yield
        finally:
            spo.VideoWriterByImageIO = original_writer


class AvatarEngine:
    def __init__(self) -> None:
        self._settings = get_settings()
        self._enabled = (
            OnlineStreamSDK is not None
            and self._settings.ditto_cfg_pkl
            and self._settings.ditto_data_root
            and self._settings.ditto_source_path
        )
        self._last_error: Exception | None = None

    async def create_session(self, session_id: str, *, source_path: Optional[str] = None) -> BaseAvatarSession:
        loop = asyncio.get_running_loop()
        if not self._enabled:
            return MockAvatarSession(session_id, loop)

        cfg_pkl = Path(self._settings.ditto_cfg_pkl)
        data_root = Path(self._settings.ditto_data_root)
        src = Path(source_path or self._settings.ditto_source_path)
        output_dir = Path(self._settings.ditto_output_dir)
        chunksize = tuple(self._settings.ditto_chunksize)
        frame_queue_size = self._settings.ditto_frame_queue_size

        session = DittoAvatarSession(
            session_id,
            loop=loop,
            cfg_pkl=cfg_pkl,
            data_root=data_root,
            source_path=src,
            output_dir=output_dir,
            frame_queue_size=frame_queue_size,
            chunksize=chunksize,
            setup_kwargs={},
        )
        session.start()
        try:
            await session.wait_until_ready()
        except Exception as exc:
            self._last_error = exc
            try:
                await session.finalize()
            except Exception:
                pass
            return MockAvatarSession(session_id, loop)
        return session

    @property
    def last_error(self) -> Exception | None:
        return self._last_error


def _to_float_mono(audio: bytes | np.ndarray | Iterable[float]) -> np.ndarray:
    if isinstance(audio, bytes):
        arr = np.frombuffer(audio, dtype=np.int16)
        arr = arr.astype(np.float32) / 32768.0
    else:
        arr = np.asarray(list(audio) if not isinstance(audio, np.ndarray) else audio, dtype=np.float32)
        if arr.ndim > 1:
            arr = arr[..., 0]
    return np.ascontiguousarray(arr, dtype=np.float32)


def _draw_text(img: np.ndarray, text: str) -> None:
    try:
        import cv2  # type: ignore

        cv2.putText(img, text, (10, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    except Exception:
        pass
