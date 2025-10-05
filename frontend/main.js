const statusEl = document.getElementById('status');
const transcriptEl = document.getElementById('transcript');
const logEl = document.getElementById('log');
const instructionsEl = document.getElementById('instructions');
const startButton = document.getElementById('start-button');
const stopButton = document.getElementById('stop-button');
const resetButton = document.getElementById('reset-button');

const TARGET_SAMPLE_RATE = 16000;
const FRAME_DURATION_MS = 20;
const FRAME_SIZE = Math.floor((TARGET_SAMPLE_RATE * FRAME_DURATION_MS) / 1000); // 320 samples
const OUTPUT_SAMPLE_RATE = 16000;
const MAX_WS_BUFFERED_AMOUNT = 2 * 1024 * 1024; // 2MB safeguard

const wsUrl = `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}/ws/realtime`;

let audioContext;
let mediaStream;
let mediaSource;
let workletNode;
let workletLoaded = false;
let websocket;
let sessionId = null;
let isCapturing = false;
let sampleAccumulator = new Float32Array(0);
let playbackTime = 0;
let playbackSources = [];
let transcriptBuffer = '';

startButton.addEventListener('click', () => {
  startStreaming().catch((error) => {
    console.error(error);
    appendLog('error', `Start failed: ${error.message ?? error}`);
    setStatus('Failed to start streaming');
    toggleControls({ start: false, stop: true, reset: true });
  });
});

stopButton.addEventListener('click', () => {
  stopStreaming().catch((error) => {
    console.error(error);
    appendLog('error', `Stop failed: ${error.message ?? error}`);
  });
});

resetButton.addEventListener('click', () => {
  resetSession().catch((error) => {
    console.error(error);
    appendLog('error', `Reset failed: ${error.message ?? error}`);
  });
});

function setStatus(message) {
  statusEl.textContent = message;
}

function appendLog(kind, message) {
  const entry = document.createElement('div');
  entry.className = 'log-entry';
  entry.innerHTML = `<strong>[${new Date().toLocaleTimeString()}] ${kind}:</strong> ${message}`;
  logEl.appendChild(entry);
  logEl.scrollTop = logEl.scrollHeight;
  while (logEl.children.length > 200) {
    logEl.removeChild(logEl.firstChild);
  }
}

function toggleControls({ start, stop, reset }) {
  startButton.disabled = start ?? startButton.disabled;
  stopButton.disabled = stop ?? stopButton.disabled;
  resetButton.disabled = reset ?? resetButton.disabled;
}

async function startStreaming() {
  if (websocket && websocket.readyState === WebSocket.OPEN) {
    await resetSession();
  }

  setStatus('Connecting to backend...');
  toggleControls({ start: true, stop: true, reset: true });

  websocket = new WebSocket(wsUrl);
  websocket.binaryType = 'arraybuffer';

  websocket.onopen = async () => {
    appendLog('socket', 'WebSocket connected');
    try {
      await initAudioCapture();
      await audioContext.resume();
      isCapturing = true;
      setStatus('Streaming microphone audio');
      toggleControls({ start: true, stop: false, reset: false });
    } catch (error) {
      appendLog('error', `Microphone init failed: ${error.message ?? error}`);
      setStatus('Microphone access denied or unavailable');
      await resetSession();
    }
  };

  websocket.onmessage = (event) => {
    if (typeof event.data === 'string') {
      handleBackendMessage(event.data);
    } else if (event.data instanceof ArrayBuffer) {
      queuePlaybackFromBuffer(event.data);
    }
  };

  websocket.onerror = (event) => {
    console.error('WebSocket error', event);
    appendLog('socket', 'WebSocket error encountered');
  };

  websocket.onclose = (event) => {
    appendLog('socket', `WebSocket closed (code=${event.code}, reason=${event.reason || 'none'})`);
    cleanupSocket();
    toggleControls({ start: false, stop: true, reset: true });
    setStatus('Session closed');
  };
}

async function initAudioCapture() {
  if (!audioContext || audioContext.state === 'closed') {
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
  }

  if (!workletLoaded) {
    await audioContext.audioWorklet.addModule('audio-worklet.js');
    workletLoaded = true;
  }

  mediaStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    },
  });

  mediaSource = audioContext.createMediaStreamSource(mediaStream);
  workletNode = new AudioWorkletNode(audioContext, 'pcm-worklet');
  workletNode.port.onmessage = (event) => handleAudioChunk(event.data);
  mediaSource.connect(workletNode);
}

async function stopStreaming() {
  if (!websocket || websocket.readyState !== WebSocket.OPEN) {
    appendLog('socket', 'No active websocket session to stop');
    return;
  }

  if (!isCapturing) {
    appendLog('client', 'Already stopped recording; awaiting response');
    return;
  }

  appendLog('client', 'Stopping capture and committing request');
  stopAudioCapture();
  flushPendingSamples();
  const payload = { type: 'commit' };
  const instructions = instructionsEl.value.trim();
  if (instructions) {
    payload.instructions = instructions;
  }
  websocket.send(JSON.stringify(payload));
  setStatus('Awaiting model response...');
  toggleControls({ start: true, stop: true, reset: false });
}

async function resetSession() {
  appendLog('client', 'Resetting session');
  stopAudioCapture();
  cleanupPlayback();
  sampleAccumulator = new Float32Array(0);
  transcriptBuffer = '';
  updateTranscript();

  if (websocket && websocket.readyState === WebSocket.OPEN) {
    websocket.send(JSON.stringify({ type: 'close' }));
    websocket.close(1000, 'client reset');
  }

  cleanupSocket();
  toggleControls({ start: false, stop: true, reset: true });
  setStatus('Idle');
}

function stopAudioCapture() {
  if (workletNode) {
    workletNode.port.onmessage = null;
    try {
      workletNode.disconnect();
    } catch (error) {
      console.debug('Worklet disconnect error', error);
    }
    workletNode = null;
  }

  if (mediaSource) {
    try {
      mediaSource.disconnect();
    } catch (error) {
      console.debug('Media source disconnect error', error);
    }
    mediaSource = null;
  }

  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
  }

  isCapturing = false;
}

function cleanupSocket() {
  if (websocket) {
    websocket.onopen = null;
    websocket.onclose = null;
    websocket.onmessage = null;
    websocket.onerror = null;
    if (websocket.readyState === WebSocket.OPEN) {
      websocket.close(1000, 'cleanup');
    }
    websocket = null;
  }
  sessionId = null;
}

function cleanupPlayback() {
  playbackSources.forEach((source) => {
    try {
      source.stop();
    } catch (error) {
      console.debug('Source stop error', error);
    }
  });
  playbackSources = [];
  playbackTime = audioContext ? audioContext.currentTime : 0;
}

function handleAudioChunk(chunk) {
  if (!(chunk instanceof Float32Array)) {
    return;
  }

  const downsampled = downsampleBuffer(chunk, audioContext.sampleRate, TARGET_SAMPLE_RATE);
  if (!downsampled || downsampled.length === 0) {
    return;
  }

  sampleAccumulator = appendFloat32(sampleAccumulator, downsampled);

  while (sampleAccumulator.length >= FRAME_SIZE) {
    const frame = sampleAccumulator.slice(0, FRAME_SIZE);
    sampleAccumulator = sampleAccumulator.slice(FRAME_SIZE);
    const pcm = floatTo16BitPCM(frame);
    sendPcmChunk(pcm);
  }
}

function appendFloat32(a, b) {
  if (a.length === 0) {
    return b;
  }
  if (b.length === 0) {
    return a;
  }
  const result = new Float32Array(a.length + b.length);
  result.set(a, 0);
  result.set(b, a.length);
  return result;
}

function sendPcmChunk(buffer) {
  if (!websocket || websocket.readyState !== WebSocket.OPEN) {
    return;
  }
  if (websocket.bufferedAmount > MAX_WS_BUFFERED_AMOUNT) {
    console.warn('Dropping audio chunk due to websocket backpressure');
    appendLog('warn', 'Dropping audio chunk (WS backpressure)');
    return;
  }
  websocket.send(buffer);
}

function floatTo16BitPCM(float32Array) {
  const buffer = new ArrayBuffer(float32Array.length * 2);
  const view = new DataView(buffer);
  for (let i = 0; i < float32Array.length; i += 1) {
    let sample = float32Array[i];
    sample = Math.max(-1, Math.min(1, sample));
    view.setInt16(i * 2, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
  }
  return buffer;
}

function flushPendingSamples() {
  if (!websocket || websocket.readyState !== WebSocket.OPEN) {
    return;
  }
  if (sampleAccumulator.length === 0) {
    return;
  }
  const pcm = floatTo16BitPCM(sampleAccumulator);
  websocket.send(pcm);
  sampleAccumulator = new Float32Array(0);
}

function downsampleBuffer(buffer, inSampleRate, outSampleRate) {
  if (!buffer || inSampleRate === outSampleRate) {
    return buffer;
  }
  if (outSampleRate > inSampleRate) {
    throw new Error('Output sample rate must be <= input sample rate');
  }
  const ratio = inSampleRate / outSampleRate;
  const newLength = Math.round(buffer.length / ratio);
  const result = new Float32Array(newLength);
  let offsetResult = 0;
  let offsetBuffer = 0;
  while (offsetResult < newLength) {
    const nextOffset = Math.round((offsetResult + 1) * ratio);
    let accum = 0;
    let count = 0;
    for (let i = offsetBuffer; i < nextOffset && i < buffer.length; i += 1) {
      accum += buffer[i];
      count += 1;
    }
    result[offsetResult] = count > 0 ? accum / count : 0;
    offsetResult += 1;
    offsetBuffer = nextOffset;
  }
  return result;
}

function handleBackendMessage(raw) {
  let event;
  try {
    event = JSON.parse(raw);
  } catch (error) {
    appendLog('warn', `Non-JSON event: ${raw}`);
    return;
  }

  switch (event.type) {
    case 'session.created':
      sessionId = event.session_id;
      appendLog('session', `Session created (${sessionId})`);
      break;
    case 'local.debug':
      appendLog('debug', JSON.stringify(event.data));
      break;
    case 'response.output_text.delta':
      transcriptBuffer += event.delta ?? event.text ?? event.content ?? '';
      updateTranscript();
      break;
    case 'response.output_text.done':
      appendLog('event', 'Text output complete');
      break;
    case 'response.output_audio.delta': {
      const base64 = event.delta ?? event.audio ?? event.audio_base64;
      if (base64) {
        queuePlaybackFromBase64(base64);
      }
      break;
    }
    case 'response.audio.delta': {
      const base64Audio = event.delta ?? event.audio ?? event.audio_base64;
      if (base64Audio) {
        queuePlaybackFromBase64(base64Audio);
      }
      break;
    }
    case 'response.audio_transcript.delta': {
      const delta =
        event.delta ?? event.text ?? event.transcript ?? event.content ?? '';
      transcriptBuffer += delta;
      updateTranscript();
      break;
    }
    case 'response.audio_transcript.done':
      appendLog('event', 'Audio transcript complete');
      break;
    case 'response.audio.done':
      appendLog('event', 'Audio stream complete');
      break;
    case 'binary.delta':
      if (event.data) {
        queuePlaybackFromBase64(event.data);
      }
      break;
    case 'response.completed':
      appendLog('event', 'Model response completed');
      setStatus('Model response completed');
      toggleControls({ start: false, stop: true, reset: false });
      if (websocket && websocket.readyState === WebSocket.OPEN) {
        websocket.send(JSON.stringify({ type: 'close' }));
      }
      break;
    case 'invalid_request_error':
      appendLog('error', JSON.stringify(event));
      setStatus(event.message ?? 'Invalid request');
      break;
    case 'error':
      appendLog('error', JSON.stringify(event.error ?? event));
      setStatus('Error received from backend');
      break;
    default:
      appendLog('event', `${event.type}`);
      break;
  }
}

function updateTranscript() {
  transcriptEl.textContent = transcriptBuffer || '(awaiting output)';
}

function queuePlaybackFromBase64(base64) {
  try {
    const arrayBuffer = base64ToArrayBuffer(base64);
    queuePlaybackFromBuffer(arrayBuffer);
  } catch (error) {
    appendLog('error', `Failed to decode audio chunk: ${error.message ?? error}`);
  }
}

function queuePlaybackFromBuffer(arrayBuffer) {
  if (!audioContext) {
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
  }

  if (arrayBuffer.byteLength === 0) {
    return;
  }

  const int16Array = new Int16Array(arrayBuffer.slice(0));
  const float32 = new Float32Array(int16Array.length);
  for (let i = 0; i < int16Array.length; i += 1) {
    float32[i] = int16Array[i] / 0x8000;
  }

  const audioBuffer = audioContext.createBuffer(1, float32.length, OUTPUT_SAMPLE_RATE);
  audioBuffer.copyToChannel(float32, 0);

  const source = audioContext.createBufferSource();
  source.buffer = audioBuffer;
  source.connect(audioContext.destination);

  if (playbackTime < audioContext.currentTime) {
    playbackTime = audioContext.currentTime;
  }

  source.start(playbackTime);
  playbackTime += audioBuffer.duration;

  playbackSources.push(source);
  source.onended = () => {
    playbackSources = playbackSources.filter((node) => node !== source);
  };
}

function base64ToArrayBuffer(base64) {
  const binary = window.atob(base64);
  const len = binary.length;
  const buffer = new ArrayBuffer(len);
  const view = new Uint8Array(buffer);
  for (let i = 0; i < len; i += 1) {
    view[i] = binary.charCodeAt(i);
  }
  return buffer;
}
