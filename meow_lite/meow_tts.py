"""Text-to-meow audio: concatenative synthesis from real cat clips.

Turns a generated response string (e.g. ``"<bite> Mrrp. Meow"``) into a wav
by concatenating clips from the prepared bank in ``assets/audio``. Engine
agnostic — input is text, not a model. numpy + stdlib only.

The clip bank is produced by ``tools/prepare_meow_audio.py`` (see
``clips.json`` + ``clips/<bucket>/``). Rendering is deterministic: the clip
selection is seeded by sha256(text), so the same text always yields the same
bytes.
"""

import io
import json
import os
import random
import re
import wave
from hashlib import sha256
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent

SAMPLE_RATE = 16_000
WORD_GAP_MS = 40
PUNCT_SILENCE_MS = 120
PEAK_TARGET_DB = -3.0  # normalize peak to -3 dBFS

# Action token -> candidate buckets. Two-bucket entries alternate
# deterministically per occurrence (bite-bucket / hiss-bucket).
ACTION_BUCKETS: dict[str, list[str]] = {
    "<purr>": ["purr"],
    "<hiss>": ["hiss"],
    "<bite>": ["bite", "hiss"],
    "<scratch>": ["bite", "hiss"],
    "<scratch_couch>": ["bite", "hiss"],
    "<knock_glass>": ["hiss", "bite"],  # own acoustic space — never shares a file with bite
    "<hairball>": ["hiss", "bite"],
    "<zoomies>": ["meow"],
    "<stare>": ["meow"],
    "<pounce>": ["meow"],
}
_LONGEST_FIRST = sorted(ACTION_BUCKETS, key=len, reverse=True)
_ACTION_RE = re.compile(
    "(" + "|".join(re.escape(token) for token in _LONGEST_FIRST) + ")",
    re.IGNORECASE,
)
_WORD_RE = re.compile(r"[A-Za-z']+|[.,!?;:]")

DEFAULT_CLIPS_DIR = _REPO_ROOT / "assets" / "audio"


def clips_dir() -> Path:
    """Resolve the clip bank directory (env MEOW_AUDIO_DIR, default assets/audio)."""
    return Path(os.environ.get("MEOW_AUDIO_DIR", str(DEFAULT_CLIPS_DIR)))


def tokenize_units(text: str) -> list[tuple[str, str]]:
    """Split a response string into (kind, unit) pairs.

    Kinds: ``action`` (single lowercase action token, mixed case allowed),
    ``word`` (meow word), ``punct`` (single terminal/separator punctuation).
    """
    units: list[tuple[str, str]] = []
    for part in _ACTION_RE.split(text):
        if not part:
            continue
        lowered = part.lower()
        if lowered in ACTION_BUCKETS:
            units.append(("action", lowered))
            continue
        for match in _WORD_RE.findall(part):
            if match.isalpha() or "'" in match:
                units.append(("word", match.lower()))
            else:
                units.append(("punct", match))
    return units


def group_units(units: list[tuple[str, str]], size: int = 2) -> list[tuple[str, str]]:
    """Group consecutive prose words into phrases of ``size`` words per clip.

    Word-per-clip turns chatty responses into a meow machine gun; grouping
    halves the rate while action tokens stay solo (they are the emphasis).
    """
    grouped: list[tuple[str, str]] = []
    pending: list[str] = []
    for kind, unit in units:
        if kind == "word":
            pending.append(unit)
            if len(pending) == size:
                grouped.append(("word", " ".join(pending)))
                pending = []
        else:
            if pending:
                grouped.append(("word", " ".join(pending)))
                pending = []
            grouped.append((kind, unit))
    if pending:
        grouped.append(("word", " ".join(pending)))
    return grouped


def load_wav_mono16k(path: Path, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Decode any PCM wav (8/16/24/32-bit, mono/stereo) to mono 16k int16."""
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sampwidth = wav.getsampwidth()
        rate = wav.getframerate()
        raw = wav.readframes(wav.getnframes())

    if sampwidth == 1:  # unsigned 8-bit, centered at 128
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0) * 256.0
    elif sampwidth == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float64)
    elif sampwidth == 3:  # 24-bit little-endian, sign-extend
        b = np.frombuffer(raw, dtype=np.uint8)
        b = b[: len(b) - len(b) % 3].reshape(-1, 3).astype(np.int32)
        x = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        x = np.where(x >= 1 << 23, x - (1 << 24), x).astype(np.float64)
    elif sampwidth == 4:
        x = np.frombuffer(raw, dtype="<i4").astype(np.float64)
    else:
        raise wave.Error(f"unsupported sample width: {sampwidth}")

    if channels > 1:
        x = x.reshape(-1, channels).mean(axis=1)

    if rate != target_sr and len(x) > 0:
        n_out = max(1, int(round(len(x) * target_sr / rate)))
        x = np.interp(
            np.linspace(0.0, len(x) - 1.0, n_out), np.arange(len(x), dtype=np.float64), x
        )
    return np.clip(x, -32768.0, 32767.0).astype(np.int16)


def load_clips(directory: Path | None = None) -> dict[str, list[dict]]:
    """Load the clip bank index from ``clips.json``.

    Returns {bucket: [{"path": Path, "duration": float}, ...]} in deterministic
    (filename) order. Raises FileNotFoundError when the bank was never prepared.
    """
    base = Path(directory) if directory is not None else clips_dir()
    manifest = json.loads((base / "clips.json").read_text(encoding="utf-8"))
    index: dict[str, list[dict]] = {}
    for entry in manifest.get("clips", []):
        rel = entry.get("path")
        if not rel:
            continue  # analyzed but not banked
        index.setdefault(entry["bucket"], []).append(
            {"path": base / rel, "duration": float(entry["duration"])}
        )
    for bucket in index:
        index[bucket].sort(key=lambda clip: clip["path"].name)
    return index


class _ClipCycler:
    """Deterministic per-bucket clip picker: seeded shuffle + wraparound."""

    def __init__(self, rng: random.Random) -> None:
        self._rng = rng
        self._orders: dict[str, list[dict]] = {}
        self._cursors: dict[str, int] = {}
        self._action_counts: dict[str, int] = {}

    def _order(self, bucket: str, index: dict[str, list[dict]]) -> list[dict]:
        if bucket not in self._orders:
            order = list(index[bucket])
            self._rng.shuffle(order)
            self._orders[bucket] = order
            self._cursors[bucket] = 0
        return self._orders[bucket]

    def buckets_for(self, kind: str, unit: str) -> list[str]:
        if kind == "action":
            buckets = ACTION_BUCKETS[unit]
            count = self._action_counts.get(unit, 0)
            self._action_counts[unit] = count + 1
            # Two-bucket actions alternate deterministically per occurrence.
            shift = count % len(buckets)
            return buckets[shift:] + buckets[:shift]
        return ["meow"]

    def pick(self, kind: str, unit: str, index: dict[str, list[dict]]) -> dict:
        for bucket in self.buckets_for(kind, unit):
            candidates = index.get(bucket)
            if not candidates:
                continue  # bucket absent from the bank; try the next
            order = self._order(bucket, index)
            clip = order[self._cursors[bucket] % len(order)]
            self._cursors[bucket] += 1
            return clip
        # Bank fallback: meow bucket, then any non-empty bucket.
        for bucket in ["meow", *sorted(index)]:
            if index.get(bucket):
                order = self._order(bucket, index)
                clip = order[self._cursors[bucket] % len(order)]
                self._cursors[bucket] += 1
                return clip
        raise RuntimeError("clip bank has no usable clips")


def _silence(ms: int) -> np.ndarray:
    return np.zeros(int(SAMPLE_RATE * ms / 1000), dtype=np.int16)


def render(
    text: str,
    directory: Path | None = None,
    out_format: str = "wav",
    seed_salt: str = "",
) -> bytes:
    """Render ``text`` to wav bytes by concatenating clips from the bank.

    Deterministic: seeded by sha256(text). Punctuation becomes 120 ms silence,
    gaps between consecutive sounded units are 40 ms. Output is 16 kHz mono
    16-bit PCM, peak-normalized to -3 dBFS.
    """
    if out_format != "wav":
        raise ValueError(f"unsupported audio format: {out_format!r}")

    index = load_clips(directory)  # FileNotFoundError -> caller handles (503)
    units = group_units(tokenize_units(text))
    rng = random.Random(
        int.from_bytes(sha256((text + "|" + seed_salt).encode("utf-8")).digest()[:8], "big")
    )
    cycler = _ClipCycler(rng)

    pcm_cache: dict[Path, np.ndarray] = {}

    def clip_pcm(clip: dict) -> np.ndarray:
        path = clip["path"]
        if path not in pcm_cache:
            pcm_cache[path] = load_wav_mono16k(path)
        return pcm_cache[path]

    pieces: list[np.ndarray] = []
    for kind, unit in units:
        if kind == "punct":
            pieces.append(_silence(PUNCT_SILENCE_MS))
            continue
        if pieces:  # word gap between consecutive sounded units
            pieces.append(_silence(WORD_GAP_MS))
        clip = cycler.pick(kind, unit, index)
        data = clip_pcm(clip)
        if len(data):
            pieces.append(data)
        else:
            pieces.append(_silence(WORD_GAP_MS))  # silent clip: keep the gap shape
    if not pieces:
        pieces = [_silence(WORD_GAP_MS)]

    return _encode_wav(np.concatenate(pieces))


def _encode_wav(pcm: np.ndarray) -> bytes:
    """Peak-normalize to -3 dBFS and emit a 16 kHz mono 16-bit wav."""
    pcm = pcm.astype(np.float64)
    peak = float(np.max(np.abs(pcm))) if len(pcm) else 0.0
    if peak > 0.0:
        target = 10.0 ** (PEAK_TARGET_DB / 20.0) * 32767.0
        pcm = np.clip(pcm * (target / peak), -32768.0, 32767.0)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm.astype(np.int16).tobytes())
    return buf.getvalue()


class UnknownTokenError(KeyError):
    """Raised by render_clip for tokens outside ACTION_BUCKETS."""


def normalize_action_token(token: str) -> str:
    """Canonical action token: case-insensitive, optional <> ("Bite" -> "<bite>")."""
    cleaned = token.strip().lower()
    if cleaned.startswith("<") and cleaned.endswith(">"):
        cleaned = cleaned[1:-1].strip()
    return f"<{cleaned}>"


def render_clip(token: str, variety: str = "", directory: Path | None = None) -> bytes:
    """Render ONE clip for an action token, as wav bytes.

    Bucket = ACTION_BUCKETS[token][0]; the clip is chosen deterministically
    from that bucket with seed sha256(token + "|" + variety). Single clip —
    no gaps, peak-normalized to -3 dBFS like render.

    Raises UnknownTokenError for tokens outside ACTION_BUCKETS and
    FileNotFoundError when the clip bank has not been prepared.
    """
    token_key = normalize_action_token(token)
    if token_key not in ACTION_BUCKETS:
        raise UnknownTokenError(token_key)
    bucket = ACTION_BUCKETS[token_key][0]

    index = load_clips(directory)  # FileNotFoundError -> caller handles (503)
    clips = index.get(bucket)
    if not clips:  # bucket absent from the bank; same fallback chain as render
        for fallback in ("meow", *sorted(index)):
            if index.get(fallback):
                clips = index[fallback]
                break
    if not clips:
        raise RuntimeError("clip bank has no usable clips")

    seed = int.from_bytes(
        sha256(f"{token_key}|{variety}".encode("utf-8")).digest()[:8], "big"
    )
    clip = clips[random.Random(seed).randrange(len(clips))]
    return _encode_wav(load_wav_mono16k(clip["path"]))
