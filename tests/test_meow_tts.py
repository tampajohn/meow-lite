"""Text-to-meow audio tests: clip preparation, rendering, endpoint."""

import io
import json
import wave
from pathlib import Path

import numpy as np
import pytest

from meow_lite import meow_tts
from meow_lite.server import app
from tools import prepare_meow_audio as prep

SAMPLE_RATE = 16_000


# --------------------------------------------------------------------------
# Tiny synthetic clip fixture: 6 wavs written by this helper, nothing large
# is committed (each file is a fraction of a second of PCM).
# --------------------------------------------------------------------------

def _write_pcm(path: Path, samples: np.ndarray) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(samples.astype(np.int16).tobytes())


def _sine(freq: float, seconds: float, amp: int = 20_000) -> np.ndarray:
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.int16)


def _noise(seconds: float, amp: int = 20_000, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.uniform(-amp, amp, int(seconds * SAMPLE_RATE))).astype(np.int16)


def _click(seconds: float = 0.03, freq: float = 300.0, amp: int = 25_000) -> np.ndarray:
    """Damped low-frequency burst: short duration, low spectral centroid."""
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    envelope = np.exp(-t * 40.0)
    return (amp * envelope * np.sin(2 * np.pi * freq * t)).astype(np.int16)


def write_fixture_clips(src: Path) -> Path:
    """Six synthetic clips: two meows, a hiss, a purr, a bite click, silence.

    Centroids are deliberately well-separated (80 Hz purr < 300 Hz click <
    400/700 Hz meows < broadband hiss) so the 15th/85th-percentile buckets are
    unambiguous even with only five non-silent clips.
    """
    src.mkdir(parents=True, exist_ok=True)
    _write_pcm(src / "meow_a.wav", _sine(400.0, 0.5))
    _write_pcm(src / "meow_b.wav", _sine(700.0, 0.4))
    _write_pcm(src / "hiss.wav", _noise(0.5))
    _write_pcm(src / "purr.wav", _sine(80.0, 0.85, amp=15_000))
    _write_pcm(src / "bite.wav", _click())
    _write_pcm(src / "silent.wav", np.zeros(int(0.3 * SAMPLE_RATE), dtype=np.int16))
    return src


@pytest.fixture()
def bank(tmp_path, monkeypatch):
    """Prepare a clip bank from the fixture clips and point MEOW_AUDIO_DIR at it."""
    src = write_fixture_clips(tmp_path / "raw")
    out = tmp_path / "assets" / "audio"
    prep.prepare(src, out)
    monkeypatch.setenv("MEOW_AUDIO_DIR", str(out))
    return out


# --------------------------------------------------------------------------
# prepare_meow_audio
# --------------------------------------------------------------------------

def test_prepare_buckets_and_bank(tmp_path):
    src = write_fixture_clips(tmp_path / "raw")
    out = tmp_path / "assets" / "audio"
    prep.prepare(src, out)

    manifest = json.loads((out / "clips.json").read_text())
    buckets = {entry["file"]: entry["bucket"] for entry in manifest["clips"]}
    # 80 Hz rumble: lowest centroid and >= 0.8s -> purr; noise: top centroid
    # -> hiss; 30 ms click: shortest -> bite; the rest -> meow.
    assert buckets["purr.wav"] == "purr"
    assert buckets["hiss.wav"] == "hiss"
    assert buckets["bite.wav"] == "bite"
    assert buckets["meow_a.wav"] == "meow"
    assert buckets["meow_b.wav"] == "meow"

    banked = [entry for entry in manifest["clips"] if "path" in entry]
    assert banked, "expected at least one banked clip"
    for entry in banked:
        path = out / entry["path"]
        assert path.exists()
        with wave.open(str(path), "rb") as wav:
            assert wav.getnchannels() == 1
            assert wav.getsampwidth() == 2
            assert wav.getframerate() == SAMPLE_RATE


def test_prepare_is_deterministic(tmp_path):
    src = write_fixture_clips(tmp_path / "raw")
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    prep.prepare(src, out_a)
    prep.prepare(src, out_b)
    assert (out_a / "clips.json").read_bytes() == (out_b / "clips.json").read_bytes()
    banked = [e["path"] for e in json.loads((out_a / "clips.json").read_text())["clips"] if "path" in e]
    for rel in banked:
        assert (out_a / rel).read_bytes() == (out_b / rel).read_bytes()


# --------------------------------------------------------------------------
# meow_tts
# --------------------------------------------------------------------------

def test_tokenize_units_action_tokens_single_including_mixed_case():
    assert meow_tts.tokenize_units("<bite> meow.") == [
        ("action", "<bite>"),
        ("word", "meow"),
        ("punct", "."),
    ]
    # Mixed case stays ONE unit, never split into <, letters, >.
    assert meow_tts.tokenize_units("<SCRATCH_COUCH> Mrrp!") == [
        ("action", "<scratch_couch>"),
        ("word", "mrrp"),
        ("punct", "!"),
    ]
    assert meow_tts.tokenize_units("<Purr><purr>") == [
        ("action", "<purr>"),
        ("action", "<purr>"),
    ]


def test_render_is_seeded_deterministic(bank):
    a = meow_tts.render("<bite> Meow. Mrrp!", directory=bank)
    b = meow_tts.render("<bite> Meow. Mrrp!", directory=bank)
    assert a == b


def test_render_wav_is_valid_mono_16k_16bit(bank):
    data = meow_tts.render("Meow meow", directory=bank)
    with wave.open(io.BytesIO(data), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == SAMPLE_RATE
        duration = wav.getnframes() / wav.getframerate()
    # Two sounded units + one 40 ms word gap; fixture clips are 0.4-0.85 s.
    assert 0.3 < duration < 3.0
    # Exact-ish: meow bucket clips (the two fixture sines) sum + 40 ms gap.
    index = meow_tts.load_clips(bank)
    assert len(index["meow"]) == 2  # 400 Hz + 700 Hz fixture clips


def test_render_duration_tracks_units(bank):
    with wave.open(io.BytesIO(meow_tts.render("Meow", directory=bank)), "rb") as wav:
        one = wav.getnframes() / wav.getframerate()
    with wave.open(io.BytesIO(meow_tts.render("Meow meow", directory=bank)), "rb") as wav:
        two = wav.getnframes() / wav.getframerate()
    # Second unit adds its clip plus the 40 ms gap.
    assert two - one > 0.04


def test_punctuation_adds_silence(bank):
    # Different texts seed different clip picks, so each render is checked
    # against the bank's own clip durations rather than against each other.
    clip_durs = [clip["duration"] for clip in meow_tts.load_clips(bank)["meow"]]

    def duration_of(text: str) -> float:
        with wave.open(io.BytesIO(meow_tts.render(text, directory=bank)), "rb") as wav:
            return wav.getnframes() / SAMPLE_RATE

    d_plain = duration_of("Meow")
    d_punct = duration_of("Meow.")
    assert d_punct > d_plain
    # Plain: exactly one clip. Punctuated: one clip + 120 ms of silence.
    assert any(abs(d_plain - dur) < 0.005 for dur in clip_durs)
    punct = meow_tts.PUNCT_SILENCE_MS / 1000.0
    assert any(abs(d_punct - (dur + punct)) < 0.005 for dur in clip_durs)


def test_render_rejects_unknown_format(bank):
    with pytest.raises(ValueError):
        meow_tts.render("Meow", directory=bank, out_format="mp3")


def test_load_clips_missing_manifest(tmp_path):
    with pytest.raises(FileNotFoundError):
        meow_tts.load_clips(tmp_path)


# --------------------------------------------------------------------------
# Server endpoint
# --------------------------------------------------------------------------

@pytest.fixture()
def client():
    from fastapi.testclient import TestClient

    return TestClient(app)


def _audio_request(client):
    return client.post(
        "/v1/meow/audio",
        json={"model": "meow-lite", "messages": [{"role": "user", "content": "pet the cat"}]},
    )


def test_endpoint_returns_wav_with_text_header(client, bank):
    response = _audio_request(client)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/wav")
    text = response.headers["x-meow-text"]
    assert text.strip()
    with wave.open(io.BytesIO(response.content), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == SAMPLE_RATE
        assert wav.getnframes() > 0


def test_endpoint_503_without_clip_bank(client, tmp_path, monkeypatch):
    empty = tmp_path / "no-bank"
    empty.mkdir()
    monkeypatch.setenv("MEOW_AUDIO_DIR", str(empty))
    response = _audio_request(client)
    assert response.status_code == 503
    assert "clips.json" in response.json()["error"]["message"]


def test_endpoint_503_does_not_break_other_endpoints(client, tmp_path, monkeypatch):
    empty = tmp_path / "no-bank"
    empty.mkdir()
    monkeypatch.setenv("MEOW_AUDIO_DIR", str(empty))
    assert _audio_request(client).status_code == 503
    assert client.get("/health").status_code == 200
    chat = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hello"}]},
    )
    assert chat.status_code == 200
    assert chat.json()["choices"][0]["message"]["content"]
