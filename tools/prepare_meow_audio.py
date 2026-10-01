"""Prepare the meow-audio clip bank from a directory of raw cat wav clips.

Run by the operator on the real corpus snapshot (liladhii/isolated-cat-meows
on Hugging Face, ~4,750 unlabeled clips, 16-44 kHz mono/stereo mixed):

    /opt/homebrew/bin/uv run python tools/prepare_meow_audio.py \
        --src /path/to/wav_snapshot --out assets/audio

Per clip it computes duration, RMS energy and spectral centroid (numpy +
stdlib only; every clip is resampled to mono 16 kHz 16-bit on read), buckets
clips by corpus percentiles, writes ``clips.json``, and copies the
longest-onset-clean clips per bucket into ``clips/<bucket>/`` (16 kHz mono
16-bit, per-clip cap keeps the repo small). Deterministic ordering
throughout.
"""

import argparse
import json
import sys
import wave
from pathlib import Path

import numpy as np

# Make `tools` imports work when run as a script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from meow_lite.meow_tts import load_wav_mono16k  # noqa: E402

SAMPLE_RATE = 16_000
PERCENTILE_LO = 15.0
PERCENTILE_HI = 85.0
PURR_MIN_DURATION = 0.8  # seconds; low-centroid clips must also be long
CLIPS_PER_BUCKET = 15
ONSET_CLEAN_MAX_S = 0.25  # leading silence tolerated for "onset clean"
MAX_CLIP_SECONDS = 4.0  # per-clip copy cap: 60 clips * 4s * 32 kB/s < 8 MB
BUCKETS = ("purr", "hiss", "bite", "meow")


def clip_features(x: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
    """Duration (s), normalized RMS and spectral centroid for a mono int16 clip."""
    duration = len(x) / sr if sr else 0.0
    if len(x) == 0:
        return {"duration": 0.0, "rms": 0.0, "centroid": 0.0}
    xf = x.astype(np.float64)
    rms = float(np.sqrt(np.mean(xf**2)) / 32768.0)
    windowed = xf * np.hanning(len(xf))
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(len(xf), 1.0 / sr)
    total = float(spectrum.sum())
    centroid = float((spectrum * freqs).sum() / total) if total > 0.0 else 0.0
    return {"duration": duration, "rms": rms, "centroid": centroid}


def leading_silence_seconds(x: np.ndarray, sr: int = SAMPLE_RATE) -> float:
    """Seconds of near-silence before the first audible sample."""
    if len(x) == 0:
        return 0.0
    peak = float(np.max(np.abs(x.astype(np.float64))))
    if peak == 0.0:
        return len(x) / sr
    above = np.nonzero(np.abs(x.astype(np.float64)) > 0.02 * peak)[0]
    if len(above) == 0:
        return len(x) / sr
    return float(above[0]) / sr


def assign_buckets(entries: list[dict]) -> None:
    """Percentile-based bucketing, in-place. purr -> hiss -> bite -> meow."""
    if not entries:
        return
    centroids = [entry["centroid"] for entry in entries]
    durations = [entry["duration"] for entry in entries]
    centroid_lo = float(np.percentile(centroids, PERCENTILE_LO))
    centroid_hi = float(np.percentile(centroids, PERCENTILE_HI))
    duration_lo = float(np.percentile(durations, PERCENTILE_LO))
    for entry in entries:
        if entry["centroid"] <= centroid_lo and entry["duration"] >= PURR_MIN_DURATION:
            entry["bucket"] = "purr"
        elif entry["centroid"] >= centroid_hi:
            entry["bucket"] = "hiss"
        elif entry["duration"] <= duration_lo:
            entry["bucket"] = "bite"
        else:
            entry["bucket"] = "meow"


def _bank_rank(entry: dict) -> tuple:
    """Longest-onset-clean first: clean onsets win, then longest, then name."""
    clean = 0 if entry["onset_silence"] <= ONSET_CLEAN_MAX_S else 1
    return (clean, -entry["duration"], entry["file"])


def write_clip(path: Path, pcm: np.ndarray, max_seconds: float = MAX_CLIP_SECONDS) -> None:
    """Write mono 16 kHz 16-bit PCM, truncated to max_seconds."""
    limit = int(max_seconds * SAMPLE_RATE)
    data = pcm[:limit]
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(data.tobytes())


def prepare(src: Path, out: Path, clips_per_bucket: int = CLIPS_PER_BUCKET,
            log=None) -> dict:
    """Analyze ``src`` wavs, bucket them, write clips.json + the clip bank."""
    src = Path(src)
    out = Path(out)
    if log is None:
        log = lambda *_args, **_kw: None  # noqa: E731

    files = sorted(p for p in src.iterdir() if p.suffix.lower() == ".wav")
    entries: list[dict] = []
    decoded: dict[str, np.ndarray] = {}
    for path in files:
        try:
            pcm = load_wav_mono16k(path)
        except (wave.Error, EOFError, ValueError) as exc:
            log(f"skipping {path.name}: {exc}")
            continue
        if len(pcm) == 0:
            log(f"skipping {path.name}: empty")
            continue
        entry = {"file": path.name, **clip_features(pcm)}
        entry["onset_silence"] = leading_silence_seconds(pcm)
        if entry["rms"] == 0.0:
            # Fully silent clips carry no feline signal and would poison the
            # percentile bucketing on small corpora.
            log(f"skipping {path.name}: silent")
            continue
        entries.append(entry)
        decoded[path.name] = pcm

    assign_buckets(entries)
    entries.sort(key=lambda entry: entry["file"])  # deterministic ordering

    clips_root = out / "clips"
    banked_count = 0
    for bucket in BUCKETS:
        members = [entry for entry in entries if entry["bucket"] == bucket]
        members.sort(key=_bank_rank)
        chosen = members[:clips_per_bucket]
        bucket_dir = clips_root / bucket
        bucket_dir.mkdir(parents=True, exist_ok=True)
        for entry in chosen:
            rel = f"clips/{bucket}/{entry['file']}"
            write_clip(bucket_dir / entry["file"], decoded[entry["file"]])
            entry["path"] = rel  # visible to meow_tts.load_clips via clips.json
            banked_count += 1

    manifest = {
        "sample_rate": SAMPLE_RATE,
        "source": str(src),
        "counts": {bucket: sum(1 for e in entries if e["bucket"] == bucket) for bucket in BUCKETS},
        "clips": entries,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "clips.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    log(
        f"analyzed {len(entries)}/{len(files)} clips; banked {banked_count} "
        + ", ".join(
            f"{bucket}={sum(1 for e in entries if e['bucket'] == bucket and 'path' in e)}"
            for bucket in BUCKETS
        )
    )
    return {"analyzed": len(entries), "banked": banked_count, "manifest": manifest}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, help="directory of raw wav clips")
    parser.add_argument("--out", default="assets/audio", help="output dir (default assets/audio)")
    parser.add_argument(
        "--clips-per-bucket", type=int, default=CLIPS_PER_BUCKET,
        help=f"clips banked per bucket (default {CLIPS_PER_BUCKET})",
    )
    args = parser.parse_args(argv)

    def log(message: str) -> None:
        print(message, file=sys.stderr)

    result = prepare(Path(args.src), Path(args.out), clips_per_bucket=args.clips_per_bucket, log=log)
    print(
        f"wrote {args.out}/clips.json and banked {result['banked']} clips "
        f"from {result['analyzed']} analyzed clips"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
