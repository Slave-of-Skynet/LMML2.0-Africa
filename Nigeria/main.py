"""Render the spectral message hidden in the LMML Nigeria WAV challenge.

Usage: python main.py [path/to/challenge.wav] [--word UPPERCASEWORD]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import wave

import numpy as np
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "wav", type=Path, nargs="?", default=BASE_DIR / "challenge.wav",
        help="Input mono 16-bit PCM WAV (default: challenge.wav beside main.py)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=BASE_DIR / "outputs",
        help="Output folder (default: outputs/ beside main.py)",
    )
    parser.add_argument(
        "--word", help="Optional uppercase word read visually from message.png",
    )
    args = parser.parse_args()
    if args.word is not None and re.fullmatch(r"[A-Z]+", args.word) is None:
        parser.error("--word must contain only uppercase ASCII letters")

    try:
        with wave.open(str(args.wav), "rb") as source:
            params = source.getparams()
            if (params.nchannels, params.sampwidth, params.comptype) != (1, 2, "NONE"):
                parser.error("Expected mono, uncompressed 16-bit PCM WAV")
            samples = np.frombuffer(source.readframes(params.nframes), dtype="<i2").astype(float)
    except (FileNotFoundError, IsADirectoryError) as exc:
        parser.error(f"Input WAV not found: {exc}")
    except (wave.Error, EOFError) as exc:
        parser.error(f"Invalid WAV: {exc}")

    nfft, hop = 2048, 64
    if samples.size < nfft:
        parser.error("Audio is too short for the analysis window")
    if params.framerate / 2 < 4000:
        parser.error("Sample rate is too low for the 2500-4000 Hz message crop")

    frames = np.lib.stride_tricks.sliding_window_view(samples, nfft)[::hop]
    spectrum = np.abs(np.fft.rfft(frames * np.hanning(nfft), axis=1)).T
    level = 20 * np.log10(spectrum + 1)
    pixels = np.clip((level - 72) / 32 * 255, 0, 255).astype(np.uint8)
    low, high = (int(freq * nfft / params.framerate) for freq in (2500, 4000))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels[::-1]).resize((1750, 900)).save(args.output_dir / "spectrogram.png")
    Image.fromarray(pixels[low:high][::-1]).resize((1750, 500)).save(args.output_dir / "message.png")
    report = {
        "input_sha256": hashlib.sha256(args.wav.read_bytes()).hexdigest(),
        "input_bytes": args.wav.stat().st_size,
        "channels": params.nchannels,
        "sample_width_bytes": params.sampwidth,
        "sample_rate_hz": params.framerate,
        "frames": params.nframes,
        "duration_seconds": params.nframes / params.framerate,
        "nfft": nfft,
        "hop_samples": hop,
        "window": "numpy.hanning",
        "crop_hz": [2500, 4000],
        "numpy_version": np.__version__,
    }
    (args.output_dir / "analysis.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Spectrogram and report: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
