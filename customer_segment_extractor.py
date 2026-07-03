#!/usr/bin/env python
"""
customer_segment_extractor.py

Production‑grade scaffold for extracting **only the customer's** speech from a
full call recording.  The platform already knows the exact AI‑voice timestamps,
so future versions will use that data to slice out the customer segments.
For now the module works in *placeholder mode* – it simply copies the original
audio file.

Typical usage
-------------
    python customer_segment_extractor.py call.wav
    python customer_segment_extractor.py call.wav timestamps.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf


# --------------------------------------------------------------------------- #
# Custom Exceptions
# --------------------------------------------------------------------------- #

class AudioLoadError(Exception):
    """Raised when an audio file cannot be read."""
    pass


class TimestampLoadError(Exception):
    """Raised when the timestamps JSON cannot be parsed."""
    pass


class AudioSaveError(Exception):
    """Raised when an audio file cannot be written."""
    pass


# --------------------------------------------------------------------------- #
# Core pipeline functions
# --------------------------------------------------------------------------- #

def load_audio(audio_path: str | Path) -> Tuple[np.ndarray, int]:
    """
    Load an audio file.

    Parameters
    ----------
    audio_path : str | Path
        Path to the input WAV file.

    Returns
    -------
    Tuple[np.ndarray, int]
        ``waveform`` – audio samples as a NumPy array.
        ``sample_rate`` – integer sample rate in Hz.

    Raises
    ------
    FileNotFoundError
        If ``audio_path`` does not exist.
    AudioLoadError
        If the file exists but cannot be read (unsupported format, corrupt, …).
    """
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    try:
        waveform, sr = sf.read(str(path))
        waveform = np.asarray(waveform)          # guarantee NumPy array
        return waveform, int(sr)
    except Exception as exc:
        raise AudioLoadError(f"Unable to load {path}: {exc}") from exc


def load_timestamps(json_path: str | Path) -> Optional[Dict[str, Any]]:
    """
    Load a JSON file containing speaker‑wise timestamps.

    Expected future format (example):
    {
        "segments": [
            {"speaker": "AI",       "start": 0.0,  "end": 4.5},
            {"speaker": "CUSTOMER","start": 4.6,  "end": 18.1}
        ]
    }

    Current behaviour:
    * If the file is missing, print a notice and return ``None`` – the caller
      will treat this as *placeholder mode* and simply copy the original audio.
    * If the file exists but cannot be parsed, raise ``TimestampLoadError``.
    """
    path = Path(json_path)
    if not path.is_file():
        print(
            "No timestamp file supplied.\nUsing placeholder mode.",
            file=sys.stderr,
        )
        return None

    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data
    except json.JSONDecodeError as exc:
        raise TimestampLoadError(f"Invalid JSON in {path}: {exc}") from exc
    except Exception as exc:
        raise TimestampLoadError(f"Failed to read {path}: {exc}") from exc


def extract_customer_audio(
    waveform: np.ndarray,
    sample_rate: int,
    timestamps: Optional[Dict[str, Any]],
) -> np.ndarray:
    """
    Return a waveform that contains **only** the customer's speech.

    Future implementation plan
    -------------------------
    1. Iterate over ``timestamps["segments"]``.
    2. For each segment where ``speaker == "CUSTOMER"``, convert the
       ``start`` and ``end`` timestamps (seconds) to sample indices:
       ``start_idx = int(start * sample_rate)``,
       ``end_idx   = int(end   * sample_rate)``.
    3. Slice ``waveform[start_idx:end_idx]`` and append it to a list.
    4. After the loop, ``np.concatenate`` the list into a single array and
       return it.

    For the current placeholder version we simply return the original
    ``waveform`` unchanged, regardless of ``timestamps``.
    """
    # Placeholder – no actual segmentation yet.
    return waveform


def save_audio(waveform: np.ndarray, sample_rate: int) -> Path:
    """
    Write the waveform to ``temp/customer_voice.wav``.

    The ``temp`` directory is created automatically if it does not exist.

    Returns
    -------
    Path
        Path of the saved file.

    Raises
    ------
    AudioSaveError
        If the file cannot be written.
    """
    temp_dir = Path("temp")
    try:
        temp_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        raise AudioSaveError(f"Could not create temp directory: {exc}") from exc

    out_path = temp_dir / "customer_voice.wav"
    try:
        sf.write(str(out_path), waveform, sample_rate)
        return out_path
    except Exception as exc:
        raise AudioSaveError(f"Could not write {out_path}: {exc}") from exc


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #

def _parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract customer speech from a full call recording."
    )
    parser.add_argument(
        "audio_path",
        type=Path,
        help="Path to the full call recording (WAV).",
    )
    parser.add_argument(
        "timestamps_path",
        type=Path,
        nargs="?",
        default=None,
        help="Optional JSON file with AI/customer timestamps.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_cli()

    print("-" * 32)
    print("Loading recording...")
    print()

    # ------------------------------------------------------------------- #
    # Load audio
    # ------------------------------------------------------------------- #
    try:
        waveform, sr = load_audio(args.audio_path)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except AudioLoadError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    duration_sec = waveform.shape[0] / sr
    print(f"Sample Rate : {sr} Hz")
    print(f"Duration    : {duration_sec:.2f} sec")
    print(f"Timestamp file : {args.timestamps_path or 'None'}")

    # ------------------------------------------------------------------- #
    # Load timestamps (optional)
    # ------------------------------------------------------------------- #
    timestamps: Optional[Dict[str, Any]] = None
    placeholder_mode = False

    if args.timestamps_path is not None:
        try:
            timestamps = load_timestamps(args.timestamps_path)
            placeholder_mode = timestamps is None
        except TimestampLoadError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
    else:
        placeholder_mode = True
        print(
            "No timestamp file supplied.\nUsing placeholder mode.",
            file=sys.stderr,
        )

    print(f"Placeholder Mode : {placeholder_mode}")

    # ------------------------------------------------------------------- #
    # Extract (placeholder) customer audio
    # ------------------------------------------------------------------- #
    customer_waveform = extract_customer_audio(waveform, sr, timestamps)

    # ------------------------------------------------------------------- #
    # Save result
    # ------------------------------------------------------------------- #
    try:
        out_path = save_audio(customer_waveform, sr)
    except AudioSaveError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Customer Audio Saved : {out_path.as_posix()}")
    print("-" * 32)


if __name__ == "__main__":
    main()
