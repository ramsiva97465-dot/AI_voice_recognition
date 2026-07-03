#!/usr/bin/env python
"""
vad_processor.py

Voice‑Activity‑Detection (VAD) utility for the AI Voice Calling Platform.
It removes silence from a customer's audio recording so that downstream
embedding generation works on speech‑only signal.

Typical usage
-------------
    python vad_processor.py temp/customer_voice.wav
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import librosa
import numpy as np
import soundfile as sf


# --------------------------------------------------------------------------- #
# Custom Exceptions
# --------------------------------------------------------------------------- #

class AudioLoadError(Exception):
    """Raised when an audio file cannot be read."""
    pass


class AudioSaveError(Exception):
    """Raised when an audio file cannot be written."""
    pass


class SilenceRemovalError(Exception):
    """Raised when VAD fails to produce any speech frames."""
    pass


# --------------------------------------------------------------------------- #
# Core processing functions
# --------------------------------------------------------------------------- #

def load_audio(audio_path: str | Path) -> Tuple[np.ndarray, int]:
    """
    Load an audio file and ensure a mono waveform.

    Parameters
    ----------
    audio_path : str | Path
        Path to the input WAV file.

    Returns
    -------
    Tuple[np.ndarray, int]
        ``waveform`` – 1‑D NumPy array (mono audio).
        ``sample_rate`` – sampling rate in Hz.

    Raises
    ------
    FileNotFoundError
        If ``audio_path`` does not exist.
    AudioLoadError
        If the file exists but cannot be decoded (unsupported format,
        corrupted data, …).
    """
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    try:
        waveform, sr = sf.read(str(path))
        waveform = np.asarray(waveform)

        # Convert to mono if needed
        if waveform.ndim > 1:
            # Average across channels (shape: (samples, channels))
            waveform = waveform.mean(axis=1)

        return waveform, int(sr)
    except Exception as exc:
        raise AudioLoadError(f"Failed to load {path}: {exc}") from exc


def remove_silence(
    waveform: np.ndarray,
    sample_rate: int,
    top_db: int = 25,
) -> np.ndarray:
    """
    Remove silence using librosa's energy‑based VAD.

    Parameters
    ----------
    waveform : np.ndarray
        Mono audio signal.
    sample_rate : int
        Sample rate of ``waveform``.
    top_db : int, optional
        Threshold (in dB) below reference to consider as silence.
        Default is 25 dB, which works well for phone‑call recordings.

    Returns
    -------
    np.ndarray
        Speech‑only waveform obtained by concatenating all non‑silent
        intervals.

    Raises
    ------
    SilenceRemovalError
        If no speech frames are detected (empty result).
    """
    # librosa expects a floating‑point array; ensure correct dtype
    if waveform.dtype != np.float32 and waveform.dtype != np.float64:
        waveform = waveform.astype(np.float32)

    intervals = librosa.effects.split(waveform, top_db=top_db)

    if intervals.size == 0:
        raise SilenceRemovalError("VAD detected no speech in the signal.")

    # Concatenate the voiced frames
    speech_parts = [waveform[start:end] for start, end in intervals]
    speech = np.concatenate(speech_parts, axis=0)

    return speech


def save_audio(waveform: np.ndarray, sample_rate: int) -> Path:
    """
    Write the processed waveform to ``temp/customer_speech.wav``.

    The ``temp`` directory is created automatically if it does not exist.

    Returns
    -------
    Path
        Path to the saved file.

    Raises
    ------
    AudioSaveError
        If the file cannot be written.
    """
    temp_dir = Path("temp")
    try:
        temp_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        raise AudioSaveError(f"Failed to create temp directory: {exc}") from exc

    out_path = temp_dir / "customer_speech.wav"
    try:
        sf.write(str(out_path), waveform, sample_rate)
        return out_path
    except Exception as exc:
        raise AudioSaveError(f"Failed to write {out_path}: {exc}") from exc


def print_statistics(
    original_duration: float,
    speech_duration: float,
) -> None:
    """
    Print a concise report about the VAD processing.

    Parameters
    ----------
    original_duration : float
        Length of the original audio (seconds).
    speech_duration : float
        Length after silence removal (seconds).
    """
    silence_removed = original_duration - speech_duration
    compression_ratio = (speech_duration / original_duration) * 100 if original_duration > 0 else 0.0

    print("-" * 32)
    print(f"Original Duration : {original_duration:.2f} sec")
    print(f"Speech Duration   : {speech_duration:.2f} sec")
    print(f"Silence Removed   : {silence_removed:.2f} sec")
    print(f"Compression       : {compression_ratio:.2f}%")
    print("-" * 32)


# --------------------------------------------------------------------------- #
# CLI handling
# --------------------------------------------------------------------------- #

def _parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove silence from a customer's recording using VAD."
    )
    parser.add_argument(
        "audio_path",
        type=Path,
        help="Path to the input wav file (e.g., temp/customer_voice.wav).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_cli()
    audio_path = args.audio_path

    print("-" * 32)
    print("Loading customer audio...")
    print()

    # ------------------------------------------------------------------- #
    # Load audio
    # ------------------------------------------------------------------- #
    try:
        waveform, sr = load_audio(audio_path)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except AudioLoadError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    original_duration = waveform.shape[0] / sr
    print(f"Sample Rate : {sr} Hz")
    print(f"Original Duration : {original_duration:.2f} sec")
    print()

    # ------------------------------------------------------------------- #
    # Remove silence
    # ------------------------------------------------------------------- #
    print("Removing silence...")
    try:
        speech_waveform = remove_silence(waveform, sr, top_db=25)
    except SilenceRemovalError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"Unexpected error during VAD: {exc}", file=sys.stderr)
        sys.exit(1)

    speech_duration = speech_waveform.shape[0] / sr
    print_statistics(original_duration, speech_duration)

    # ------------------------------------------------------------------- #
    # Save processed speech
    # ------------------------------------------------------------------- #
    print("Saving processed speech...")
    try:
        out_path = save_audio(speech_waveform, sr)
    except AudioSaveError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Saved : {out_path.as_posix()}")
    print("-" * 32)


if __name__ == "__main__":
    main()
