#!/usr/bin/env python
"""
extract_customer_voice.py

A production-grade pipeline foundation for extracting the customer portion
of a recorded phone call. The current implementation simply copies the
original audio file. Future phases will introduce speaker diarization,
voice activity detection, and merging.

Usage:
    python extract_customer_voice.py audio/calls/customer_call.wav
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf


# --------------------------------------------------------------------------- #
# Custom Exceptions
# --------------------------------------------------------------------------- #

class AudioLoadError(Exception):
    """Exception raised when loading an audio file fails."""
    pass


class AudioSaveError(Exception):
    """Exception raised when saving an audio file fails."""
    pass


# --------------------------------------------------------------------------- #
# Core Pipeline Functions
# --------------------------------------------------------------------------- #

def load_audio(audio_path: str | Path) -> Tuple[np.ndarray, int]:
    """
    Loads an audio file from the specified path.

    Parameters
    ----------
    audio_path : str | Path
        The path to the audio file.

    Returns
    -------
    Tuple[np.ndarray, int]
        A tuple containing the waveform as a NumPy array and the sample rate in Hz.

    Raises
    ------
    FileNotFoundError
        If the audio file does not exist.
    AudioLoadError
        If the file exists but cannot be loaded (e.g. corrupt or unsupported).
    """
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found at: {path}")

    try:
        waveform, sample_rate = sf.read(str(path))
        # Ensure it is a numpy array
        waveform = np.asarray(waveform)
        return waveform, int(sample_rate)
    except Exception as exc:
        raise AudioLoadError(f"Failed to load audio from {path}: {exc}") from exc


def analyze_audio(waveform: np.ndarray, sample_rate: int) -> None:
    """
    Analyzes the audio waveform and prints its basic characteristics.

    Parameters
    ----------
    waveform : np.ndarray
        The audio waveform array.
    sample_rate : int
        The sample rate of the audio in Hz.
    """
    channels = 1 if waveform.ndim == 1 else waveform.shape[1]
    total_samples = len(waveform)
    duration_sec = total_samples / sample_rate

    print(f"Sample Rate : {sample_rate} Hz")
    print(f"Duration : {duration_sec:.2f} sec")
    print(f"Channels : {channels}")
    print(f"Total Samples : {total_samples}")


def extract_customer_segments(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Extracts the customer's speech segments from the call recording.

    Future implementation details:
    ------------------------------
    1. AI speaking timestamps will come from the AI platform.
    2. Customer speech begins after AI playback ends.
    3. Customer speech segments will be extracted using these timestamps.
    4. Later these segments will be merged into one audio stream.

    For now, simply return the original waveform unchanged.
    """
    # Placeholder: currently returns the original waveform unchanged.
    return waveform


def save_customer_audio(waveform: np.ndarray, sample_rate: int) -> Path:
    """
    Saves the customer waveform to temp/customer_voice.wav.
    Creates the temp folder automatically if it does not exist.

    Parameters
    ----------
    waveform : np.ndarray
        The audio waveform to save.
    sample_rate : int
        The sample rate of the audio.

    Returns
    -------
    Path
        The path to the saved customer audio file.

    Raises
    ------
    AudioSaveError
        If saving the audio file fails.
    """
    temp_dir = Path("temp")
    try:
        temp_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        raise AudioSaveError(f"Failed to create temp directory: {exc}") from exc

    save_path = temp_dir / "customer_voice.wav"
    try:
        sf.write(str(save_path), waveform, sample_rate)
        return save_path
    except Exception as exc:
        raise AudioSaveError(f"Failed to write audio file to {save_path}: {exc}") from exc


# --------------------------------------------------------------------------- #
# CLI implementation
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract the customer voice segment from a call recording."
    )
    parser.add_argument(
        "audio_path",
        type=Path,
        help="Path to the call recording audio file."
    )
    args = parser.parse_args()

    print("-" * 32)
    print("Loading Call Recording...")
    print()

    try:
        waveform, sample_rate = load_audio(args.audio_path)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except AudioLoadError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    analyze_audio(waveform, sample_rate)
    print()

    customer_waveform = extract_customer_segments(waveform, sample_rate)

    try:
        saved_path = save_customer_audio(customer_waveform, sample_rate)
    except AudioSaveError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print("Customer extraction pipeline complete.")
    print()
    print("Saved:")
    print()
    print(saved_path.as_posix())
    print()
    print("NOTE:")
    print()
    print("Current version copies the original audio.")
    print()
    print("Speaker separation will be implemented in a future phase.")
    print()
    print("-" * 32)


if __name__ == "__main__":
    main()
