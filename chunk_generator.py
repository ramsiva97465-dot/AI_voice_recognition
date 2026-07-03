#!/usr/bin/env python
"""
chunk_generator.py

Utility to split a pre‑processed customer speech recording into uniform
chunks suitable for downstream embedding generation.

Typical usage
-------------
    python chunk_generator.py temp/customer_speech.wav
    # Optional overrides
    python chunk_generator.py temp/customer_speech.wav --min 12 --max 28 --target 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf


# --------------------------------------------------------------------------- #
# Custom Exceptions
# --------------------------------------------------------------------------- #

class AudioLoadError(Exception):
    """Raised when an audio file cannot be read."""
    pass


class ChunkSaveError(Exception):
    """Raised when a chunk cannot be written to disk."""
    pass


# --------------------------------------------------------------------------- #
# Core processing functions
# --------------------------------------------------------------------------- #

def load_audio(audio_path: str | Path) -> Tuple[np.ndarray, int]:
    """
    Load a WAV file and return a mono waveform.

    Parameters
    ----------
    audio_path : str | Path
        Path to the input audio file.

    Returns
    -------
    Tuple[np.ndarray, int]
        (waveform, sample_rate) where ``waveform`` is a 1‑D NumPy array.

    Raises
    ------
    FileNotFoundError
        If ``audio_path`` does not exist.
    AudioLoadError
        If the file cannot be decoded.
    """
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    try:
        waveform, sr = sf.read(str(path))
        waveform = np.asarray(waveform)

        # Convert to mono if needed
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)

        return waveform, int(sr)
    except Exception as exc:
        raise AudioLoadError(f"Failed to load {path}: {exc}") from exc


def split_into_chunks(
    waveform: np.ndarray,
    sample_rate: int,
    min_len: float,
    max_len: float,
    target_len: float,
) -> List[Tuple[int, int]]:
    """
    Split a waveform into time‑based chunks respecting length constraints.

    Parameters
    ----------
    waveform : np.ndarray
        Mono audio data.
    sample_rate : int
        Sample rate of the audio.
    min_len : float
        Minimum chunk duration (seconds).
    max_len : float
        Maximum chunk duration (seconds).
    target_len : float
        Preferred chunk length (seconds). The algorithm will aim for this
        length but will never violate the min/max bounds.

    Returns
    -------
    List[Tuple[int, int]]
        List of (start_idx, end_idx) sample indices for each chunk.
    """
    total_samples = waveform.shape[0]
    total_seconds = total_samples / sample_rate

    # Convert limits to samples once for efficiency
    min_samples = int(min_len * sample_rate)
    max_samples = int(max_len * sample_rate)
    target_samples = int(target_len * sample_rate)

    chunks: List[Tuple[int, int]] = []
    start = 0

    while start < total_samples:
        remaining = total_samples - start

        # Decide chunk length in samples
        if remaining <= max_samples and remaining >= min_samples:
            # The rest fits comfortably within limits.
            chunk_samples = remaining
        elif remaining > max_samples:
            # Still more than the maximum – honour the max bound.
            chunk_samples = max_samples
        else:  # remaining < min_samples
            # Too short to be a standalone chunk → merge with previous.
            if not chunks:
                # Edge case: the whole file is shorter than min_len.
                chunk_samples = remaining
            else:
                # Extend the previous chunk to include the remainder.
                prev_start, _ = chunks[-1]
                chunks[-1] = (prev_start, total_samples)
                break

        end = start + chunk_samples
        chunks.append((start, end))
        start = end

    return chunks


def save_chunks(
    waveform: np.ndarray,
    sample_rate: int,
    chunks: List[Tuple[int, int]],
    out_dir: Path,
) -> List[Path]:
    """
    Write each chunk as a separate WAV file.

    Parameters
    ----------
    waveform : np.ndarray
        Full audio waveform.
    sample_rate : int
        Sample rate.
    chunks : List[Tuple[int, int]]
        List of (start, end) indices.
    out_dir : Path
        Destination directory (will be created if missing).

    Returns
    -------
    List[Path]
        Paths of the saved chunk files.

    Raises
    ------
    ChunkSaveError
        If writing any chunk fails.
    """
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        raise ChunkSaveError(f"Failed to create output folder {out_dir}: {exc}") from exc

    saved_paths: List[Path] = []

    for idx, (start, end) in enumerate(chunks, start=1):
        chunk_waveform = waveform[start:end]
        filename = f"chunk_{idx:03d}.wav"
        out_path = out_dir / filename

        try:
            sf.write(str(out_path), chunk_waveform, sample_rate)
            saved_paths.append(out_path)
        except Exception as exc:
            raise ChunkSaveError(f"Failed to write chunk {filename}: {exc}") from exc

    return saved_paths


def print_statistics(
    original_duration: float,
    chunk_durations: List[float],
    out_dir: Path,
) -> None:
    """
    Print a concise summary of the chunking operation.

    Parameters
    ----------
    original_duration : float
        Total length of the input audio (seconds).
    chunk_durations : List[float]
        Length of each produced chunk (seconds).
    out_dir : Path
        Directory where chunks have been saved.
    """
    if not chunk_durations:
        print("No chunks were generated.")
        return

    count = len(chunk_durations)
    avg = sum(chunk_durations) / count
    shortest = min(chunk_durations)
    longest = max(chunk_durations)

    print("-" * 32)
    print(f"Original Duration : {original_duration:.2f} sec")
    print(f"Chunk Count       : {count}")
    print(f"Average Chunk Dur : {avg:.2f} sec")
    print(f"Shortest Chunk    : {shortest:.2f} sec")
    print(f"Longest Chunk     : {longest:.2f} sec")
    print(f"Saved Directory   : {out_dir.as_posix()}")
    print("-" * 32)


# --------------------------------------------------------------------------- #
# CLI handling
# --------------------------------------------------------------------------- #

def _parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a speech recording into fixed‑length chunks."
    )
    parser.add_argument(
        "audio_path",
        type=Path,
        help="Path to the input wav file (e.g., temp/customer_speech.wav).",
    )
    parser.add_argument(
        "--min",
        type=float,
        default=10.0,
        help="Minimum chunk length in seconds (default: 10).",
    )
    parser.add_argument(
        "--max",
        type=float,
        default=30.0,
        help="Maximum chunk length in seconds (default: 30).",
    )
    parser.add_argument(
        "--target",
        type=float,
        default=20.0,
        help="Preferred chunk length in seconds (default: 20).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("temp") / "chunks",
        help="Directory to store generated chunks (default: temp/chunks).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_cli()

    if args.min <= 0 or args.max <= 0 or args.target <= 0:
        print("Error: Length arguments must be positive numbers.", file=sys.stderr)
        sys.exit(1)
    if args.min > args.max:
        print("Error: --min cannot be larger than --max.", file=sys.stderr)
        sys.exit(1)

    print("-" * 32)
    print("Loading audio...")
    print()

    # ------------------------------------------------------------------- #
    # Load source audio
    # ------------------------------------------------------------------- #
    try:
        waveform, sr = load_audio(args.audio_path)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except AudioLoadError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    original_duration = waveform.shape[0] / sr
    print(f"Sample Rate      : {sr} Hz")
    print(f"Original Duration: {original_duration:.2f} sec")
    print()

    # ------------------------------------------------------------------- #
    # Split into chunks
    # ------------------------------------------------------------------- #
    chunks = split_into_chunks(
        waveform,
        sr,
        min_len=args.min,
        max_len=args.max,
        target_len=args.target,
    )
    chunk_durations = [
        (end - start) / sr for start, end in chunks
    ]

    # ------------------------------------------------------------------- #
    # Save chunks
    # ------------------------------------------------------------------- #
    try:
        saved_paths = save_chunks(waveform, sr, chunks, args.out_dir)
    except ChunkSaveError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------- #
    # Report statistics
    # ------------------------------------------------------------------- #
    print_statistics(original_duration, chunk_durations, args.out_dir)

    print("Chunk generation completed successfully.")
    print("-" * 32)


if __name__ == "__main__":
    main()

