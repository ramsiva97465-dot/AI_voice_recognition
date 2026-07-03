#!/usr/bin/env python
# ---------------------------------------------------------------
# enroll.py
# ---------------------------------------------------------------
# Command‑line tool to enroll a speaker by generating a 192‑dimensional ECAPA‑TDNN embedding
# and storing it under: database/speakers/<SpeakerName>.npy
#
# Usage example:
#   python enroll.py audio/test/siva.wav.wav --name Siva
# ---------------------------------------------------------------

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

# Import the shared embedding routine
from app.embedding import generate_embedding


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enroll a speaker by generating a SpeechBrain ECAPA‑TDNN embedding."
    )
    parser.add_argument(
        "audio_path",
        type=str,
        help="Path to the input audio file (wav, flac, etc.).",
    )
    parser.add_argument(
        "--name",
        required=True,
        type=str,
        help="Speaker name – used as the filename for the stored embedding.",
    )
    return parser.parse_args()


def _ensure_speaker_dir() -> pathlib.Path:
    """Create the database/speakers directory if it does not exist."""
    speakers_dir = pathlib.Path("database") / "speakers"
    speakers_dir.mkdir(parents=True, exist_ok=True)
    return speakers_dir


def main() -> None:
    args = _parse_args()

    audio_path = pathlib.Path(args.audio_path)
    if not audio_path.is_file():
        print(f"Error: audio file not found – {audio_path}")
        sys.exit(1)

    try:
        embedding = generate_embedding(str(audio_path))
    except RuntimeError as exc:
        print(f"Embedding generation failed: {exc}")
        sys.exit(1)

    # -----------------------------------------------------------------
    # Save the embedding.
    # -----------------------------------------------------------------
    speakers_dir = _ensure_speaker_dir()
    safe_name = args.name.strip().replace(" ", "_")
    out_path = speakers_dir / f"{safe_name}.npy"

    try:
        np.save(out_path, embedding)
    except Exception as exc:  # pragma: no cover
        print(f"Failed to save embedding to '{out_path}': {exc}")
        sys.exit(1)

    # -----------------------------------------------------------------
    # Reporting.
    # -----------------------------------------------------------------
    print("\nEnrollment Successful")
    print(f"Speaker Name   : {args.name}")
    print(f"Embedding Dim. : {embedding.shape[0]}")
    print(f"Saved To       : {out_path}\n")


if __name__ == "__main__":
    main()
