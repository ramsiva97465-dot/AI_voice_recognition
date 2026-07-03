#!/usr/bin/env python
# prototype_embedding.py
# ---------------------------------------------------------------
# Minimal prototype that extracts a speaker embedding using the
# SpeechBrain ECAPA‑TDNN model (speechbrain/spkrec-ecapa-voxceleb).
# ---------------------------------------------------------------
# Steps:
#   1. Load a WAV file (mono or stereo) from the command line.
#   2. Resample to 16 kHz (the sampling rate expected by the model).
#   3. Print audio sample rate and duration.
#   4. Initialise the SpeechBrain ECAPA‑TDNN encoder (saved in models/ecapa).
#   5. Compute a single speaker embedding.
#   6. Print diagnostics (shape, dimension, first 10 values).
#   7. Save the embedding as a 1‑D NumPy vector (embedding.npy).
# ---------------------------------------------------------------

import argparse
import pathlib
import sys

import numpy as np
import torch
import librosa
import soundfile as sf

# SpeechBrain provides the EncoderClassifier class for speaker embedding.
try:
    from speechbrain.inference import EncoderClassifier  # type: ignore
except Exception as e:
    print("Failed to import SpeechBrain EncoderClassifier:")
    print(e)
    sys.exit(1)


def load_waveform(wav_path: pathlib.Path) -> tuple[torch.Tensor, int]:
    """
    Load a WAV file and return a mono tensor and its original sample rate.

    Parameters
    ----------
    wav_path : pathlib.Path
        Path to the input WAV file.

    Returns
    -------
    waveform : torch.Tensor
        Tensor of shape (1, num_samples) (float32).
    sr : int
        Original sample rate of the audio file.
    """
    try:
        # Load audio with soundfile (supports many formats without FFmpeg)
        audio_np, sr = sf.read(str(wav_path))  # audio_np shape: (samples,) or (samples, channels)
    except Exception as exc:
        print(f"Error loading audio file '{wav_path}': {exc}")
        sys.exit(1)

    # If the audio has multiple channels, convert to mono by averaging
    if audio_np.ndim > 1:
        audio_np = np.mean(audio_np, axis=1)
    # Convert NumPy array to a torch tensor of shape (1, num_samples)
    waveform = torch.from_numpy(audio_np).float().unsqueeze(0).contiguous()

    return waveform, sr


def main() -> None:
    # -----------------------------------------------------------------
    # Parse command‑line arguments.
    # -----------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Generate a speaker embedding using SpeechBrain ECAPA‑TDNN"
    )
    parser.add_argument(
        "wav_path",
        type=pathlib.Path,
        help="Path to the input WAV file (mono or stereo).",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("embedding.npy"),
        help="File where the embedding will be saved (NumPy .npy).",
    )
    args = parser.parse_args()

    # Verify that the provided audio file exists
    if not args.wav_path.exists():
        print(f"Audio file not found: {args.wav_path}")
        sys.exit(1)

    # -----------------------------------------------------------------
    # Load the audio file.
    # -----------------------------------------------------------------
    waveform, sr = load_waveform(args.wav_path)

    # -----------------------------------------------------------------
    # Resample to 16 kHz if needed (ECAPA‑TDNN expects 16000 Hz).
    # -----------------------------------------------------------------
    target_sr = 16000
    if sr != target_sr:
        # Convert torch tensor to NumPy for librosa resampling
        audio_np = waveform.squeeze(0).numpy()
        audio_np = librosa.resample(
            audio_np,
            orig_sr=sr,
            target_sr=target_sr,
        )
        waveform = torch.from_numpy(audio_np).float().unsqueeze(0)
        sr = target_sr

    # Print audio metadata before embedding.
    duration_seconds = waveform.shape[1] / sr
    print(f"Audio Sample Rate : {sr} Hz")
    print(f"Audio Duration    : {duration_seconds:.2f} seconds")
    print(f"Audio Tensor Shape : {waveform.shape}")
    # SpeechBrain expects shape (batch, num_samples). Add batch dim.
    batched_waveform = waveform  # (1, T)

    # -----------------------------------------------------------------
    # Initialise the ECAPA‑TDNN encoder.
    # -----------------------------------------------------------------
    print("Loading SpeechBrain ECAPA‑TDNN model...")
    try:
        # Model will be cached under the workspace's models/ecapa directory.
        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="models/ecapa",
        )
    except Exception as exc:
        print(f"Failed to load the SpeechBrain model: {exc}")
        sys.exit(1)
    print("Model loaded successfully.")

    # -----------------------------------------------------------------
    # Compute the speaker embedding.
    # -----------------------------------------------------------------
    try:
        with torch.no_grad():
            embedding = classifier.encode_batch(batched_waveform)  # (1, dim) or (1,1,dim)
    except Exception as exc:
        print(f"Failed to generate embedding: {exc}")
        sys.exit(1)

    # -----------------------------------------------------------------
    # Print diagnostics (shape‑independent handling).
    # -----------------------------------------------------------------
    print(f"Embedding Shape : {embedding.shape}")
    # Convert to a 1‑D NumPy array irrespective of extra singleton dims.
    embedding_np = embedding.squeeze().cpu().numpy()
    print(f"Embedding Dimension : {embedding_np.shape[0]}")
    print("First 10 values :", embedding_np[:10])

    # -----------------------------------------------------------------
    # Save the embedding as a 1‑D NumPy vector.
    # -----------------------------------------------------------------
    np.save(args.output, embedding_np)
    print(f"Embedding saved to : {args.output}")


if __name__ == "__main__":
    main()
