# utils.py - Audio loading and preprocessing utilities
"""Utility functions for handling audio files in the Speaker Recognition Service.

The functions here are deliberately lightweight and have no external side‑effects.
They are written with type hints and exhaustive inline comments so that a beginner
can understand each step.
"""

import pathlib
from typing import Tuple

import numpy as np
import torch
import torchaudio

# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------
# The tele‑calling platform uses 8 kHz PCM audio.  All models expect
# a sampling rate of 16 kHz, but we will resample to 8 kHz to keep the
# pipeline compatible with the eventual live audio source.
TARGET_SAMPLE_RATE = 8000

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def load_wav(file_path: str) -> Tuple[torch.Tensor, int]:
    """Load a WAV file and return a 1‑D Torch tensor and its original sample rate.

    Parameters
    ----------
    file_path: str
        Path to the ``.wav`` file on disk.

    Returns
    -------
    Tuple[torch.Tensor, int]
        ``waveform`` – a tensor of shape ``(num_samples,)`` (mono).
        ``sample_rate`` – integer sampling rate of the original file.
    """
    # ``torchaudio.load`` returns a tensor of shape (channels, samples).
    # Most tele‑calling audio is mono; if we encounter multi‑channel data we
    # simply average across channels to obtain a mono signal.
    wav_path = pathlib.Path(file_path)
    if not wav_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    waveform, sr = torchaudio.load(wav_path)  # type: torch.Tensor, int
    # Convert to mono if necessary
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    # Remove channel dimension – we work with a 1‑D tensor throughout the pipeline
    waveform = waveform.squeeze(0)
    return waveform, sr


def resample(waveform: torch.Tensor, orig_sr: int, target_sr: int = TARGET_SAMPLE_RATE) -> torch.Tensor:
    """Resample an audio waveform to ``target_sr`` using high‑quality sinc interpolation.

    Parameters
    ----------
    waveform: torch.Tensor
        1‑D tensor containing raw audio samples.
    orig_sr: int
        Original sampling rate of ``waveform``.
    target_sr: int, default ``TARGET_SAMPLE_RATE``
        Desired sampling rate after resampling.

    Returns
    -------
    torch.Tensor
        Resampled waveform (still 1‑D).
    """
    if orig_sr == target_sr:
        return waveform
    # ``Resample`` expects (num_channels, num_samples).  We temporarily add a
    # channel dimension, perform the operation, then squeeze back.
    resampler = torchaudio.transforms.Resample(orig_sr, target_sr, dtype=waveform.dtype)
    resampled = resampler(waveform.unsqueeze(0)).squeeze(0)
    return resampled


def normalize(waveform: torch.Tensor) -> torch.Tensor:
    """Apply zero‑mean, unit‑variance normalization to a waveform.

    Normalization is crucial for neural‑network based speaker embeddings because
    the model has been trained on roughly standardized audio.  Without this step
    the embedding quality degrades dramatically.
    """
    if waveform.numel() == 0:
        raise ValueError("Empty waveform cannot be normalized")
    mean = waveform.mean()
    std = waveform.std()
    # Guard against division by zero in pathological cases (silence).
    if std < 1e-9:
        return waveform - mean
    return (waveform - mean) / std


def preprocess_audio(file_path: str, target_sr: int = TARGET_SAMPLE_RATE) -> torch.Tensor:
    """High‑level helper that loads, resamples, and normalizes a WAV file.

    The returned tensor is ready to be fed into the ECAPA‑TDNN encoder.
    """
    wav, sr = load_wav(file_path)
    wav = resample(wav, sr, target_sr)
    wav = normalize(wav)
    # Ensure the tensor is ``float32`` – the model expects float32 inputs.
    return wav.to(torch.float32)

# End of utils.py
