#!/usr/bin/env python
# ---------------------------------------------------------------
# app/embedding.py
# ---------------------------------------------------------------
# Provides a single public function:
#   generate_embedding(audio_path: str) -> numpy.ndarray
# which loads a wav file, resamples to 16 kHz, runs the
# SpeechBrain ECAPA‑TDNN model, and returns a 192‑dimensional
# NumPy vector (shape = (192,)).
# ---------------------------------------------------------------

from __future__ import annotations

import pathlib
import sys
from typing import Optional

import numpy as np
import torch
import librosa
import soundfile as sf

# ---------------------------------------------------------------
# Lazy import of the SpeechBrain EncoderClassifier.
# This import may fail if the optional k2 integration is missing,
# but the pretrained ECAPA model does not require it.
# ---------------------------------------------------------------
try:
    from speechbrain.pretrained import EncoderClassifier  # type: ignore
except Exception as e:  # pragma: no cover
    print("Failed to import SpeechBrain EncoderClassifier:")
    print(e)
    sys.exit(1)

# ---------------------------------------------------------------
# Global (cached) model instance – loaded only once.
# ---------------------------------------------------------------
_model: Optional[EncoderClassifier] = None


def _load_model() -> EncoderClassifier:
    """Load the ECAPA‑TDNN model (cached on first call)."""
    global _model
    if _model is None:
        try:
            _model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="models/ecapa",
            )
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"Failed to load the SpeechBrain model: {exc}") from exc
    return _model


def _load_waveform(wav_path: pathlib.Path) -> tuple[torch.Tensor, int]:
    """Load a WAV file, convert to mono, and return a (1, N) torch Tensor together with its original sample rate."""
    try:
        audio_np, sr = sf.read(str(wav_path))  # (samples,) or (samples, channels)
    except Exception as exc:
        raise RuntimeError(f"Error loading audio file '{wav_path}': {exc}") from exc

    # Stereo → mono by averaging all channels
    if audio_np.ndim > 1:
        audio_np = np.mean(audio_np, axis=1)

    # (1, num_samples) tensor, made contiguous for safety
    waveform = torch.from_numpy(audio_np).float().unsqueeze(0).contiguous()
    return waveform, sr


def generate_embedding(audio_path: str) -> np.ndarray:
    """Public API.

    Parameters
    ----------
    audio_path : str
        Path to a wav (or any format supported by soundfile) audio file.

    Returns
    -------
    numpy.ndarray
        1‑D array of shape (192,) containing the ECAPA‑TDNN speaker embedding.

    Raises
    ------
    RuntimeError
        If any step (loading, resampling, model inference) fails.
    """
    wav_path = pathlib.Path(audio_path)

    # -----------------------------------------------------------------
    # Load and optionally resample.
    # -----------------------------------------------------------------
    waveform, sr = _load_waveform(wav_path)

    target_sr = 16000
    if sr != target_sr:
        # librosa works on NumPy arrays
        audio_np = waveform.squeeze(0).numpy()
        audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=target_sr)
        waveform = torch.from_numpy(audio_np).float().unsqueeze(0)
        sr = target_sr
        # Trim leading and trailing silence
        audio_np, _ = librosa.effects.trim(audio_np, top_db=25)
        # Normalize waveform volume
        peak = np.max(np.abs(audio_np))
        if peak > 0:
            audio_np = audio_np / peak
        # Convert back to a contiguous torch tensor
        waveform = torch.from_numpy(audio_np).float().unsqueeze(0).contiguous()

    # -----------------------------------------------------------------
    # Model inference (model is cached after the first call).
    # -----------------------------------------------------------------
    model = _load_model()
    try:
        with torch.no_grad():
            # Returns (1, dim) or (1, 1, dim)
            embedding = model.encode_batch(waveform)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Failed to generate embedding: {exc}") from exc

    # Convert to a plain 1‑D NumPy vector
    embedding_np = embedding.squeeze().cpu().numpy()
    return embedding_np

# Pre-load the model at import time so the first request doesn't timeout
_load_model()
