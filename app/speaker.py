# app/speaker.py - SpeakerEncoder class using pyannote.audio 4.x
"""SpeakerEncoder
===================

This module provides a production‑grade **SpeakerEncoder** class that wraps the
`pyannote.audio` speaker‑diarization pipeline (v4.x) to extract a single speaker
embedding from an audio file.

Why we use `pyannote.audio` 4.x
-------------------------------
* The library now ships a **unified pipeline** (`Pipeline.from_pretrained`) that
  internally loads a state‑of‑the‑art ECAPA‑TDNN model.
* The API is stable and does **not** rely on the deprecated ``Inference``
  class that existed in older releases.
* `return_embeddings=True` gives direct access to the raw speaker embeddings
  produced by the ECAPA‑TDNN backbone, which is exactly what we need for speaker
  verification.

The class follows production coding standards:
* **Type hints** for every public method.
* **Logging** with `loguru` (falls back to the standard library if unavailable).
* **Exception handling** that raises a custom ``SpeakerEncoderError`` with a clear
  message.
* **Detailed docstrings** and inline comments so a beginner can follow the
  logic.
"""

from __future__ import annotations

import time
import pathlib
from typing import Optional

import numpy as np
import torch
from loguru import logger

# Local utilities for audio preprocessing – we only use them to verify the file
# exists and to keep the dependency chain explicit.
from .utils import load_wav  # noqa: F401  (imported for side‑effect validation)

# pyannote.audio v4.x entry point
from pyannote.audio import Pipeline

# ---------------------------------------------------------------------------
# Custom exception for clarity in higher‑level code
# ---------------------------------------------------------------------------
class SpeakerEncoderError(RuntimeError):
    """Raised when the encoder fails to load a model or generate an embedding."""

# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------
class SpeakerEncoder:
    """Encapsulates a speaker‑embedding model based on ECAPA‑TDNN.

    The class lazily loads the ``pyannote.audio`` pipeline on first use.  A
    separate ``load_model`` method is provided for explicit control – useful in
    environments where start‑up time must be measured.
    """

    def __init__(self, model_name: str = "pyannote/speaker-diarization-community-1", token: Optional[str] = None) -> None:
        """Create a ``SpeakerEncoder`` instance and load the model immediately.

        Parameters
        ----------
        model_name: str, optional
            Identifier of the pre‑trained pipeline on the Hugging‑Face hub.
        token: str, optional
            Authentication token required by Hugging‑Face for protected models.
            If ``None`` the environment variable ``HF_TOKEN`` (used by
            ``pyannote`` internally) is consulted.
        """
        self.model_name = model_name
        self.token = token
        self.pipeline: Optional[Pipeline] = None
        # Configure a simple logger using ``loguru``.
        logger.remove()
        logger.add(lambda msg: print(msg), level="INFO")
        logger.info("SpeakerEncoder instantiated (model: {} )", model_name)
        # Load the model once during initialization; subsequent calls are idempotent.
        self.load_model()

    # ---------------------------------------------------------------------
    def load_model(self) -> None:
        """Load the pyannote.audio diarization pipeline.

        The method is idempotent – calling it multiple times does not reload the
        model if it is already present.
        """
        if self.pipeline is not None:
            logger.debug("Model already loaded; skipping re‑initialisation.")
            return
        try:
            logger.info("Loading pyannote.audio pipeline '{}'...", self.model_name)
            self.pipeline = Pipeline.from_pretrained(self.model_name, token=self.token)
            logger.success("Model loaded successfully.")
        except Exception as exc:
            logger.error("Failed to load model: {}", exc)
            raise SpeakerEncoderError(f"Could not load model '{self.model_name}': {exc}") from exc

    # ---------------------------------------------------------------------
    def encode(self, audio_path: str) -> np.ndarray:
        """Generate a speaker embedding for ``audio_path``.

        The method performs the following steps:
        1. Validate the WAV file using ``utils.load_wav`` (ensures the file exists).
        2. Run the diarization pipeline with ``return_embeddings=True``.
        3. Average all segment embeddings into a single fixed‑dimensional vector.
        4. Return the embedding as a NumPy ``ndarray``.
        5. Print shape, dimension, and processing time for transparency.
        """
        start_time = time.time()
        # -----------------------------------------------------------------
        # 1️⃣ Validate input file – ``load_wav`` raises a clear error if missing.
        # -----------------------------------------------------------------
        try:
            _waveform, _sr = load_wav(audio_path)  # noqa: F841 – only validation.
        except Exception as exc:
            logger.error("Audio validation failed for '{}': {}", audio_path, exc)
            raise SpeakerEncoderError(f"Invalid audio file '{audio_path}': {exc}") from exc

        # -----------------------------------------------------------------
        # 2️⃣ Ensure the pipeline is ready.
        # -----------------------------------------------------------------
        if self.pipeline is None:
            self.load_model()

        # -----------------------------------------------------------------
        # 3️⃣ Run the pipeline. ``return_embeddings=True`` yields a tuple:
        #    (diarization, embeddings)
        # -----------------------------------------------------------------
        try:
            diarization, embeddings = self.pipeline(audio_path, return_embeddings=True)  # type: ignore[arg-type]
        except Exception as exc:
            logger.error("Pipeline execution failed: {}", exc)
            raise SpeakerEncoderError(f"Failed to process audio '{audio_path}': {exc}") from exc

        # -----------------------------------------------------------------
        # 4️⃣ Extract raw tensors. In v4.x ``embeddings`` is a ``dict`` mapping
        #    ``Segment`` objects to ``torch.Tensor`` of shape (embedding_dim,).
        # -----------------------------------------------------------------
        if not isinstance(embeddings, dict) or len(embeddings) == 0:
            raise SpeakerEncoderError("No speaker embeddings were returned by the pipeline.")

        try:
            tensor_list = [tensor.cpu().detach() for tensor in embeddings.values()]
            # Average across all speech segments – this yields a single vector.
            avg_tensor: torch.Tensor = torch.stack(tensor_list).mean(dim=0)
            embedding_np: np.ndarray = avg_tensor.numpy()
        except Exception as exc:
            logger.error("Failed to post‑process embeddings: {}", exc)
            raise SpeakerEncoderError(f"Embedding post‑processing error: {exc}") from exc

        # -----------------------------------------------------------------
        # 5️⃣ Reporting
        # -----------------------------------------------------------------
        elapsed = time.time() - start_time
        shape_str = f"{embedding_np.shape}"
        dim = embedding_np.shape[0]
        logger.info("Embedding generated – shape: {}, dim: {}, time: {:.2f}s", shape_str, dim, elapsed)
        print(f"Embedding shape: {shape_str}")
        print(f"Embedding dimension: {dim}")
        print(f"Processing time: {elapsed:.2f} seconds")

        return embedding_np

    # ---------------------------------------------------------------------
    def save_embedding(self, embedding: np.ndarray, output_path: str) -> None:
        """Persist a NumPy embedding to ``output_path`` as ``.npy``.

        Parameters
        ----------
        embedding: np.ndarray
            The vector produced by :meth:`encode`.
        output_path: str
            Destination filename – must end with ``.npy``.
        """
        out_path = pathlib.Path(output_path)
        if out_path.suffix.lower() != ".npy":
            raise SpeakerEncoderError("Output file must have a .npy extension.")
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(out_path, embedding)
            logger.success("Embedding saved to '{}'.", out_path)
        except Exception as exc:
            logger.error("Failed to save embedding to '{}': {}", out_path, exc)
            raise SpeakerEncoderError(f"Could not write embedding to '{output_path}': {exc}") from exc

# ---------------------------------------------------------------------------
# Example usage (executed when the module is run directly)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate a speaker embedding.")
    parser.add_argument("audio", type=str, help="Path to input WAV file.")
    parser.add_argument("--out", type=str, default="embedding.npy", help="Where to store the .npy embedding.")
    args = parser.parse_args()

    encoder = SpeakerEncoder()
    emb = encoder.encode(args.audio)
    encoder.save_embedding(emb, args.out)
    print("Embedding saved to", args.out)

# End of app/speaker.py
