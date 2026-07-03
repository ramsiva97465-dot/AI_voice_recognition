# Configuration module for the Voice Biometrics project

# Directory where speaker embeddings are stored
DATABASE_DIR = "database/speakers"

# Temporary working directory used for intermediate audio files
TEMP_DIR = "temp"

# Default cosine‑similarity threshold for verification / quality filtering
THRESHOLD = 0.75

# Dimensionality of the ECAPA‑TDNN embedding vector (as produced by generate_embedding)
EMBEDDING_SIZE = 192
