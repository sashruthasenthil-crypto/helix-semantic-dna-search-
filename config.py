"""Configuration and application constants for Helix Semantic Archive."""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data storage paths
DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
METADATA_FILE = os.path.join(DATA_DIR, "metadata.json")
EMBEDDINGS_FILE = os.path.join(DATA_DIR, "embeddings.npy")
SAMPLE_IMAGES_DIR = os.path.join(BASE_DIR, "sample_images")

# Ensure required storage directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(SAMPLE_IMAGES_DIR, exist_ok=True)

# Vision Transformer Model configuration (Safe CPU-first configuration without hard torch import)
MODEL_NAME = "google/vit-base-patch16-224-in21k"
DEVICE = "cpu"
EMBEDDING_DIM = 768

# Allowed upload extensions
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MAX_CONTENT_LENGTH = 30 * 1024 * 1024  # 30 MB maximum request payload
MAX_IMAGE_FILE_SIZE = 10 * 1024 * 1024  # 10 MB per image

# k-NN Graph Message Passing Refinement defaults
DEFAULT_KNN_NEIGHBORS = 3
DEFAULT_GRAPH_ALPHA = 0.35  # Diffusion weight (1 - alpha) * x + alpha * (A @ x)

# Simulated DNA mapping specifications
# 768 float dimensions -> 768 bytes (uint8: 0..255) -> 3,072 base-4 nucleotides (A, C, G, T)
NUCLEOTIDE_MAP = {
    "00": "A",
    "01": "C",
    "10": "G",
    "11": "T",
}
SIMULATED_DNA_LENGTH = EMBEDDING_DIM * 4  # 3072 nucleotides
