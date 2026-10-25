"""Storage and Persistence Service for Helix Semantic Archive.

Manages image files, metadata persistence (JSON), vector caching, deduplication (SHA-256),
and thread-safe index operations.
"""

import os
import json
import uuid
import hashlib
import logging
import threading
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image

from config import (
    UPLOAD_DIR,
    METADATA_FILE,
    EMBEDDINGS_FILE,
    ALLOWED_EXTENSIONS,
    MAX_IMAGE_FILE_SIZE,
)
from services.dna_service import get_dna_statistics, format_dna_preview

logger = logging.getLogger(__name__)


class StorageService:
    """Thread-safe index storage engine with on-disk JSON/NPY persistence."""

    _instance: Optional["StorageService"] = None

    def __init__(self):
        self._lock = threading.Lock()
        self._items: List[Dict[str, Any]] = []
        self._sha256_set = set()
        self._load_from_disk()

    @classmethod
    def get_instance(cls) -> "StorageService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_from_disk(self):
        """Loads previously indexed metadata and vector embeddings from disk."""
        with self._lock:
            if not os.path.exists(METADATA_FILE):
                logger.info("No existing metadata file found at %s. Starting fresh.", METADATA_FILE)
                return

            try:
                with open(METADATA_FILE, "r", encoding="utf-8") as f:
                    metadata = json.load(f)

                if os.path.exists(EMBEDDINGS_FILE):
                    embeddings = np.load(EMBEDDINGS_FILE)
                else:
                    embeddings = None

                valid_items = []
                sha_set = set()
                for idx, entry in enumerate(metadata):
                    img_path = os.path.join(UPLOAD_DIR, entry["filename"])
                    if not os.path.exists(img_path):
                        continue

                    # Attach loaded embedding vector
                    if embeddings is not None and idx < len(embeddings):
                        entry["embedding"] = embeddings[idx].astype(np.float32)
                    elif "embedding" in entry and isinstance(entry["embedding"], list):
                        entry["embedding"] = np.array(entry["embedding"], dtype=np.float32)
                    else:
                        continue

                    valid_items.append(entry)
                    if "sha256" in entry:
                        sha_set.add(entry["sha256"])

                self._items = valid_items
                self._sha256_set = sha_set
                logger.info("Loaded %d persisted items from %s", len(self._items), METADATA_FILE)
            except Exception as exc:
                logger.error("Failed to load persisted metadata: %s", exc)
                self._items = []
                self._sha256_set = set()

    def _save_to_disk(self):
        """Serializes current index metadata and vector embeddings to disk."""
        try:
            metadata_to_save = []
            embeddings_list = []

            for item in self._items:
                embeddings_list.append(item["embedding"])
                clean_item = {
                    "id": item["id"],
                    "name": item["name"],
                    "filename": item["filename"],
                    "url": item.get("url", f"/uploads/{item['filename']}"),
                    "file_size": item.get("file_size", 0),
                    "dimensions": item.get("dimensions", [224, 224]),
                    "sha256": item.get("sha256", ""),
                    "dna": item["dna"],
                    "dna_preview": item.get("dna_preview") or format_dna_preview(item["dna"]),
                    "dna_length": len(item["dna"]),
                    "gc_content": item.get("gc_content", 50.0),
                    "indexed_at": item.get("indexed_at", datetime.now(timezone.utc).isoformat()),
                }
                metadata_to_save.append(clean_item)

            with open(METADATA_FILE, "w", encoding="utf-8") as f:
                json.dump(metadata_to_save, f, indent=2)

            if embeddings_list:
                embeddings_arr = np.vstack(embeddings_list).astype(np.float32)
                np.save(EMBEDDINGS_FILE, embeddings_arr)
            elif os.path.exists(EMBEDDINGS_FILE):
                os.remove(EMBEDDINGS_FILE)

            logger.info("Persisted %d indexed items to disk.", len(metadata_to_save))
        except Exception as exc:
            logger.error("Failed to persist index to disk: %s", exc)

    def is_allowed_file(self, filename: str) -> bool:
        ext = os.path.splitext(filename)[1].lower()
        return ext in ALLOWED_EXTENSIONS

    def compute_sha256(self, file_bytes: bytes) -> str:
        return hashlib.sha256(file_bytes).hexdigest()

    def is_duplicate(self, sha256_hash: str) -> bool:
        with self._lock:
            return sha256_hash in self._sha256_set

    def add_item(
        self,
        name: str,
        image_bytes: bytes,
        image: Image.Image,
        embedding: np.ndarray,
        dna_sequence: str,
    ) -> Dict[str, Any]:
        """Saves image to disk, calculates metadata & DNA stats, and appends to index.

        Args:
            name: Original filename.
            image_bytes: Raw binary content of the file.
            image: Decoded PIL Image.
            embedding: 768-D normalized float32 vector.
            dna_sequence: 3,072-nt simulated DNA string.

        Returns:
            Dictionary with added image's metadata.
        """
        sha256_hash = self.compute_sha256(image_bytes)

        with self._lock:
            if sha256_hash in self._sha256_set:
                raise ValueError(f"Image '{name}' is already indexed (duplicate content).")

        item_id = uuid.uuid4().hex
        ext = os.path.splitext(name)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            ext = ".jpg"
        filename = f"{item_id}{ext}"
        filepath = os.path.join(UPLOAD_DIR, filename)

        # Save image file to local uploads directory
        image.save(filepath, quality=92, optimize=True)

        dna_stats = get_dna_statistics(dna_sequence)

        item = {
            "id": item_id,
            "name": name,
            "filename": filename,
            "url": f"/uploads/{filename}",
            "file_size": len(image_bytes),
            "dimensions": list(image.size),  # [width, height]
            "sha256": sha256_hash,
            "embedding": embedding,
            "dna": dna_sequence,
            "dna_preview": format_dna_preview(dna_sequence),
            "dna_length": len(dna_sequence),
            "gc_content": dna_stats["gc_content"],
            "dna_stats": dna_stats,
            "indexed_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        }

        with self._lock:
            self._items.append(item)
            self._sha256_set.add(sha256_hash)
            self._save_to_disk()

        return item

    def get_all_items(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._items)

    def get_item_by_id(self, item_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            for item in self._items:
                if item["id"] == item_id:
                    return item
            return None

    def get_count(self) -> int:
        with self._lock:
            return len(self._items)

    def clear(self):
        """Clears all indexed images and deletes stored upload files and metadata."""
        with self._lock:
            for item in self._items:
                filepath = os.path.join(UPLOAD_DIR, item["filename"])
                if os.path.exists(filepath):
                    try:
                        os.remove(filepath)
                    except OSError:
                        pass
            self._items = []
            self._sha256_set = set()
            if os.path.exists(METADATA_FILE):
                os.remove(METADATA_FILE)
            if os.path.exists(EMBEDDINGS_FILE):
                os.remove(EMBEDDINGS_FILE)


def get_storage_service() -> StorageService:
    return StorageService.get_instance()
