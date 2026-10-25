"""Helix — Semantic Archive: AI-Based Hybrid Semantic Search for DNA Data Storage.

Flask application entrypoint exposing web routes and REST APIs.
Uses Vision Transformer (google/vit-base-patch16-224-in21k) and k-NN graph similarity refinement,
with a safe CPU-compatible fallback when PyTorch DLL loading is restricted by system policies.
"""

import os
import io
import time
import logging
from typing import List, Dict, Any
from flask import Flask, render_template, request, jsonify, send_from_directory, abort
from PIL import Image

from config import (
    UPLOAD_DIR,
    SAMPLE_IMAGES_DIR,
    MAX_CONTENT_LENGTH,
    MAX_IMAGE_FILE_SIZE,
    MODEL_NAME,
    DEVICE,
    EMBEDDING_DIM,
    SIMULATED_DNA_LENGTH,
)
from vit_service import get_vit_service
from services.dna_service import (
    encode_embedding_to_dna,
    validate_dna_sequence,
    get_dna_statistics,
    format_dna_preview,
)
from graph_search_service import search_similar_images
from storage_service import get_storage_service

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("helix_app")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# Initialize services (safe fallback prevents startup crash if torch is blocked)
vit_service = get_vit_service()
storage_service = get_storage_service()


# ==========================================
# PAGE ROUTES
# ==========================================


@app.route("/")
def page_overview():
    """Renders the Overview Dashboard."""
    return render_template(
        "overview.html",
        active_page="overview",
        indexed_count=storage_service.get_count(),
        device=vit_service.device,
        model_name=MODEL_NAME,
        vit_ready=vit_service.is_ready,
        vit_error=vit_service.error_reason,
    )


@app.route("/image-index")
def page_image_index():
    """Renders the Image Indexing Page."""
    return render_template(
        "index_images.html",
        active_page="index",
        indexed_count=storage_service.get_count(),
        device=vit_service.device,
        vit_ready=vit_service.is_ready,
        vit_error=vit_service.error_reason,
    )


@app.route("/semantic-search")
def page_semantic_search():
    """Renders the Semantic Image Search Page."""
    return render_template(
        "search.html",
        active_page="search",
        indexed_count=storage_service.get_count(),
        device=vit_service.device,
        vit_ready=vit_service.is_ready,
        vit_error=vit_service.error_reason,
    )


@app.route("/pipeline")
def page_pipeline():
    """Renders the Visual Pipeline Architecture Page."""
    return render_template(
        "pipeline.html",
        active_page="pipeline",
        indexed_count=storage_service.get_count(),
        device=vit_service.device,
        vit_ready=vit_service.is_ready,
        vit_error=vit_service.error_reason,
    )


# ==========================================
# STATIC FILE SERVING
# ==========================================


@app.route("/uploads/<path:filename>")
def serve_upload(filename):
    """Safely serves uploaded images from the data/uploads directory."""
    return send_from_directory(UPLOAD_DIR, filename)


@app.route("/samples/<path:filename>")
def serve_sample(filename):
    """Safely serves pre-generated sample images."""
    return send_from_directory(SAMPLE_IMAGES_DIR, filename)


# ==========================================
# REST API ENDPOINTS
# ==========================================


@app.route("/api/status", methods=["GET"])
def api_status():
    """Returns application health, model status, and indexed item counts."""
    count = storage_service.get_count()
    return jsonify({
        "status": "ready" if vit_service.is_ready else "degraded",
        "ready": vit_service.is_ready,
        "device": vit_service.device,
        "indexed": count,
        "model_name": MODEL_NAME,
        "embedding_dim": EMBEDDING_DIM,
        "dna_length": SIMULATED_DNA_LENGTH,
        "storage_mode": "persistent_json_npy",
        "model_error": vit_service.error_reason,
    })


@app.route("/api/images", methods=["GET"])
def api_get_images():
    """Returns list of indexed images and summary metadata."""
    items = storage_service.get_all_items()
    clean_items = []
    for item in items:
        clean_items.append({
            "id": item["id"],
            "name": item["name"],
            "filename": item["filename"],
            "url": item["url"],
            "file_size": item.get("file_size", 0),
            "dimensions": item.get("dimensions", [224, 224]),
            "dna_preview": item.get("dna_preview") or format_dna_preview(item["dna"]),
            "dna_length": len(item["dna"]),
            "gc_content": item.get("gc_content", 50.0),
            "indexed_at": item.get("indexed_at", ""),
        })
    return jsonify({"images": clean_items, "total": len(clean_items)})


@app.route("/api/images/<image_id>/dna", methods=["GET"])
def api_get_image_dna(image_id):
    """Returns the full simulated DNA sequence and detailed nucleotide composition for an image."""
    item = storage_service.get_item_by_id(image_id)
    if not item:
        return jsonify({"error": "Image not found."}), 404

    dna = item["dna"]
    stats = get_dna_statistics(dna)
    return jsonify({
        "id": item["id"],
        "name": item["name"],
        "dna_sequence": dna,
        "dna_length": len(dna),
        "statistics": stats,
        "disclaimer": "Software simulation for semantic hashing demonstration. No physical biological synthesis.",
    })


@app.route("/api/index", methods=["POST"])
def api_index_images():
    """Uploads and indexes one or more images into the semantic archive."""
    if not vit_service.is_ready:
        return jsonify({
            "error": "Vision Transformer extraction is currently unavailable.",
            "code": "MODEL_UNAVAILABLE",
            "details": vit_service.error_reason or "PyTorch native libraries blocked by Windows Application Control.",
            "hint": "Non-model features (viewing existing archive, DNA inspection, and pipeline) are active. To run inference, launch using an approved virtual environment.",
        }), 503

    files = request.files.getlist("images")
    if not files or all(not f.filename for f in files):
        return jsonify({"error": "Please select at least one valid image file to index."}), 400

    added = []
    skipped = []
    errors = []

    for file in files:
        if not file or not file.filename:
            continue

        filename = file.filename
        if not storage_service.is_allowed_file(filename):
            skipped.append(f"{filename}: Unsupported format (JPG, PNG, WEBP, BMP allowed).")
            continue

        try:
            raw_bytes = file.read()
            if len(raw_bytes) > MAX_IMAGE_FILE_SIZE:
                skipped.append(f"{filename}: File exceeds 10 MB limit.")
                continue

            # Verify image integrity via PIL
            image = Image.open(io.BytesIO(raw_bytes))
            image.verify()
            image = Image.open(io.BytesIO(raw_bytes))

            # Deduplication check
            sha_hash = storage_service.compute_sha256(raw_bytes)
            if storage_service.is_duplicate(sha_hash):
                skipped.append(f"{filename}: Already indexed in the archive (duplicate content).")
                continue

            # Extract 768-D Vision Transformer embedding
            embedding = vit_service.extract_embedding(image)

            # Generate simulated DNA sequence
            dna_sequence = encode_embedding_to_dna(embedding)

            # Persist to disk and update database
            item = storage_service.add_item(
                name=os.path.basename(filename),
                image_bytes=raw_bytes,
                image=image,
                embedding=embedding,
                dna_sequence=dna_sequence,
            )

            added.append({
                "id": item["id"],
                "name": item["name"],
                "url": item["url"],
                "file_size": item["file_size"],
                "dna_preview": item["dna_preview"],
                "dna_length": item["dna_length"],
                "gc_content": item["gc_content"],
            })

        except Exception as exc:
            logger.error("Error processing file %s: %s", filename, exc, exc_info=True)
            errors.append(f"{filename}: {str(exc)}")

    total_indexed = storage_service.get_count()

    if not added and (skipped or errors):
        return jsonify({
            "error": "No new images were indexed.",
            "skipped": skipped,
            "errors": errors,
            "indexed": total_indexed,
        }), 400

    return jsonify({
        "message": f"Successfully indexed {len(added)} image(s).",
        "added": added,
        "skipped": skipped,
        "errors": errors,
        "indexed": total_indexed,
    })


@app.route("/api/index-samples", methods=["POST"])
def api_index_samples():
    """One-click demo endpoint: indexes all pre-generated sample images."""
    if not vit_service.is_ready:
        return jsonify({
            "error": "Vision Transformer extraction is currently unavailable.",
            "code": "MODEL_UNAVAILABLE",
            "details": vit_service.error_reason or "PyTorch native libraries blocked by Windows Application Control.",
        }), 503

    if not os.path.exists(SAMPLE_IMAGES_DIR):
        return jsonify({"error": "Sample images directory does not exist."}), 404

    sample_files = sorted([f for f in os.listdir(SAMPLE_IMAGES_DIR) if f.endswith(".png")])
    if not sample_files:
        return jsonify({"error": "No sample images found."}), 404

    added = []
    skipped = []

    for s_name in sample_files:
        s_path = os.path.join(SAMPLE_IMAGES_DIR, s_name)
        try:
            with open(s_path, "rb") as f:
                raw_bytes = f.read()

            sha_hash = storage_service.compute_sha256(raw_bytes)
            if storage_service.is_duplicate(sha_hash):
                skipped.append(s_name)
                continue

            image = Image.open(io.BytesIO(raw_bytes))
            embedding = vit_service.extract_embedding(image)
            dna_sequence = encode_embedding_to_dna(embedding)

            item = storage_service.add_item(
                name=s_name.replace("_", " ").replace(".png", "").title(),
                image_bytes=raw_bytes,
                image=image,
                embedding=embedding,
                dna_sequence=dna_sequence,
            )

            added.append({
                "id": item["id"],
                "name": item["name"],
                "url": item["url"],
                "dna_preview": item["dna_preview"],
                "dna_length": item["dna_length"],
                "gc_content": item["gc_content"],
            })
        except Exception as exc:
            logger.error("Error indexing sample %s: %s", s_name, exc)

    return jsonify({
        "message": f"Sample dataset ready: {len(added)} indexed, {len(skipped)} already in archive.",
        "added": added,
        "skipped": skipped,
        "indexed": storage_service.get_count(),
    })


@app.route("/api/search", methods=["POST"])
def api_search():
    """Executes semantic image search using Vision Transformer + k-NN graph refinement."""
    if not vit_service.is_ready:
        return jsonify({
            "error": "Vision Transformer search is currently unavailable.",
            "code": "MODEL_UNAVAILABLE",
            "details": vit_service.error_reason or "PyTorch native libraries blocked by Windows Application Control.",
        }), 503

    file = request.files.get("query_image")
    if not file or not file.filename:
        return jsonify({"error": "Please provide a query image."}), 400

    indexed_items = storage_service.get_all_items()
    if not indexed_items:
        return jsonify({
            "error": "The archive index is currently empty. Please index images before searching.",
            "code": "EMPTY_INDEX",
        }), 400

    # Validate Top-K parameter
    top_k_str = request.form.get("top_k", "5")
    try:
        top_k = max(1, min(10, int(top_k_str)))
    except ValueError:
        top_k = 5

    try:
        raw_bytes = file.read()
        image = Image.open(io.BytesIO(raw_bytes))
        image.verify()
        image = Image.open(io.BytesIO(raw_bytes))

        # 1. Extract query embedding using ViT
        query_embedding = vit_service.extract_embedding(image)

        # 2. Generate simulated query DNA fingerprint
        query_dna = encode_embedding_to_dna(query_embedding)
        query_dna_stats = get_dna_statistics(query_dna)

        # 3. Perform cosine similarity + k-NN graph message passing refinement
        search_result = search_similar_images(
            query_embedding=query_embedding,
            indexed_items=indexed_items,
            top_k=top_k,
        )

        return jsonify({
            "message": f"Search complete: found {len(search_result['results'])} nearest matches.",
            "results": search_result["results"],
            "latency_ms": search_result["latency_ms"],
            "total_indexed": search_result["total_indexed"],
            "refinement_method": search_result["refinement_method"],
            "disclaimer": search_result["disclaimer"],
            "query_dna_preview": format_dna_preview(query_dna, 100),
            "query_dna_length": len(query_dna),
            "query_dna_stats": query_dna_stats,
        })

    except Exception as exc:
        logger.error("Failed to execute search: %s", exc, exc_info=True)
        return jsonify({"error": f"Failed to process query image: {str(exc)}"}), 400


@app.route("/api/clear", methods=["POST"])
def api_clear_index():
    """Clears all indexed images and metadata from the archive."""
    storage_service.clear()
    return jsonify({
        "message": "Archive index successfully cleared.",
        "indexed": 0,
    })


# ==========================================
# APPLICATION STARTUP
# ==========================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info("Starting Helix Semantic Archive on http://127.0.0.1:%d", port)
    app.run(host="127.0.0.1", port=port, debug=False)
