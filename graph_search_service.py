"""Semantic Image Search and Graph Similarity Refinement Service.

Computes cosine similarity between query and candidate image embeddings,
and applies k-nearest-neighbor (k-NN) graph message passing diffusion.

SCIENTIFIC DISTINCTION & ARCHITECTURAL NOTE:
The similarity refinement implemented in this milestone is a heuristic,
training-free k-NN graph diffusion baseline (random walk / Laplacian smoothing).
It does NOT use a trained Graph Neural Network (such as GCN, GAT, or GraphSAGE).
The interface and modular design allow a trained inductive GNN model to be dropped
in during Milestone 2 without altering the external API.
"""

import time
import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from config import DEFAULT_KNN_NEIGHBORS, DEFAULT_GRAPH_ALPHA

logger = logging.getLogger(__name__)


def knn_graph_diffusion(
    embeddings: np.ndarray,
    k: int = DEFAULT_KNN_NEIGHBORS,
    alpha: float = DEFAULT_GRAPH_ALPHA,
) -> np.ndarray:
    """Applies a 1-step k-NN graph diffusion (message passing) across embedding vectors.

    Mathematical formulation:
    1. Compute pairwise cosine similarity matrix S in [-1, 1] for all items (indexed + query).
    2. Zero out diagonal self-similarity (to select genuine neighbors).
    3. Construct k-NN adjacency matrix A where A[i, j] = max(S[i, j], 0) for j in top-k neighbors of i.
    4. Symmetrize adjacency: A_sym = max(A, A.T).
    5. Add identity self-loop: A_tilde = A_sym + I.
    6. Row-normalize to yield Markov transition matrix: P = D^(-1) A_tilde.
    7. Diffuse embeddings: X_refined = (1 - alpha) * X + alpha * (P @ X).
    8. L2-normalize each row of X_refined.

    Args:
        embeddings: 2D numpy array of shape (N, D).
        k: Number of nearest neighbors to connect per node.
        alpha: Diffusion weight in [0, 1]. alpha=0 retains pure ViT features; alpha=1 relies purely on neighbors.

    Returns:
        2D numpy array of shape (N, D) containing refined, unit-normalized embeddings.
    """
    embeddings = np.asarray(embeddings, dtype=np.float32)
    n_samples = len(embeddings)

    if n_samples <= 1:
        return embeddings

    # Ensure k does not exceed available candidate neighbors
    effective_k = min(k, n_samples - 1)

    # 1. Pairwise cosine similarity
    sim_matrix = cosine_similarity(embeddings)

    # 2. Exclude self-loop temporarily to find distinct neighbors
    sim_no_self = sim_matrix.copy()
    np.fill_diagonal(sim_no_self, -1.0)

    # 3. Build directed k-NN adjacency
    adjacency = np.zeros((n_samples, n_samples), dtype=np.float32)
    for i in range(n_samples):
        neighbor_indices = np.argsort(sim_no_self[i])[-effective_k:]
        adjacency[i, neighbor_indices] = np.maximum(sim_no_self[i, neighbor_indices], 0.0)

    # 4. Symmetrize graph connections
    adjacency = np.maximum(adjacency, adjacency.T)

    # 5. Add identity self-loop
    adjacency += np.eye(n_samples, dtype=np.float32)

    # 6. Row normalization (stochastic transition matrix)
    row_sums = adjacency.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-8)
    transition_matrix = adjacency / row_sums

    # 7. Linear diffusion (message passing)
    refined = (1.0 - alpha) * embeddings + alpha * (transition_matrix @ embeddings)

    # 8. Re-normalize to unit hypersphere
    norms = np.linalg.norm(refined, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return (refined / norms).astype(np.float32)


def search_similar_images(
    query_embedding: np.ndarray,
    indexed_items: List[Dict[str, Any]],
    top_k: int = 5,
    k_neighbors: int = DEFAULT_KNN_NEIGHBORS,
    alpha: float = DEFAULT_GRAPH_ALPHA,
) -> Dict[str, Any]:
    """Performs semantic image search over indexed embeddings with k-NN graph refinement.

    Args:
        query_embedding: 1D numpy array of shape (768,) for the query image.
        indexed_items: List of indexed image dictionaries containing 'embedding' and metadata.
        top_k: Number of ranked results to return.
        k_neighbors: Number of neighbors for graph message passing.
        alpha: Graph diffusion weighting factor.

    Returns:
        Dictionary containing ranked results, search latency in ms, and baseline comparison.
    """
    start_time = time.perf_counter()

    if not indexed_items:
        return {
            "results": [],
            "latency_ms": 0.0,
            "total_indexed": 0,
            "message": "Archive index is empty. Please index images first.",
        }

    n_indexed = len(indexed_items)
    indexed_embeddings = np.vstack([item["embedding"] for item in indexed_items])

    # 1. Base Cosine Similarity (pure ViT, prior to graph refinement)
    base_scores = cosine_similarity(query_embedding.reshape(1, -1), indexed_embeddings)[0]

    # 2. Joint Graph Refinement (indexed items + query node)
    joint_embeddings = np.vstack([indexed_embeddings, query_embedding.reshape(1, -1)])
    refined_embeddings = knn_graph_diffusion(joint_embeddings, k=k_neighbors, alpha=alpha)

    refined_query = refined_embeddings[-1].reshape(1, -1)
    refined_candidates = refined_embeddings[:-1]
    refined_scores = cosine_similarity(refined_query, refined_candidates)[0]

    # 3. Rank by refined similarity
    effective_top_k = max(1, min(top_k, n_indexed))
    ranked_indices = np.argsort(refined_scores)[::-1][:effective_top_k]

    results = []
    for rank, idx in enumerate(ranked_indices, start=1):
        item = indexed_items[int(idx)]
        base_score = float(base_scores[idx])
        refined_score = float(refined_scores[idx])
        score_delta = round(refined_score - base_score, 4)

        results.append({
            "rank": rank,
            "id": item["id"],
            "name": item["name"],
            "url": item.get("url") or f"/uploads/{item['filename']}",
            "filename": item["filename"],
            "base_score": round(max(0.0, min(1.0, base_score)), 4),
            "score": round(max(0.0, min(1.0, refined_score)), 4),
            "score_delta": score_delta,
            "dna_preview": item.get("dna_preview") or (item["dna"][:72] + "…"),
            "dna_length": len(item.get("dna", "")),
            "gc_content": item.get("gc_content", 50.0),
        })

    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

    return {
        "results": results,
        "latency_ms": elapsed_ms,
        "total_indexed": n_indexed,
        "refinement_method": "k-NN Graph Message Passing (Diffusion Baseline)",
        "disclaimer": "Scores represent visual feature proximity in ViT latent space, not guaranteed biological or semantic identity.",
    }
