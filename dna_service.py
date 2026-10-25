"""Simulated DNA Fingerprint Generation Service.

Converts normalized Vision Transformer image embeddings (768-D) into deterministic
simulated DNA-compatible nucleotide sequences (A, T, G, C).

DISCLAIMER & RESEARCH NOTE:
This module implements an in-silico software simulation for semantic hashing and
concept-of-operations demonstration. It does NOT perform biological synthesis,
macromolecular biochemical synthesis, enzymatic oligo assembly, or physical DNA sequencing.
"""

from typing import Dict, Any, Tuple
import numpy as np

from config import NUCLEOTIDE_MAP, SIMULATED_DNA_LENGTH, EMBEDDING_DIM

VALID_NUCLEOTIDES = set("ATGC")


def encode_embedding_to_dna(embedding: np.ndarray) -> str:
    """Deterministically maps a 768-D normalized embedding vector to a 3,072-nt simulated DNA sequence.

    Mapping procedure:
    1. Clip and quantize each float dimension [-1.0, 1.0] into an 8-bit unsigned integer [0, 255]:
       q = round((embedding + 1.0) * 127.5) clamped to [0, 255].
    2. Convert each uint8 value to an 8-character binary string (e.g., 200 -> '11001000').
    3. Pair adjacent bits (2 bits per nucleotide) and map according to canonical 2-bit DNA encoding:
       '00' -> 'A' (Adenine)
       '01' -> 'C' (Cytosine)
       '10' -> 'G' (Guanine)
       '11' -> 'T' (Thymine)
    4. Concatenate to yield a sequence of 768 * 4 = 3,072 nucleotides.

    Args:
        embedding: 1D numpy array of shape (768,) representing the ViT embedding.

    Returns:
        String of exactly 3,072 characters composed strictly of 'A', 'T', 'G', and 'C'.
    """
    if embedding is None or len(embedding) != EMBEDDING_DIM:
        raise ValueError(f"Embedding must have shape ({EMBEDDING_DIM},), got {getattr(embedding, 'shape', None)}")

    # 1. Deterministic 8-bit uniform quantization
    quantized = np.clip(np.round((embedding + 1.0) * 127.5), 0, 255).astype(np.uint8)

    # 2. Binary expansion
    binary_stream = "".join(f"{int(val):08b}" for val in quantized)

    # 3. 2-bit to nucleotide transcription
    dna_chars = [NUCLEOTIDE_MAP[binary_stream[i : i + 2]] for i in range(0, len(binary_stream), 2)]
    sequence = "".join(dna_chars)

    assert len(sequence) == SIMULATED_DNA_LENGTH, (
        f"Generated sequence length mismatch: {len(sequence)} != {SIMULATED_DNA_LENGTH}"
    )
    return sequence


def validate_dna_sequence(sequence: str) -> Tuple[bool, str]:
    """Validates that a generated DNA sequence strictly adheres to nucleotide specifications.

    Args:
        sequence: The DNA sequence string.

    Returns:
        Tuple of (is_valid: bool, reason: str).
    """
    if not isinstance(sequence, str):
        return False, "Sequence must be a string."

    if len(sequence) == 0:
        return False, "Sequence cannot be empty."

    invalid_chars = set(sequence) - VALID_NUCLEOTIDES
    if invalid_chars:
        return False, f"Invalid nucleotide characters detected: {invalid_chars}"

    return True, "Sequence is valid."


def get_dna_statistics(sequence: str) -> Dict[str, Any]:
    """Calculates nucleotide composition frequencies and GC content percentage.

    Args:
        sequence: The DNA sequence string.

    Returns:
        Dictionary with nucleotide counts, percentages, and overall GC content.
    """
    total = len(sequence)
    if total == 0:
        return {
            "total_nucleotides": 0,
            "counts": {"A": 0, "C": 0, "G": 0, "T": 0},
            "percentages": {"A": 0.0, "C": 0.0, "G": 0.0, "T": 0.0},
            "gc_content": 0.0,
            "is_valid": False,
        }

    counts = {
        "A": sequence.count("A"),
        "C": sequence.count("C"),
        "G": sequence.count("G"),
        "T": sequence.count("T"),
    }

    percentages = {base: round((count / total) * 100, 2) for base, count in counts.items()}
    gc_count = counts["G"] + counts["C"]
    gc_content = round((gc_count / total) * 100, 2)
    is_valid, _ = validate_dna_sequence(sequence)

    return {
        "total_nucleotides": total,
        "counts": counts,
        "percentages": percentages,
        "gc_content": gc_content,
        "is_valid": is_valid,
    }


def format_dna_preview(sequence: str, max_chars: int = 72) -> str:
    """Returns a shortened preview string formatted with an ellipsis if truncated."""
    if len(sequence) <= max_chars:
        return sequence
    return sequence[:max_chars] + "…"
