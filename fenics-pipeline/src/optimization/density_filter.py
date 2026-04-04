# src/optimization/density_filter.py
#
# Density filter for SIMP topology optimization.
#
# Implements the linear density filter from Bourdin (2001):
#   rho_filtered[e] = sum(w[e,f] * rho[f]) / sum(w[e,f])
#   where w[e,f] = max(0, r - dist(centroid_e, centroid_f))
#
# The filter radius r controls minimum feature size.
# r should be 2-3x the target_element_size from Stage 2.
# Too small: checkerboard survives. Too large: over-smoothed, loses detail.
#
# Built as a sparse weight matrix so the filter is a single matrix-vector
# multiply per iteration — cheap relative to the FEA solve.

from __future__ import annotations
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from typing import Optional


def compute_element_centroids(
    points: np.ndarray,
    tets: np.ndarray,
) -> np.ndarray:
    """
    Compute centroid of each tetrahedral element.
    points: (n_nodes, 3) node coordinates
    tets:   (n_elements, 4) node indices (0-indexed)
    Returns: (n_elements, 3) centroids
    """
    return points[tets].mean(axis=1)


def build_filter_matrix(
    centroids: np.ndarray,
    filter_radius: float,
    chunk_size: int = 2000,
) -> csr_matrix:
    """
    Build the sparse filter weight matrix H where H[e, f] = max(0, r - d(e,f)).

    Uses chunked distance computation to avoid O(n²) memory on large meshes.
    For n=100k elements this matrix is ~100k x 100k but very sparse at
    typical filter radii — expect < 0.1% fill.

    chunk_size: number of elements to process per batch.
                Reduce if hitting memory limits during filter construction.
    """
    n = len(centroids)
    H = lil_matrix((n, n), dtype=np.float64)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = centroids[start:end]  # (chunk, 3)

        # Pairwise distances: chunk vs all centroids
        diff = centroids[np.newaxis, :, :] - chunk[:, np.newaxis, :]  # (chunk, n, 3)
        dist = np.linalg.norm(diff, axis=2)  # (chunk, n)

        weights = np.maximum(0.0, filter_radius - dist)  # (chunk, n)

        for local_i, global_i in enumerate(range(start, end)):
            nonzero = np.where(weights[local_i] > 0)[0]
            H[global_i, nonzero] = weights[local_i, nonzero]

    H_csr = H.tocsr()

    # Row sums for normalization — precompute once
    return H_csr


def apply_filter(
    H: csr_matrix,
    rho: np.ndarray,
) -> np.ndarray:
    """
    Apply density filter: rho_filtered = H @ rho / (H @ ones).
    Both numerator and denominator are O(nnz) sparse operations.
    """
    Hs = np.array(H.sum(axis=1)).ravel()   # row sums
    return (H @ rho) / (Hs + 1e-16)


def apply_sensitivity_filter(
    H: csr_matrix,
    rho: np.ndarray,
    dc: np.ndarray,
) -> np.ndarray:
    """
    Filter sensitivities (alternative to density filter — not used by default
    but kept here because some SIMP implementations prefer it).
    dc_filtered[e] = (H @ (rho * dc))[e] / (rho[e] * Hs[e])
    """
    Hs = np.array(H.sum(axis=1)).ravel()
    return (H @ (rho * dc)) / (rho * Hs + 1e-16)