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
    chunk_size: int = 500,
) -> csr_matrix:
    """
    filter_radius is in mm. Centroids are expected in metres.
    Converts filter_radius to metres internally.
    """
    from scipy.spatial import cKDTree
    from scipy.sparse import csr_matrix as csr
    import numpy as np

    filter_radius_m = filter_radius / 1000.0  # mm → m
    n = len(centroids)
    tree = cKDTree(centroids)
    pairs = tree.query_pairs(filter_radius_m, output_type='ndarray')
    print(f"    KD-tree: {len(pairs):,} pairs found")

    if len(pairs) > 0:
        i_idx = pairs[:, 0]
        j_idx = pairs[:, 1]
        dists = np.linalg.norm(centroids[i_idx] - centroids[j_idx], axis=1)
        weights = np.maximum(0.0, filter_radius_m - dists)

        # Build COO arrays directly — no Python loop needed
        rows = np.concatenate([i_idx, j_idx, np.arange(n)])
        cols = np.concatenate([j_idx, i_idx, np.arange(n)])
        data = np.concatenate([weights, weights,
                                np.full(n, filter_radius_m)])
    else:
        rows = np.arange(n)
        cols = np.arange(n)
        data = np.full(n, filter_radius_m)

    from scipy.sparse import coo_matrix
    H = coo_matrix((data, (rows, cols)), shape=(n, n))
    return H.tocsr()


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