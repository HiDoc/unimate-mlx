"""NumPy preprocessing for parent-index skeleton trees.

Graph distances and spectral features follow
``unimate.utils.topology_utils`` in the official UniMate repository. This
module deliberately leaves the learned graph-bias relation IDs to the caller;
their convention is documented below because end-effector and T-pose token
connections are model-specific additions to ordinary parent/child relations.
"""

from __future__ import annotations

from collections import deque
from typing import Sequence

import numpy as np


def _parents_array(parents: Sequence[int] | np.ndarray) -> np.ndarray:
    raw = np.asarray(parents)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError("parents must be a nonempty one-dimensional array")
    if not np.issubdtype(raw.dtype, np.integer):
        raise ValueError("parent indices must be integers")
    result = raw.astype(np.int64, copy=True)
    n = result.size
    roots = []
    for joint, parent in enumerate(result):
        if parent == joint:  # Official topology utilities accept self-parent roots.
            result[joint] = -1
            roots.append(joint)
        elif parent == -1:
            roots.append(joint)
        elif parent < 0 or parent >= n:
            raise ValueError(f"invalid parent index parents[{joint}]={parent}")
    if len(roots) != 1:
        raise ValueError(f"a parent-index tree must have exactly one root; got {len(roots)}")
    # Require a connected acyclic tree; a parent cycle otherwise has no root.
    root = roots[0]
    seen = {root}
    for start in range(n):
        node = start
        chain = set()
        while node not in seen:
            if node in chain:
                raise ValueError("parent indices contain a cycle")
            chain.add(node)
            parent = result[node]
            if parent == -1:
                break
            node = int(parent)
        seen.update(chain)
    if any(result[j] != -1 and int(result[j]) not in seen for j in range(n)):
        raise ValueError("parent indices do not form a connected tree")
    # Every chain must resolve to the sole root.
    for start in range(n):
        node = start
        chain = set()
        while result[node] != -1:
            if node in chain:
                raise ValueError("parent indices contain a cycle")
            chain.add(node)
            node = int(result[node])
    return result


def adjacency_matrix(parents: Sequence[int] | np.ndarray) -> np.ndarray:
    """Return a symmetric boolean ``(J, J)`` adjacency matrix without loops."""
    parents = _parents_array(parents)
    adjacency = np.zeros((parents.size, parents.size), dtype=bool)
    for child, parent in enumerate(parents):
        if parent >= 0:
            adjacency[child, parent] = True
            adjacency[parent, child] = True
    return adjacency


def graph_distances(
    parents: Sequence[int] | np.ndarray, max_path_len: int = 5
) -> np.ndarray:
    """Return UniMate-style pairwise shortest distances, saturated at 5 by default.

    This matches ``compute_edge_relations_and_distances``: ``dist[i, j]`` is
    the undirected parent/child hop count, clipped to ``max_path_len`` and
    stored as ``int16``.
    """
    if (
        not isinstance(max_path_len, int)
        or isinstance(max_path_len, bool)
        or not 0 <= max_path_len <= np.iinfo(np.int16).max
    ):
        raise ValueError("max_path_len must be an integer in the int16 range")
    adjacency = adjacency_matrix(parents)
    n = adjacency.shape[0]
    distances = np.full((n, n), max_path_len, dtype=np.int16)
    for source in range(n):
        row = np.full(n, np.iinfo(np.int16).max, dtype=np.int32)
        row[source] = 0
        queue = deque([source])
        while queue:
            current = queue.popleft()
            if row[current] >= max_path_len:
                continue
            for neighbor in np.flatnonzero(adjacency[current]):
                if row[neighbor] > row[current] + 1:
                    row[neighbor] = row[current] + 1
                    queue.append(int(neighbor))
        distances[source] = np.minimum(row, max_path_len).astype(np.int16)
    return distances


def normalized_laplacian_eigenvectors(
    parents: Sequence[int] | np.ndarray, max_freqs: int = 8
) -> tuple[np.ndarray, np.ndarray]:
    """Return UniMate's symmetric normalized Laplacian eigenfeatures.

    The trivial lowest eigenvector is omitted, columns are L2-normalized, and
    outputs are zero-padded to ``max_freqs``. Returns ``(eigenvectors,
    eigenvalues)`` with shapes ``(J, max_freqs)`` and ``(max_freqs,)``. As in
    the reference implementation, eigenvector signs are not canonicalized;
    sign-invariant SignNet/RoPE processing is needed for stable use in models.
    """
    if not isinstance(max_freqs, int) or isinstance(max_freqs, bool) or max_freqs < 0:
        raise ValueError("max_freqs must be a nonnegative integer")
    adjacency = adjacency_matrix(parents).astype(np.float64)
    n = adjacency.shape[0]
    degree = adjacency.sum(axis=1)
    inv_sqrt_degree = np.zeros_like(degree)
    nonzero = degree > 0
    inv_sqrt_degree[nonzero] = 1.0 / np.sqrt(degree[nonzero])
    laplacian = np.diag(nonzero.astype(np.float64)) - (
        inv_sqrt_degree[:, None] * adjacency * inv_sqrt_degree[None, :]
    )
    all_values, all_vectors = np.linalg.eigh(laplacian)
    count = min(max(n - 1, 0), max_freqs)
    values = np.maximum(all_values[1 : 1 + count], 0.0)
    vectors = all_vectors[:, 1 : 1 + count].copy()
    for column in range(count):
        norm = np.linalg.norm(vectors[:, column])
        if norm > 1e-12:
            vectors[:, column] /= norm
    if count < max_freqs:
        vectors = np.pad(vectors, ((0, 0), (0, max_freqs - count)))
        values = np.pad(values, (0, max_freqs - count))
    return vectors.astype(np.float32), values.astype(np.float32)


# Official ``joint_relations`` IDs, for model code that needs to construct the
# bias matrix: 0=self, 1=parent, 2=child, 3=sibling, 4=no relation,
# 5=end effector, 6=T-pose-token connection. The official graph attention bias
# embedding is configured for six edge types (IDs 0..5); ID 6 belongs to the
# special T-pose token path. For exact parity, its per-joint precedence (e.g.
# a leaf diagonal is ID 5 rather than ID 0) must be preserved by that caller.
