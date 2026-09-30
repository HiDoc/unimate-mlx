import unittest

import numpy as np

from unimate_mlx.skeleton import (
    adjacency_matrix,
    graph_distances,
    normalized_laplacian_eigenvectors,
)


class SkeletonGraphTests(unittest.TestCase):
    def test_adjacency_and_saturated_shortest_paths(self):
        parents = [-1, 0, 1, 2, 3, 4, 5]
        adjacency = adjacency_matrix(parents)
        self.assertEqual(adjacency.dtype, np.bool_)
        self.assertTrue(np.array_equal(adjacency, adjacency.T))
        self.assertEqual(int(adjacency.sum()), 12)
        distances = graph_distances(parents)
        self.assertEqual(distances.dtype, np.int16)
        self.assertEqual(distances[0, 4], 4)
        self.assertEqual(distances[0, 5], 5)
        self.assertEqual(distances[0, 6], 5)
        self.assertTrue(np.array_equal(distances, distances.T))

    def test_reference_normalized_laplacian_eigenfeatures(self):
        parents = [-1, 0, 0]
        vectors, values = normalized_laplacian_eigenvectors(parents, max_freqs=4)
        adjacency = adjacency_matrix(parents).astype(np.float64)
        degree = adjacency.sum(axis=1)
        inv = 1 / np.sqrt(degree)
        laplacian = np.eye(3) - inv[:, None] * adjacency * inv[None, :]
        self.assertEqual(vectors.shape, (3, 4))
        self.assertEqual(values.shape, (4,))
        self.assertTrue(np.allclose(np.linalg.norm(vectors[:, :2], axis=0), 1.0))
        self.assertTrue(np.allclose(vectors[:, 2:], 0.0))
        self.assertTrue(np.allclose(values, [1.0, 2.0, 0.0, 0.0]))
        self.assertTrue(np.allclose(laplacian @ vectors[:, :2], vectors[:, :2] * values[:2]))

    def test_accepts_self_parent_root_and_rejects_non_tree(self):
        self.assertTrue(np.array_equal(adjacency_matrix([0, 0]), [[False, True], [True, False]]))
        with self.assertRaises(ValueError):
            adjacency_matrix([-1, 2, 1])
        with self.assertRaises(ValueError):
            adjacency_matrix([-1, -1])


if __name__ == "__main__":
    unittest.main()
