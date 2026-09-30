import numpy as np

from unimate_mlx.conditioning import denormalize_motion, prepare_condition


def test_prepares_normalized_root_local_tpose_and_parent_features():
    topology = {
        "parents": [-1, 0, 1],
        "tpos_first_frame": [[0, 2, 0], [1, 3, 0], [2, 4, 0]],
        "joint_graph_dists": [[0, 1, 2], [1, 0, 1], [2, 1, 0]],
        "joint_relations": [[0, 2, 4], [1, 0, 2], [4, 1, 5]],
        "joint_depths": [0, 1, 2],
        "spectral_feats": np.zeros((3, 8)),
    }
    stats = {"mixamo": {
        "mean_root": np.ones(12), "std_root": np.full(12, 2),
        "mean_local": np.zeros(12), "std_local": np.ones(12),
    }}
    result = prepare_condition(
        topology, stats, dataset_type="mixamo", caption_embedding=np.zeros(768),
        joint_name_embeddings=np.zeros((3, 768)), motion_length=4,
    )
    assert result["tpos_first_frame"].shape == (1, 61, 12)
    np.testing.assert_allclose(result["tpos_first_frame"][0, 0, :3], [-0.5, -0.5, -0.5])
    np.testing.assert_allclose(result["tpos_first_frame"][0, 1, :3], [1, 1, 0])
    np.testing.assert_allclose(result["tpos_first_frame_parents"][0, 2], result["tpos_first_frame"][0, 1])
    np.testing.assert_allclose(result["tpos_first_frame"][0, 1, 3:9], [1, 0, 0, 0, 1, 0])
    assert result["std"][0, 3, 0] == 1
    assert result["graph_dist"][0, 0, 2] == 2

    raw = np.zeros((1, 61, 12, 4), dtype=np.float32)
    exported = denormalize_motion(raw, result)
    assert exported.shape == (4, 3, 12)
    np.testing.assert_allclose(exported[0, 0], stats["mixamo"]["mean_root"])


def test_v2_condition_uses_71_joint_padding():
    topology = {
        "parents": [-1, 0],
        "tpos_first_frame": [[0, 0, 0], [0, 1, 0]],
        "joint_graph_dists": [[0, 1], [1, 0]],
        "joint_relations": [[0, 2], [1, 5]],
        "joint_depths": [0, 1],
        "spectral_feats": np.zeros((2, 8)),
    }
    stats = {"objaverse": {
        "mean_root": np.zeros(12), "std_root": np.ones(12),
        "mean_local": np.zeros(12), "std_local": np.ones(12),
    }}
    condition = prepare_condition(
        topology, stats, dataset_type="objaverse",
        caption_embedding=np.zeros(768), joint_name_embeddings=np.zeros((2, 768)),
        max_joints=71,
    )
    assert condition["graph_dist"].shape == (1, 71, 71)
    assert denormalize_motion(np.zeros((1, 71, 12, 3)), condition).shape == (3, 2, 12)
