import numpy as np
import pytest

from unimate_mlx.sequence import chain_motion_clips


def test_chain_preserves_duration_endpoints_and_inputs():
    first = np.zeros((60, 3, 12), dtype=np.float32)
    second = np.ones((60, 3, 12), dtype=np.float32)
    first_before = first.copy()
    second_before = second.copy()

    chained = chain_motion_clips([first, second])

    assert chained.shape == (120, 3, 12)
    np.testing.assert_array_equal(chained[0], first[0])
    np.testing.assert_array_equal(chained[-1], second[-1])
    np.testing.assert_array_equal(first, first_before)
    np.testing.assert_array_equal(second, second_before)


def test_smooth_transition_reduces_boundary_jump():
    first = np.zeros((60, 2, 12), dtype=np.float32)
    second = np.full((60, 2, 12), 10, dtype=np.float32)

    plain = chain_motion_clips([first, second], blend_frames=0)
    smoothed = chain_motion_clips([first, second], blend_frames=8)

    assert np.max(np.abs(np.diff(smoothed[:, 0, 0]))) < np.max(
        np.abs(np.diff(plain[:, 0, 0]))
    )
    np.testing.assert_array_equal(smoothed[:52], first[:52])
    np.testing.assert_array_equal(smoothed[68:], second[8:])


@pytest.mark.parametrize(
    "clips, blend_frames, message",
    [
        ([], 8, "at least one"),
        ([np.zeros((2, 3))], 8, "shape"),
        ([np.zeros((60, 2, 12)), np.zeros((60, 3, 12))], 1, "matching"),
        ([np.zeros((10, 2, 12))], -1, "non-negative"),
        ([np.zeros((60, 2, 12))], 31, r"2 \* blend_frames"),
    ],
)
def test_rejects_invalid_clip_inputs(clips, blend_frames, message):
    with pytest.raises(ValueError, match=message):
        chain_motion_clips(clips, blend_frames=blend_frames)


def test_rejects_non_finite_features():
    clip = np.zeros((60, 2, 12), dtype=np.float32)
    clip[3, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        chain_motion_clips([clip])
