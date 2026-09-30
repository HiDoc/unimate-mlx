import json
import struct

from unimate_mlx.asset_facing import infer_front_hip_pair


def test_infers_named_front_hip_pair_only_from_skin_joints(tmp_path):
    document = {
        "nodes": [
            {"name": "base"}, {"name": "FR_hip"}, {"name": "FL_hip"},
            {"name": "front_right_hip"}, {"name": "front_left_hip"},
        ],
        "skins": [{"joints": [0, 1, 2]}],
    }
    payload = json.dumps(document).encode()
    payload += b" " * (-len(payload) % 4)
    path = tmp_path / "robot.glb"
    path.write_bytes(
        struct.pack("<4sII", b"glTF", 2, 20 + len(payload))
        + struct.pack("<I4s", len(payload), b"JSON") + payload
    )
    assert infer_front_hip_pair(path) == ("FR_hip", "FL_hip")
    assert infer_front_hip_pair(tmp_path / "robot.fbx") is None

