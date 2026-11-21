"""Read an extracted Wei export, including the actual LeRobot parquet rows."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds

JOINTS = (
    ["right_wrist"]
    + [
        f"right_{finger}-{joint}"
        for finger in "prmi"
        for joint in ("abd", "mcp", "pip")
    ]
    + ["right_t-cmc", "right_t-abd", "right_t-mcp", "right_t-pip"]
)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def names():
    def pose(body):
        return [f"{body}.position.{k}" for k in "xyz"] + [
            f"{body}.quaternion.{k}" for k in "xyzw"
        ]

    return {
        "observation.state": [f"{j}.position" for j in JOINTS]
        + [f"{j}.velocity" for j in JOINTS]
        + pose("base")
        + [
            f"base.{v}.{k}"
            for v in ("linear_velocity", "angular_velocity")
            for k in "xyz"
        ],
        "observation.environment_state": pose("block") + pose("tray"),
        "action": [f"{j}.position_target" for j in JOINTS] + pose("base_target"),
    }


def inspect(root, *, for_rollout=False):
    metadata = read(root / "wei-export.json")
    manifest = read(root / "wei-manifest.json")
    listed = {r["path"]: r for r in manifest["files"]}
    actual = {
        p.relative_to(root).as_posix(): p
        for p in root.rglob("*")
        if p.is_file() and p != root / "wei-manifest.json"
    }
    if actual.keys() != listed.keys() or any(
        sha(p) != listed[k]["sha256"] for k, p in actual.items()
    ):
        raise ValueError(
            "Export does not match wei-manifest.json. Extract a clean copy."
        )
    info = read(root / "meta/info.json")
    if metadata["schema"] != "wei.dataset.export.v1" or not metadata["stateOnly"]:
        raise ValueError("Expected a state-only Wei Studio dataset ZIP")
    if metadata["fps"] != info["fps"] or not 1 <= info["fps"] <= 240:
        raise ValueError("Invalid dataset sampling rate")
    # ACT derives input/output widths from the dataset, including two-hand vectors.
    for key in ("observation.state", "observation.environment_state", "action"):
        feature = metadata["features"][key]
        stored = info["features"][key]
        if any(feature[k] != stored[k] for k in ("names", "shape", "dtype")):
            raise ValueError(f"Export and LeRobot disagree about {key}")
        if (
            feature["dtype"] != "float32"
            or len(feature["shape"]) != 1
            or not feature["names"]
            or feature["shape"] != [len(feature["names"])]
            or len(set(feature["names"])) != len(feature["names"])
        ):
            raise ValueError(f"Expected named float32 vector for {key}")
    if not for_rollout:
        return metadata
    if (
        metadata["robotType"] != "orca-v2-right"
        or metadata["robot"]["jointIds"] != JOINTS
    ):
        raise ValueError("This rollout requires the right Orca v2 hand")
    for key, expected in names().items():
        for feature in (metadata["features"][key], info["features"][key]):
            if (
                feature["shape"] != [len(expected)]
                or feature["names"] != expected
                or feature["dtype"] != "float32"
            ):
                raise ValueError(f"Incompatible ordering, type or width for {key}")
    for e in metadata["episodeList"]:
        scene = e.get("scene") or metadata["scene"]
        if (
            scene.get("taskVersion", scene.get("id")) != "table-v1"
            or (
                "taskVersion" not in scene
                and (
                    (e.get("evaluator") or {}).get("id") != "orca-table-v1-pick-place"
                    or str(scene.get("revision")) != "1"
                )
            )
            or scene.get("taskId", "orca.block-pick-place") != "orca.block-pick-place"
            or (scene.get("lessonObject") or {}).get("id", "block") != "block"
        ):
            raise ValueError("This runner supports the Table-v1 block task")
    return metadata


def initial_conditions(root, metadata, index):
    episode = next(
        (r for r in metadata["episodeList"] if r["episodeIndex"] == index), None
    )
    if episode is None:
        raise ValueError(f"Unknown episode index {index}")
    data = ds.dataset(str(root / "data"), format="parquet")
    table = data.to_table(
        columns=["observation.state", "observation.environment_state"],
        filter=(ds.field("episode_index") == index) & (ds.field("frame_index") == 0),
    )
    if len(table) != 1:
        raise ValueError("Expected one initial row for the selected episode")
    row = table.to_pylist()[0]
    state = row["observation.state"]
    env = row["observation.environment_state"]
    scene = episode.get("scene") or metadata["scene"]
    if "taskVersion" not in scene:
        # Studio worker exports identify the evaluated scene by profile and revision.
        scene = {
            **scene,
            "trayCenter": env[7:9],
            "trayFloorTop": env[9],
            "blockSize": [0.045] * 3,
        }
    if not np.allclose(
        env[7:14], [*scene["trayCenter"], scene["trayFloorTop"], 0, 0, 0, 1], atol=1e-5
    ):
        raise ValueError("Tray pose disagrees with the selected episode")
    obj = {
        "id": "block",
        "position": env[:3],
        "quaternion": env[3:7],
        "linear_velocity": [0] * 3,
        "angular_velocity": [0] * 3,
    }
    initial = {
        "joint_positions": state[:17],
        "joint_velocities": state[17:34],
        "base_pose": {"position": state[34:37], "quaternion": state[37:41]},
        "base_twist_world": {"linear": state[41:44], "angular": state[44:47]},
        "objects": [obj],
    }
    geometry = {
        "block_mass_kg": 0.08,
        "block_position": env[:3],
        "block_size": scene["blockSize"],
        "tray_center": scene["trayCenter"],
        "tray_floor_top": scene["trayFloorTop"],
        "tray_internal_width": 0.12,
        "tray_wall_height": 0.02,
        "table_position": [-0.045, 0.43, 0.4025],
        "table_size": [1.05, 0.6, 0.03],
    }
    return initial, geometry
