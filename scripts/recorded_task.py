"""Convert native Orca takes to a local ACT dataset for a single named task."""

import argparse
from bisect import bisect_right
import hashlib
import tempfile
import json
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def pose(item):
    return [*item["position"], *item["quaternion"]]


def channels(state, following, sides, objects, tray=None):
    observation, action, environment = [], [], []
    for side in sides:
        hand = state if side == "right" else state["hands"][side]
        command = following if side == "right" else following["hands"][side]
        achieved = hand["observation"]["after"]
        observation.extend([*achieved["joint_positions"], *pose(achieved["base_pose"])])
        applied = command["action"]
        action.extend(
            [*applied["joint_position_targets"], *pose(applied["base_target"])]
        )
    visible = {o["id"]: o for o in [state["object"], *state.get("custom_objects", [])]}
    if tray is not None:
        visible["tray"] = tray
    for obj in objects:
        environment.extend(pose(visible[obj]))
    return {
        "observation.state": observation,
        "action": action,
        "observation.environment_state": environment,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("takes", nargs="+", type=Path)
    parser.add_argument("--task", required=True)
    parser.add_argument("--object", action="append", required=True, dest="objects")
    parser.add_argument("--two-hands", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()
    if (
        args.output.exists()
        or not 1 <= args.fps <= 120
        or len(set(args.objects)) != len(args.objects)
    ):
        parser.error("Use a new output directory, distinct object IDs and 1–120 fps")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as staging:
        root = Path(staging) / "dataset"
        build(args, root)
        root.rename(args.output)


def build(args, root):
    sides = ["right", "left"] if args.two_hands else ["right"]
    dataset, contract, sources = None, None, []
    for path in args.takes:
        raw = path.read_bytes()
        take = json.loads(raw)
        ready = take["episode"]["initial_conditions"]["model_and_scene"]
        tray = None
        if "tray" in args.objects:
            scene = ready["scene"]
            tray = {
                "position": [*scene["tray_center"], scene["tray_floor_top"]],
                "quaternion": [0, 0, 0, 1],
            }
        frames = [f["state"] for f in take["frames"]]
        times = [s["sim_time"] for s in frames]
        if (
            len(times) < 2
            or not np.isfinite(times).all()
            or min(np.diff(times)) < 0
            or max(np.diff(times)) > 2 / args.fps
        ):
            raise ValueError(
                f"{path}: missing or discontinuous simulator clock; choose a lower recording rate or a continuous take"
            )
        fields = []
        pins = []
        for side in sides:
            model = ready if side == "right" else ready["hands"]["left"]
            joints = model["joint_ids"]
            if len(set(joints)) != 17:
                raise ValueError(f"{path}: invalid {side} joint ordering")
            pins.append(model["model_sha256"])
            fields += [f"{side}.{j}" for j in joints] + [
                f"{side}.base.{c}" for c in ("x", "y", "z", "qx", "qy", "qz", "qw")
            ]
        env = [
            f"{obj}.{c}"
            for obj in args.objects
            for c in ("x", "y", "z", "qx", "qy", "qz", "qw")
        ]
        if dataset is None:
            contract = fields, pins
            features = {
                key: {"dtype": "float32", "shape": (len(names),), "names": names}
                for key, names in [
                    ("observation.state", fields),
                    ("observation.environment_state", env),
                    ("action", fields),
                ]
            }
            dataset = LeRobotDataset.create(
                repo_id="local/wei-recorded-task",
                root=root,
                fps=args.fps,
                robot_type="orca-" + "-".join(sides),
                features=features,
                use_videos=False,
            )
        if contract != (fields, pins):
            raise ValueError("Takes use different Orca models or joint orders")
        count = int((times[-1] - times[0]) * args.fps)
        if count < 2:
            raise ValueError(f"{path}: take is too short")
        for i in range(count):
            idx = bisect_right(times, times[0] + i / args.fps) - 1
            row = channels(frames[idx], frames[idx + 1], sides, args.objects, tray)
            arrays = {k: np.asarray(v, dtype=np.float32) for k, v in row.items()}
            if any(not np.isfinite(v).all() for v in arrays.values()):
                raise ValueError(f"{path}: nonfinite sample")
            dataset.add_frame({**arrays, "task": args.task})
        dataset.save_episode()
        sources.append(
            {
                "take": take["episode"]["id"],
                "sha256": hashlib.sha256(raw).hexdigest(),
                "frames": count,
            }
        )
    dataset.finalize()
    (root / "recording-sources.json").write_text(
        json.dumps(
            {
                "task": args.task,
                "fps": args.fps,
                "objects": args.objects,
                "sources": sources,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
