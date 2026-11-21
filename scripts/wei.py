"""Train an ACT checkpoint or evaluate it with a live MuJoCo observation at each policy tick."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from studio_export import inspect, initial_conditions, sha


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def train(args, metadata):
    selected = args.episodes
    available = {e["episodeIndex"] for e in metadata["episodeList"]}
    if selected is None:
        selected = metadata["evaluation"]["passedEpisodes"]
    if (
        not selected
        or len(set(selected)) != len(selected)
        or not set(selected) <= available
    ):
        raise ValueError("Choose one or more distinct episode indices from the export.")
    cmd = [
        sys.executable,
        "-m",
        "lerobot.scripts.lerobot_train",
        f"--dataset.repo_id={metadata['repoId']}",
        f"--dataset.root={args.dataset}",
        f"--dataset.episodes={json.dumps(selected)}",
        "--policy.type=act",
        f"--policy.device={args.device}",
        "--policy.push_to_hub=false",
        "--policy.pretrained_backbone_weights=null",
        "--policy.chunk_size=30",
        "--policy.n_action_steps=1",
        "--policy.dim_model=128",
        "--policy.n_heads=4",
        "--policy.dim_feedforward=512",
        "--policy.n_encoder_layers=2",
        "--policy.use_vae=false",
        f"--batch_size={args.batch_size}",
        "--num_workers=0",
        f"--steps={args.steps}",
        f"--save_freq={args.steps}",
        f"--seed={args.seed}",
        "--env_eval_freq=0",
        "--trackio.enable=false",
        f"--output_dir={args.output / 'training'}",
    ]
    args.output.mkdir(parents=True, exist_ok=False)
    save(
        args.output / "training-input.json",
        {
            "manifest_sha256": sha(args.dataset / "wei-manifest.json"),
            "episodes": [
                e for e in metadata["episodeList"] if e["episodeIndex"] in selected
            ],
            "fps": metadata["fps"],
            "command": cmd,
        },
    )
    subprocess.run(cmd, check=True)
    print(
        "Checkpoint:",
        next(
            (args.output / "training/checkpoints" / f"{args.steps:06d}").glob(
                "pretrained_model"
            )
        ),
    )


def evaluate(args, metadata):
    from model import model_fingerprint
    from orca_lerobot_policy import ACTCheckpoint
    from simulation import rollout, json_default

    initial, scene = initial_conditions(args.dataset, metadata, args.episode)
    robot = metadata["robot"]
    actual = model_fingerprint()
    if actual != robot["modelManifest"]["xml_sha256"]:
        raise ValueError(
            "Install the pinned right-hand model; XML fingerprint differs from the export."
        )
    limits = np.column_stack([robot["jointLowerRad"], robot["jointUpperRad"]])
    if limits.shape != (17, 2) or not np.isfinite(limits).all():
        raise ValueError("Invalid joint bounds")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "pipeline_passed": False,
        "task_passed": False,
        "episode": args.episode,
        "checkpoint": str(args.checkpoint),
        "model_sha256": actual,
        "dataset_manifest_sha256": sha(args.dataset / "wei-manifest.json"),
        "fps": metadata["fps"],
        "seconds": args.seconds,
    }
    try:
        policy = ACTCheckpoint(args.checkpoint, args.device, action_steps=1)
        result = rollout(
            scene,
            initial,
            robot["baseGains"],
            policy,
            metadata["fps"],
            args.seconds,
            limits,
        )
        frames = result.pop("frames")
        (args.output / "trajectory.json").write_text(
            json.dumps(frames, default=json_default, allow_nan=False)
        )
        report.update(
            json.loads(json.dumps(result, default=json_default)),
            pipeline_passed=result["failure"] is None,
            task_passed=bool(result["passed"]),
        )
        if args.video:
            render(args.output, scene, frames)
    except Exception as exc:
        report["pipeline_passed"] = False
        report["error"] = str(exc)
        raise
    finally:
        save(args.output / "report.json", report)
    print(
        "Pipeline completed:",
        report["pipeline_passed"],
        " Task passed:",
        report["task_passed"],
    )
    if not report["pipeline_passed"]:
        raise SystemExit(2)


def render(output, scene, frames):
    import mujoco
    from simulation import task_model, IDS

    model, _ = task_model(scene, "tray-clearance-environment-3.5ms")
    model.geom_rgba[model.geom("replay_block_geom").id] = [1, 0.3, 0.03, 1]
    data = mujoco.MjData(model)
    addresses = [model.jnt_qposadr[model.joint(n).id] for n in IDS]
    base = model.jnt_qposadr[model.joint("superdex_base").id]
    block = model.jnt_qposadr[model.joint("replay_block_free").id]
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0, 0.205, 0.49]
    camera.distance = 0.65
    camera.azimuth = 45
    camera.elevation = -30
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        "640x480",
        "-r",
        "30",
        "-i",
        "-",
        "-an",
        "-c:v",
        "mpeg4",
        "-q:v",
        "3",
        "-pix_fmt",
        "yuv420p",
        str(output / "rollout.mp4"),
    ]
    with (
        mujoco.Renderer(model, height=480, width=640) as renderer,
        subprocess.Popen(cmd, stdin=subprocess.PIPE) as proc,
    ):
        for f in frames[::2]:
            data.qpos[addresses] = f["joints"]
            data.qpos[base : base + 7] = [
                *f["base_position"],
                *f["base_quaternion_wxyz"],
            ]
            data.qpos[block : block + 7] = [
                *f["object_position"],
                *f["object_quaternion_wxyz"],
            ]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            proc.stdin.write(renderer.render().tobytes())
        proc.stdin.close()
        if proc.wait():
            raise RuntimeError("ffmpeg failed to encode the rollout")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ["inspect", "train", "evaluate"]:
        c = sub.add_parser(name)
        c.add_argument("dataset", type=Path)
        if name != "inspect":
            c.add_argument("--output", type=Path, required=True)
            c.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
        if name == "train":
            c.add_argument("--episodes", nargs="+", type=int)
            c.add_argument("--steps", type=int, default=5000)
            c.add_argument("--seed", type=int, default=7)
            c.add_argument("--batch-size", type=int, default=4)
        if name == "evaluate":
            c.add_argument("--episode", type=int, default=0)
            c.add_argument("--checkpoint", type=Path, required=True)
            c.add_argument("--seconds", type=float, default=15)
            c.add_argument("--video", action="store_true")
    args = p.parse_args()
    args.dataset = args.dataset.resolve()
    meta = inspect(args.dataset)
    if args.command == "inspect":
        print(
            json.dumps(
                {
                    k: meta[k]
                    for k in ["repoId", "fps", "episodes", "frames", "evaluation"]
                },
                indent=2,
            )
        )
        return
    args.output = args.output.resolve()
    if args.command == "train":
        if min(args.steps, args.batch_size) < 1:
            p.error("steps and batch size must be positive")
        train(args, meta)
    else:
        if not 0 < args.seconds <= 90:
            p.error("seconds must be in (0,90]")
        evaluate(args, meta)


if __name__ == "__main__":
    main()
