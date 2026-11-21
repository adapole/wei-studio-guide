"""Pinned upstream right Orca v2 scene, with Wei simulation-only adaptations."""

import hashlib
from pathlib import Path
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "models/orca_sim/src/orca_sim"


def source_scene():
    return ASSETS / "scenes/v2/scene_right_cube_orientation.xml"


def model_spec(free_base=False, gravity=False):
    spec = mujoco.MjSpec.from_file(str(source_scene()))
    spec.option.timestep = 0.0002
    # The sample's small in-palm cube is irrelevant to controller calibration.
    for geom in spec.geoms:
        if geom.name.startswith("task_cube"):
            geom.contype = geom.conaffinity = 0
    if not gravity:
        spec.option.gravity = [0, 0, 0]
    if free_base:
        base = spec.body("right_mount")
        base.add_freejoint(name="studio_base")
        base.mass = 0.1
        base.inertia = [0.001, 0.001, 0.001]
    return spec


def quaternion_error(desired, actual):
    inverse = actual * np.array([1.0, -1.0, -1.0, -1.0])
    relative = np.empty(4)
    mujoco.mju_mulQuat(relative, desired, inverse)
    if relative[0] < 0:
        relative *= -1
    result = np.empty(3)
    mujoco.mju_quat2Vel(result, relative, 1.0)
    return result


def model_fingerprint():
    digest = hashlib.sha256()
    folder = ASSETS
    for path in sorted(folder.rglob("*.xml")) + sorted(folder.rglob("*.mjcf")):
        digest.update(path.relative_to(folder).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()
