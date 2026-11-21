"""Pinned ORCA simulation utilities. No hardware interface is imported."""
from pathlib import Path
import hashlib
import os
import numpy as np
import mujoco
from orca_sim.versions import resolve_scene_path

ROOT = Path(os.environ.get('WEI_WORKSPACE_ROOT', Path(__file__).resolve().parents[1])).resolve()
BASE_GAINS = dict(position=400., velocity=60., rotation=15., angular_velocity=3., force_limit=80., torque_limit=8.)


def source_scene():
    # Resolve the same pinned XML without allocating an unused upstream model/data.
    return resolve_scene_path('scene_right_cube_orientation.xml', version='v2')


def model_spec(free_base=False, gravity=False):
    spec = mujoco.MjSpec.from_file(str(source_scene()))
    spec.option.timestep = .0002
    # The sample's small in-palm cube is irrelevant to controller calibration.
    for geom in spec.geoms:
        if geom.name.startswith("task_cube"):
            geom.contype = geom.conaffinity = 0
    if not gravity:
        spec.option.gravity = [0, 0, 0]
    if free_base:
        base = spec.body("right_mount")
        base.add_freejoint(name="studio_base")
        base.mass = .1
        base.inertia = [.001, .001, .001]
    return spec


def joint_addresses(model):
    ids = model.actuator_trnid[:, 0]
    if len(set(ids)) != 17 or model.nu != 17:
        raise ValueError("Expected 17 unique ORCA joint actuators")
    if not np.all(model.actuator_forcelimited):
        raise ValueError("Unbounded actuator force")
    return model.jnt_qposadr[ids], model.jnt_dofadr[ids]


def quaternion_error(desired, actual):
    inverse = actual * np.array([1., -1., -1., -1.])
    relative = np.empty(4)
    mujoco.mju_mulQuat(relative, desired, inverse)
    if relative[0] < 0:
        relative *= -1
    result = np.empty(3)
    mujoco.mju_quat2Vel(result, relative, 1.)
    return result


class BaseController:
    def __init__(self, model, profile='pose-pd-v1'):
        self.body = model.body("right_mount").id
        self.gains = BASE_GAINS.copy()
        if profile not in {'pose-pd-v1','velocity-feedforward-v2'}:
            raise ValueError('Unknown base controller profile')
        self.profile = profile
        self.target_velocity = np.zeros(6)
        self.previous_target = None

    def set_target(self, position, quaternion, dt):
        if self.previous_target is not None and self.profile == 'velocity-feedforward-v2':
            old_position, old_quaternion = self.previous_target
            self.target_velocity[:3] = quaternion_error(quaternion, old_quaternion) / dt
            self.target_velocity[3:] = (position - old_position) / dt
            for offset, limit in [(0, 2.), (3, .6)]:
                speed = np.linalg.norm(self.target_velocity[offset:offset+3])
                if speed > limit:
                    self.target_velocity[offset:offset+3] *= limit / speed
        self.previous_target = (position.copy(), quaternion.copy())

    def apply(self, model, data, position, quaternion):
        velocity = np.empty(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, self.body, velocity, 0)
        g = self.gains
        force = g['position'] * (position - data.xpos[self.body]) + g['velocity'] * (self.target_velocity[3:] - velocity[3:])
        # Gravity compensation applies to the complete base subtree.
        force -= model.body_subtreemass[self.body] * model.opt.gravity
        torque = g['rotation'] * quaternion_error(quaternion, data.xquat[self.body]) + g['angular_velocity'] * (self.target_velocity[:3] - velocity[:3])
        requested = np.concatenate([force, torque])
        applied = requested.copy()
        for offset, limit in [(0, g['force_limit']), (3, g['torque_limit'])]:
            length = np.linalg.norm(applied[offset:offset+3])
            if length > limit:
                applied[offset:offset+3] *= limit / length
        data.xfrc_applied[self.body] = applied
        return dict(profile=self.profile,target_velocity=self.target_velocity.tolist(),desired_position=position.tolist(),desired_quaternion=quaternion.tolist(),twist_order='angular_then_linear_world',requested_wrench=requested.tolist(), applied_wrench=applied.tolist(),
                    saturated=bool(np.any(np.abs(requested-applied) > 1e-10)),
                    position=data.xpos[self.body].tolist(), quaternion=data.xquat[self.body].tolist(),
                    twist=velocity.tolist(), gains=g)


def model_fingerprint():
    digest = hashlib.sha256()
    folder = ROOT / 'artifacts/orca_sim/src/orca_sim'
    for path in sorted(folder.rglob('*.xml')) + sorted(folder.rglob('*.mjcf')):
        digest.update(path.relative_to(folder).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def asset_fingerprint():
    digest = hashlib.sha256()
    folder = ROOT / 'artifacts/orca_sim/src/orca_sim/models/v2'
    # Match the original Windows case-insensitive ordering on Linux/WSL too.
    for path in sorted(folder.rglob('*'), key=lambda item: item.relative_to(folder).as_posix().lower()):
        if path.is_file() and path.suffix.lower() in {'.stl', '.obj'}:
            digest.update(path.relative_to(folder).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()
