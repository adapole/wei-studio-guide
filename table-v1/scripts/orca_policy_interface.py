"""Closed-loop policy boundary. Inputs and outputs use simulation Z-up / XYZW."""
import numpy as np


def observation(joints, velocities, base, linear, angular, block, tray):
    state = np.asarray([*joints, *velocities, *base, *linear, *angular], dtype=np.float32)
    environment = np.asarray([*block, *tray], dtype=np.float32)
    if state.shape != (47,) or environment.shape != (14,) or not np.isfinite(np.r_[state, environment]).all():
        raise ValueError('Invalid simulator observation')
    return {'observation.state': state, 'observation.environment_state': environment}


def action(value, ranges):
    value = np.asarray(value, dtype=float).reshape(-1)
    ranges = np.asarray(ranges, dtype=float)
    if value.shape != (24,) or ranges.shape != (17, 2) or not np.isfinite(value).all():
        raise ValueError('Policy must return 24 finite values')
    if np.any(value[:17] < ranges[:, 0]) or np.any(value[:17] > ranges[:, 1]):
        raise ValueError('Policy joint target exceeds robot limits')
    if np.linalg.norm(value[17:20]) > 2:
        raise ValueError('Policy base target exceeds local workspace')
    norm = np.linalg.norm(value[20:24])
    if norm < 1e-6:
        raise ValueError('Policy returned a degenerate base quaternion')
    return {'joint_position_targets': value[:17].tolist(),
            'base_target': {'position': value[17:20].tolist(), 'quaternion': (value[20:24] / norm).tolist()}}
