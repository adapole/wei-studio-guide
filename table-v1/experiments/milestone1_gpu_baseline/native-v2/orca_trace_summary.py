"""Compact diagnostics shared by scripts and future Evaluate jobs."""
import json
import numpy as np

def summarize_trace(path,joint_ids):
    position=[];angles=[];qerrors=[];overshoot=np.zeros(len(joint_ids));first=None;torques=[];force_sat=torque_sat=0;samples=0;contact_peak=0.;contact_pairs={}
    with path.open() as stream:
        for line in stream:
            row=json.loads(line)
            if row['type']!='sample':continue
            state=row['state_after'];action=row['submitted_action'];command=row['controller_output'];samples+=1
            position.append(np.linalg.norm(np.array(state['base_pose']['position'])-action['base_target']['position']))
            qa=np.array(state['base_pose']['quaternion']);qb=np.array(action['base_target']['quaternion']);angles.append(2*np.arccos(np.clip(abs(qa@qb)/(np.linalg.norm(qa)*np.linalg.norm(qb)),0,1)))
            qerrors.append(np.abs(np.array(state['joint_positions'])-action['joint_position_targets']))
            current=np.array(row['joint_limit_overshoot_rad']);overshoot=np.maximum(overshoot,current)
            if first is None and np.any(current>.02):first={'time_s':row['after_time_s'],'phase':row['phase'],'joints':[joint_ids[i] for i in np.flatnonzero(current>.02)]}
            torques.append(np.max(np.abs(command['joint_torque_applied'])));force_sat+=command['base_saturated'][0];torque_sat+=command['base_saturated'][1]
            for contact in row['contacts']:
                key=' / '.join(contact['bodies']);contact_pairs[key]=max(contact_pairs.get(key,0.),contact['penetration_m']);contact_peak=max(contact_peak,contact['normal_force_n'])
    def stats(values):
        x=np.array(values);return {'median':float(np.median(x)),'p95':float(np.quantile(x,.95)),'maximum':float(np.max(x))} if len(x) else None
    return {'scope':'sampled diagnostic metrics; authoritative acceptance uses all physics steps','samples':samples,'base_position_error_m':stats(position),'base_rotation_error_rad':stats(angles),'joint_target_error_rad':{name:stats([x[i] for x in qerrors]) for i,name in enumerate(joint_ids)},'achieved_joint_limit_overshoot_rad':dict(zip(joint_ids,overshoot.tolist())),'first_achieved_limit_excess_over_0_02_rad':first,'maximum_applied_joint_torque_nm':max(torques,default=0.),'base_force_saturated_samples':int(force_sat),'base_torque_saturated_samples':int(torque_sat),'peak_sampled_normal_contact_force_n':contact_peak,'contact_pair_penetration_m':contact_pairs}
