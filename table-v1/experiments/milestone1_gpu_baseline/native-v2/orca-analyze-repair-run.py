"""Analyze existing traces and prediction arrays; no retraining or physics changes."""
import argparse,bisect,json
from pathlib import Path
import numpy as np

def samples(path):
    with path.open() as f:return [r for r in (json.loads(line) for line in f) if r['type']=='sample']
def pose_error(a,b):
    position=float(np.linalg.norm(np.array(a['base_target']['position'])-b['base_target']['position']));qa=np.array(a['base_target']['quaternion']);qb=np.array(b['base_target']['quaternion']);angle=float(2*np.arccos(np.clip(abs(qa@qb)/(np.linalg.norm(qa)*np.linalg.norm(qb)),0,1)));joint=float(np.max(np.abs(np.array(a['joint_position_targets'])-b['joint_position_targets'])))
    return position,angle,joint
def summarize(rows,reference):
    times=[r['time_s'] for r in reference];differences=[];first=None;streak=0;streak_start=None
    for row in rows:
        i=bisect.bisect_right(times,row['time_s']+1e-9)-1
        if i<0:continue
        expected=reference[i];p,q,j=pose_error(row['submitted_action'],expected['submitted_action']);different=p>.005 or q>.05 or j>.05
        if different:
            if not streak:streak_start={'time_s':row['time_s'],'oracle_phase':expected['phase'],'policy_phase':row['phase'],'base_target_difference_m':p,'orientation_difference_rad':q,'maximum_joint_difference_rad':j}
            streak+=1
        else:streak=0
        if first is None and streak>=3:first=streak_start
        differences.append((p,q,j))
    rates=[]
    for a,b in zip(rows,rows[1:]):
        dt=b['time_s']-a['time_s']
        if dt<=0:raise ValueError('Non-monotonic trace')
        p,q,j=pose_error(a['submitted_action'],b['submitted_action']);rates.append((p/dt,q/dt,j/dt))
    last=rows[-1];obj=last['state_after']['objects'][0];expected=reference[-1]
    def distribution(x):
        x=np.asarray(x);return {'mean':float(np.mean(x)),'p95':float(np.quantile(x,.95)),'maximum':float(np.max(x))} if len(x) else None
    return {'samples':len(rows),'phase_sample_counts':{p:sum(r['phase']==p for r in rows) for p in sorted({r['phase'] for r in rows})},'first_sustained_target_divergence':first,'divergence_definition':'3 sampled commands exceeding 5mm base position or 0.05rad orientation/joint difference from oracle at the same time; diagnostic only, not task gate','same_clock_base_target_difference_m':distribution([d[0] for d in differences]),'target_translation_speed_m_s':distribution([d[0] for d in rates]),'target_rotation_speed_rad_s':distribution([d[1] for d in rates]),'maximum_joint_target_speed_rad_s':distribution([d[2] for d in rates]),'final_phase':last['phase'],'final_hand_load_n':last['task']['hand_load_n'],'final_opposing_support':last['task']['opposing_support'],'final_inside_tray':last['task']['inside_tray'],'final_landed':last['task']['landed'],'final_object_pose':obj,'final_submitted_action':last['submitted_action'],'oracle_final_submitted_action':expected['submitted_action'],'final_achieved_joints':last['state_after']['joint_positions'],'oracle_final_achieved_joints':expected['state_after']['joint_positions']}

def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();reference=samples(a.run/'oracle.ndjson');report={'scope':'offline diagnosis of saved data, not another task qualification','oracle':summarize(reference,reference),'recipes':{}}
    for name in ['one-action','ten-actions','temporal-ensemble']:
        path=a.run/(name+'.ndjson')
        if path.exists():
            rows=samples(path)
            if rows:report['recipes'][name]=summarize(rows,reference)
    predpath=a.run/'teacher-forced.npz'
    if predpath.exists():
        with np.load(predpath) as data:
            time=data['time']-data['time'][0];pred=data['predicted'];truth=data['submitted'];phases=[reference[max(0,bisect.bisect_right([r['time_s'] for r in reference],t+1e-9)-1)]['phase'] for t in time];report['teacher_forced_phase_errors']={}
            for phase in sorted(set(phases)):
                mask=np.array([x==phase for x in phases]);errors=np.linalg.norm(pred[mask,17:20]-truth[mask,17:20],axis=1);report['teacher_forced_phase_errors'][phase]={'frames':int(mask.sum()),'mean_base_target_error_m':float(errors.mean()),'p95_base_target_error_m':float(np.quantile(errors,.95)),'mean_max_joint_error_rad':float(np.max(np.abs(pred[mask,:17]-truth[mask,:17]),axis=1).mean())}
    with a.output.open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
    print(a.output)
if __name__=='__main__':main()
