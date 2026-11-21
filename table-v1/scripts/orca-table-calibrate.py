"""Independent synthetic table component tests; never human training examples or stand calibration edits."""
import argparse
import copy
import gzip
import json
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
from orca_table_task import digest,episode_contract,install_table_geometry,load_frozen_validator,scene_identity,validate_scene


def default(value):
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,np.ndarray):return value.tolist()
    raise TypeError(type(value).__name__)


class AllStep:
    def __init__(self,r,latest):self.r=r;self.latest=latest;self.main=None;self.last=-1.;self.metrics={'steps':0,'forearm_table_peak_n':0.,'hand_block_peak_n':0.,'grip_resistance_error_rad':0.,'final_settled_seconds':0.}
    def __enter__(self):
        self.initialize=self.r.initialize;self.forward=self.r.mujoco.mj_forward
        self.patches=[patch.object(self.r,'initialize',self.init),patch.object(self.r.mujoco,'mj_forward',self.observe)]
        for p in self.patches:p.start()
        return self
    def __exit__(self,*args):
        for p in reversed(self.patches):p.stop()
    def init(self,model,state):
        result=self.initialize(model,state);self.main,self.qp,_=result;self.model=model;self.base=model.body('right_mount').id;return result
    def observe(self,model,data,*args,**kwargs):
        result=self.forward(model,data,*args,**kwargs)
        if data is not self.main or data.time<=self.last:return result
        self.last=float(data.time);m=self.metrics;m['steps']+=1;load=0.
        for i,c in enumerate(data.contact):
            names=[model.geom(g).name or '' for g in [c.geom1,c.geom2]];bodies=[model.body(int(model.geom_bodyid[g])).name or '' for g in [c.geom1,c.geom2]]
            wrench=np.empty(6);self.r.mujoco.mj_contactForce(model,data,i,wrench);normal=abs(float(wrench[0]))
            if 'table' in names and any('ForeArm' in name for name in bodies):m['forearm_table_peak_n']=max(m['forearm_table_peak_n'],normal)
            if 'replay_block_geom' in names and any(name.startswith('right_') for name in bodies):load+=normal
        m['hand_block_peak_n']=max(m['hand_block_peak_n'],load)
        if load>.1 and .8<data.time<3.4:m['grip_resistance_error_rad']=max(m['grip_resistance_error_rad'],float(np.max(np.abs(data.qpos[self.qp]-self.latest['action'][:17]))))
        block=model.body('replay_block').id;twist=np.empty(6);self.r.mujoco.mj_objectVelocity(model,data,self.r.mujoco.mjtObj.mjOBJ_BODY,block,twist,0)
        rest=self.latest['rest'];settled=abs(data.xpos[block,2]-rest)<=.002 and np.linalg.norm(twist[3:])<.02 and np.linalg.norm(twist[:3])<.1
        m['final_settled_seconds']=m['final_settled_seconds']+model.opt.timestep if settled else 0.
        return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workspace',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();repo=Path(__file__).resolve().parents[1];r,profile,parent=load_frozen_validator(a.workspace,repo/'experiments/milestone1_gpu_baseline/native-v2')
    from orca_diagnostic_trace import DiagnosticTrace
    from orca_compressed_trace import CompressedTracePath
    sys.path.insert(0,str(repo/'experiments/superdex_feasibility'));from table_placements import TABLE_STARTS
    raw,source_sha=r.load(a.source);contract=episode_contract(raw,r);install_table_geometry(r);scene=raw['episode']['initial_conditions']['model_and_scene']['scene'];reference=json.loads((repo/'public/superdex/orca-table-reference.v1.json').read_text())
    compiled,configuration=r.task_model(scene,profile);real_factory=r.task_model
    def cached(scene_arg,profile_arg):
        validate_scene(scene_arg)
        keys=['task_version','pickup_support','table_position','table_size','block_size','block_mass_kg','tray_center','tray_floor_top','tray_internal_width','tray_wall_height']
        if any(scene_arg[k]!=scene[k] for k in keys) or profile_arg!=profile:raise ValueError('Cached model geometry/controller mismatch')
        return compiled,{**configuration,'scene_sha256':scene_identity(scene_arg)}
    r.task_model=cached;a.output.mkdir(parents=True,exist_ok=False)
    report={'schema':'orca.table.calibration.v1','scope':'simulation_only synthetic component qualification; excluded from human learning data',
        'source_geometry_recording_sha256':source_sha,'contract':contract,'profile':profile,'parent_robot_calibration_sha256':parent,
        'runner_sha256':digest(__file__),'adapter_sha256':digest(repo/'scripts/orca_table_task.py'),'validator_sha256':digest(a.workspace/'scripts/orca-superdex-validate.py'),
        'physics_dt':.0002,'penetration_ceiling_m':.002,'joint_limit_tolerance_rad':.02,'base_force_limit_n':80,'base_torque_limit_nm':8,
        'cases':[],'complete':False,'passed':False,'training_eligible':False,'stand_calibration_modified':False,'model_compilation_reused':True}
    def save():
        temporary=a.output/'report.tmp';temporary.write_text(json.dumps(report,default=default,indent=2,allow_nan=False));temporary.replace(a.output/'report.json')
    save();specs=[{'name':f'static-{i:02}','kind':'environment','position':list(point),'height':0.,'duration':1.5} for i,point in enumerate(TABLE_STARTS)]
    specs += [{'name':f'drop-{int(height*100):02}cm','kind':'environment','position':[-.065,.27],'height':height,'duration':1.6} for height in [.02,.05,.1]]
    specs += [{'name':'finger-resistance-opening','kind':'finger','position':[-.045,.24],'height':0.,'duration':5.8},
        {'name':'forearm-table','kind':'forearm','position':[-.045,.24],'height':0.,'duration':2.},
        {'name':'base-saturation','kind':'saturation','position':[-.045,.24],'height':0.,'duration':.0002}]
    for spec in specs:
        folder=a.output/spec['name'];folder.mkdir();results=[];frames=[];metrics=[]
        for trial in [1,2]:
            source=copy.deepcopy(raw);ep=source['episode'];initial=ep['initial_conditions'];local=initial['model_and_scene']['scene'];local['block_position'][:2]=spec['position']
            initial['sim_time']=0.;state=initial['achieved_before_first_sample'];state['joint_positions']=reference['open'].copy();state['joint_velocities']=[0.]*17
            state['base_pose']={'position':[.05,-.05,1.1],'quaternion':reference['initial_base']['quaternion']};state['base_twist_world']={'linear':[0.]*3,'angular':[0.]*3}
            block=next(o for o in state['objects'] if o['id']=='block');block.update(position=[*spec['position'],local['block_position'][2]+spec['height']],quaternion=[0,0,0,1],linear_velocity=[0.]*3,angular_velocity=[0.]*3)
            ep['command_application_events']=[];ep['discontinuities']=[];ep['id']='synthetic-table-calibration-'+spec['name'];source['frames']=[{'state':{'sim_time':spec['duration']}}]
            if spec['kind']=='finger':state['base_pose']=copy.deepcopy(reference['initial_base'])
            if spec['kind']=='forearm':
                state['base_pose']={'position':[0.,0.,.8],'quaternion':[0.,0.,0.,1.]};data,_,_=r.initialize(compiled,state)
                geom=next(g for g in range(compiled.ngeom) if compiled.geom_contype[g] and 'ForeArm' in compiled.body(int(compiled.geom_bodyid[g])).name)
                mesh=int(compiled.geom_dataid[geom]);vertices=compiled.mesh_vert[compiled.mesh_vertadr[mesh]:compiled.mesh_vertadr[mesh]+compiled.mesh_vertnum[mesh]]
                world=vertices@data.geom_xmat[geom].reshape(3,3).T+data.geom_xpos[geom];lowest=world[np.argmin(world[:,2])]
                top=local['table_position'][2]+local['table_size'][2]/2
                state['base_pose']['position']=(np.asarray(state['base_pose']['position'])+np.array([-.32,.55,top+.003])-lowest).tolist()
            base=copy.deepcopy(state['base_pose']);tick=[0];latest={'action':np.r_[reference['open'],base['position'],base['quaternion']],'rest':local['block_position'][2]}
            def commands(features):
                t=tick[0]/60;tick[0]+=1;q=np.array(reference['open']);position=np.array(base['position']);quaternion=np.array(base['quaternion'])
                smooth=lambda x:float(np.clip(x,0,1)**2*(3-2*np.clip(x,0,1)))
                if spec['kind']=='finger':
                    closed=smooth((t-.8)/2.2)*(1-smooth((t-3.4)/1.2));q=q*(1-closed)+np.array(reference['closed'])*closed;position[2]+=.08*smooth((t-4.6)/.5)
                elif spec['kind']=='forearm':position[2]+=-.008*smooth(t/.3)+.028*smooth((t-1.3)/.3)
                elif spec['kind']=='saturation':
                    position[0]+=1.;out=np.empty(4);r.mujoco.mju_mulQuat(out,np.array([np.cos(1.5),np.sin(1.5),0,0]),quaternion[[3,0,1,2]]);quaternion=out[[1,2,3,0]]
                latest['action']=np.r_[q,position,quaternion];return latest['action']
            trace=CompressedTracePath(folder/f'trial-{trial}.ndjson.gz')
            with AllStep(r,latest) as observer:
                with DiagnosticTrace(r,source,trace,policy=True,control_source='synthetic_calibration_commands'):
                    result=r.replay(source,profile,policy=commands,duration_override=spec['duration'])
            last=result['frames'][-1];physical=bool(result['gates']['finite'] and result['gates']['penetration'] and result['gates']['joint_limits'])
            extra=observer.metrics
            if spec['kind']=='environment':criterion=extra['final_settled_seconds']>=1 and abs(last['object_position'][2]-local['block_position'][2])<=.002
            elif spec['kind']=='finger':
                extra['opening_error_rad']=float(np.max(np.abs(np.array(last['joints'])-reference['open'])));criterion=extra['hand_block_peak_n']>.1 and extra['grip_resistance_error_rad']>.01 and extra['opening_error_rad']<=.02
            elif spec['kind']=='forearm':criterion=extra['forearm_table_peak_n']>.1
            else:
                extra.update(requested_wrench=last['requested_wrench'],applied_wrench=last['applied_wrench'])
                criterion=np.linalg.norm(last['requested_wrench'][:3])>80 and np.linalg.norm(last['requested_wrench'][3:])>8 and np.linalg.norm(last['applied_wrench'][:3])<=80+1e-9 and np.linalg.norm(last['applied_wrench'][3:])<=8+1e-9
            frames.append(result.pop('frames'));row={'passed':bool(physical and criterion),'physical_result':result,'measurements':extra,'trace_sha256':digest(trace.path)}
            path=folder/f'trial-{trial}.json';path.write_text(json.dumps(row,default=default,indent=2,allow_nan=False));results.append({'report':str(path),'sha256':digest(path),'passed':row['passed']});metrics.append(extra)
        comparable=lambda result:{k:v for k,v in result.items() if k!='wall_seconds'}
        first=json.loads((folder/'trial-1.json').read_text());second=json.loads((folder/'trial-2.json').read_text())
        repeat=frames[0]==frames[1] and metrics[0]==metrics[1] and comparable(first['physical_result'])==comparable(second['physical_result'])
        report['cases'].append({'specification':spec,'trials':results,'repeatable':repeat,'passed':all(x['passed'] for x in results) and repeat});save()
        print(spec['name'],'passed',report['cases'][-1]['passed'],'repeat',repeat,'peak mm',first['physical_result']['max_penetration_m']*1000,'metrics',first['measurements'],flush=True)
    report.update(complete=True,passed=all(case['passed'] for case in report['cases']));save()


if __name__=='__main__':main()
