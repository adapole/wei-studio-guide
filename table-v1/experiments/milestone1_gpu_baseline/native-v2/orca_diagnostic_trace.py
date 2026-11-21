"""Read-only hooks around the pinned replay. Persist before/after state and faults.

No model, timestep, controller output or physics state is written by the observer.
Hooks are process-local and restored on exit; do not use concurrently in one process.
"""
import bisect,copy,json,math
import numpy as np
from unittest.mock import patch
from orca_task_phases import TaskPhases

class DiagnosticTrace:
    def __init__(self,replay,source,path,policy=False,control_source=None):
        self.r=replay;self.source=source;self.path=path;self.policy=policy;self.control_source=control_source or ('policy' if policy else 'recorded_commands');self.phases=TaskPhases();self.frames=0
        self.start=source['episode']['initial_conditions']['sim_time'];self.ready=source['episode']['initial_conditions']['model_and_scene'];self.scene=self.ready['scene']
        self.events=[e for e in source['episode']['command_application_events'] if e['first_applied_sim_time']>=self.start-1e-9];self.times=[e['first_applied_sim_time'] for e in self.events]
        self.active=copy.deepcopy(source['episode']['initial_conditions']['active_action']);self.raw=None;self.next_sample=0.;self.last_command=None;self.policy_index=0;self.pending=None;self.main=None;self.gravity_bias=None

    def emit(self,value):
        def finite(v):
            if isinstance(v,np.ndarray):return finite(v.tolist())
            if isinstance(v,np.generic):return finite(v.item())
            if isinstance(v,float) and not math.isfinite(v):return None
            if isinstance(v,dict):return {k:finite(x) for k,x in v.items()}
            if isinstance(v,(list,tuple)):return [finite(x) for x in v]
            return v
        self.file.write(json.dumps(finite(value),allow_nan=False)+'\n');self.file.flush()

    def __enter__(self):
        self.file=self.path.open('x',encoding='utf-8')
        self.emit({'type':'header','schema':'orca.diagnostic.trace.v1','state_source':'mujoco_simulation','frame':'right-handed-z-up','quaternion_order':'xyzw','units':'metres/radians/seconds/newtons/newton-metres','nonfinite_encoding':'null; original fault retained','observation_semantics':'state_before immediately before observed physics step; state_after after the same step and forward update','control_source':self.control_source,'source_episode_id':self.source['episode']['id']})
        self.original_init=self.r.initialize;self.original_step=self.r.mujoco.mj_step;self.original_forward=self.r.mujoco.mj_forward;self.original_action=self.r.policy_action
        self.patches=[patch.object(self.r,'initialize',self.initialize),patch.object(self.r,'policy_action',self.action),patch.object(self.r.mujoco,'mj_step',self.step),patch.object(self.r.mujoco,'mj_forward',self.forward)]
        for p in self.patches:p.start()
        return self

    def __exit__(self,kind,error,tb):
        for p in reversed(self.patches):p.stop()
        try:
            if error:self.emit({'type':'fault','time_s':float(self.main.time) if self.main is not None else None,'error':f'{kind.__name__}: {error}','raw_action':self.raw})
            self.emit({'type':'footer','samples':self.frames,'completed_without_exception':error is None,'phases':self.phases.report()})
        finally:
            self.file.close();self.main=None;self.model=None
        return False

    def initialize(self,model,state):
        result=self.original_init(model,state);self.main,self.qp,self.dv=result;self.model=model
        self.base=model.body('right_mount').id;self.block=model.body('replay_block').id;self.block_geom=model.geom('replay_block_geom').id;self.wrist=int(model.jnt_bodyid[model.joint('right_wrist').id])
        self.jacp=np.zeros((3,model.nv));self.jacr=np.zeros((3,model.nv));self.ranges=np.array([model.jnt_range[model.joint(n).id] for n in self.r.IDS]);self.hand=set()
        for i in range(model.nbody):
            ancestor=i
            while ancestor>0 and ancestor!=self.base:ancestor=int(model.body_parentid[ancestor])
            if ancestor==self.base:self.hand.add(i)
        return result

    def action(self,value,ranges):
        self.raw=np.asarray(value,dtype=float).reshape(-1).tolist()
        result=self.original_action(value,ranges);self.active=copy.deepcopy(result);self.policy_index+=1
        return result

    def pose(self,data,body):return {'position':data.xpos[body].tolist(),'quaternion':data.xquat[body][[1,2,3,0]].tolist()}

    def state(self,data):
        m=self.model;self.r.mujoco.mj_jac(m,data,self.jacp,self.jacr,data.xpos[self.base],self.base)
        twist=np.empty(6);self.r.mujoco.mj_objectVelocity(m,data,self.r.mujoco.mjtObj.mjOBJ_BODY,self.block,twist,0)
        return {'joint_positions':data.qpos[self.qp].tolist(),'joint_velocities':data.qvel[self.dv].tolist(),'base_pose':self.pose(data,self.base),'base_twist_world':{'linear':(self.jacp@data.qvel).tolist(),'angular':(self.jacr@data.qvel).tolist()},'wrist_pose':self.pose(data,self.wrist),'wrist_body':m.body(self.wrist).name,'objects':[{'id':'block',**self.pose(data,self.block),'linear_velocity':twist[3:].tolist(),'angular_velocity':twist[:3].tolist()}]}

    def step(self,model,data,*args,**kwargs):
        if data is self.main:
            if not self.policy:
                i=bisect.bisect_right(self.times,self.start+float(data.time)+1e-9)-1
                if i>=0:self.active=copy.deepcopy(self.events[i])
                identity=i
            else:identity=self.policy_index
            if data.time+1e-9>=self.next_sample or identity!=self.last_command:
                before=self.state(data);targets=np.asarray(self.active['joint_position_targets']);p,q=self.r.pose(self.active['base_target']);g=self.ready['base_gains'];up=-model.body_subtreemass[self.base]*model.opt.gravity
                linear=np.array(before['base_twist_world']['linear']);angular=np.array(before['base_twist_world']['angular'])
                force=g['position']*(p-data.xpos[self.base])-g['velocity']*linear+up
                torque=g['rotation']*self.r.quaternion_error(q,data.xquat[self.base])-g['angular_velocity']*angular+np.cross(data.subtree_com[self.base]-data.xpos[self.base],up)
                requested=np.r_[force,torque];applied=requested.copy();saturated=[]
                for offset,limit in [(0,g['force_limit']),(3,g['torque_limit'])]:
                    length=np.linalg.norm(applied[offset:offset+3]);saturated.append(bool(length>limit))
                    if length>limit:applied[offset:offset+3]*=limit/length
                raw_pd=2*(targets-data.qpos[self.qp])-.02*data.qvel[self.dv];bias=self.gravity_bias[self.dv] if self.gravity_bias is not None else np.zeros(17)
                raw_tau=np.clip(raw_pd,-1,1)+bias
                self.pending={'type':'sample','time_s':float(data.time),'source_sim_time':self.start+float(data.time),'command_id':identity,'raw_policy_action':self.raw if self.policy else None,'submitted_action':{'joint_position_targets':targets.tolist(),'base_target':self.active['base_target']},'state_before':before,'controller_output':{'requested_base_wrench':requested.tolist(),'applied_base_wrench':applied.tolist(),'base_saturated':saturated,'joint_torque_applied':data.qfrc_applied[self.dv].tolist(),'joint_torque_saturated':(np.abs(raw_tau)>1).tolist(),'joint_pd_saturated':(np.abs(raw_pd)>1).tolist(),'gains':g,'finger_kp':2,'finger_kd':.02,'joint_torque_limit_nm':1}}
                self.next_sample=float(data.time)+1/60;self.last_command=identity
        return self.original_step(model,data,*args,**kwargs)

    def forward(self,model,data,*args,**kwargs):
        result=self.original_forward(model,data,*args,**kwargs)
        if self.main is not None and model is self.model:
            if data is not self.main:self.gravity_bias=data.qfrc_bias.copy()
            elif self.pending is not None:
                row=self.pending;self.pending=None;row['after_time_s']=float(data.time);row['state_after']=self.state(data)
                row['joint_limit_overshoot_rad']=np.maximum(np.maximum(self.ranges[:,0]-data.qpos[self.qp],data.qpos[self.qp]-self.ranges[:,1]),0).tolist()
                contacts=[];forces={};tray=np.zeros(3);external=np.zeros(3)
                for i,c in enumerate(data.contact):
                    wrench=np.empty(6);self.r.mujoco.mj_contactForce(model,data,i,wrench);world=np.array(c.frame).reshape(3,3).T@wrench[:3]
                    contacts.append({'bodies':[model.body(int(model.geom_bodyid[g])).name for g in [c.geom1,c.geom2]],'geoms':[model.geom(g).name for g in [c.geom1,c.geom2]],'penetration_m':max(0.,-float(c.dist)),'normal_force_n':float(wrench[0])})
                    if self.block_geom not in [c.geom1,c.geom2]:continue
                    other=c.geom2 if c.geom1==self.block_geom else c.geom1;body=int(model.geom_bodyid[other]);name=model.geom(other).name or ''
                    if c.geom1==self.block_geom:world=-world
                    if body in self.hand:forces[body]=forces.get(body,np.zeros(3))+world
                    else:
                        external+=world
                        if name.startswith('tray_'):tray+=world
                loads=[f for f in forces.values() if np.linalg.norm(f)>.05];opposed=any(float(a@b)/(np.linalg.norm(a)*np.linalg.norm(b))<-.3 for i,a in enumerate(loads) for b in loads[i+1:]);load=sum(np.linalg.norm(f) for f in forces.values());upward=sum((f[2] for f in forces.values()),0.)
                pos=data.xpos[self.block];extent=np.abs(data.xmat[self.block].reshape(3,3))@(np.array(self.scene['block_size'])/2);obj=row['state_after']['objects'][0]
                lifted=self.r.supported_lift(pos[2],self.scene['block_position'][2],opposed,upward,np.linalg.norm(external))
                inside=np.all(np.abs(pos[:2]-self.scene['tray_center'])+extent[:2]<=self.scene['tray_internal_width']/2+.001)
                landed=bool(inside and abs(pos[2]-extent[2]-self.scene['tray_floor_top'])<=self.r.ENVELOPE and tray[2]>.5 and load<.05 and np.linalg.norm(obj['linear_velocity'])<.02 and np.linalg.norm(obj['angular_velocity'])<.1)
                row['contacts']=contacts;row['task']={'hand_load_n':float(load),'opposing_support':opposed,'hand_upward_force_n':float(upward),'tray_force':tray.tolist(),'external_support_force':external.tolist(),'inside_tray':bool(inside),'supported_lift':bool(lifted),'landed':landed}
                row['phase']=self.phases.update(float(data.time),load>.1,opposed,lifted,landed,load)
                self.emit(row);self.frames+=1
        return result
