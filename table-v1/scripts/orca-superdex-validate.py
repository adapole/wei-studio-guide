"""Independent command replay. Recorded achieved poses are initialization/comparison only."""
import argparse,hashlib,json,math,time,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import mujoco
from orca_physics import ROOT,model_spec,quaternion_error,model_fingerprint,asset_fingerprint
from orca_policy_interface import observation as policy_observation, action as policy_action

IDS=['right_wrist','right_p-abd','right_p-mcp','right_p-pip','right_r-abd','right_r-mcp','right_r-pip','right_m-abd','right_m-mcp','right_m-pip','right_i-abd','right_i-mcp','right_i-pip','right_t-cmc','right_t-abd','right_t-mcp','right_t-pip']
ENVELOPE=.002
PROFILES={'source-default':None,'stiff-5ms':.005,'tray-only-3.5ms':None,'tray-clearance-3.5ms':None,'tray-clearance-environment-3.5ms':None}
def json_default(value):
 if isinstance(value,np.generic):return value.item()
 if isinstance(value,np.ndarray):return value.tolist()
 raise TypeError(f'Unsupported report value: {type(value).__name__}')

def vector(value,count):
 if not isinstance(value,list) or len(value)!=count or any(type(x) not in [int,float] or not math.isfinite(x) for x in value):raise ValueError(f'Expected {count} finite numbers')
 return np.array(value,dtype=float)

def pose(value):
 p=vector(value['position'],3);q=vector(value['quaternion'],4)
 if abs(np.linalg.norm(q)-1)>.01:raise ValueError('Non-unit XYZW quaternion')
 return p,q[[3,0,1,2]]/np.linalg.norm(q)

def load(path):
 raw=path.read_bytes()
 if len(raw)>64*1024*1024:raise ValueError('Episode exceeds 64MB')
 source=json.loads(raw)
 if source.get('schema')!='orca.physics.episode.v4':raise ValueError('Requires physical v4; legacy v3 readers remain separate')
 episode=source['episode'];initial=episode['initial_conditions'];ready=initial['model_and_scene']
 pin=json.loads((ROOT/'public/superdex/orca-model.json').read_text())
 source_pin=json.loads((ROOT/'artifacts/superdex/orca-candidate/manifest.json').read_text())['source_model']
 if asset_fingerprint()!=source_pin['collision_assets_sha256']:raise ValueError('Pinned MuJoCo collision assets changed')
 if episode['state_source']!='superdex_simulation' or ready['frame']!='right-handed-z-up' or ready['quaternion_order']!='xyzw':raise ValueError('Unexpected state source or coordinate frame')
 if ready['joint_ids']!=IDS or ready['model_sha256']!=pin['runtime_urdf_sha256'] or episode['model_sha256']!=ready['model_sha256']:raise ValueError('Model hash or canonical joint mapping mismatch')
 cache=ROOT/'artifacts/superdex/orca-baked/orca-right.urdf'
 if hashlib.sha256(cache.read_bytes()).hexdigest()!=ready['model_sha256']:raise ValueError('Local pinned collision model changed')
 # Replay must use the exact immutable source and baked collision assets.
 for folder,manifest_name in [('orca-candidate','manifest.json'),('orca-baked','manifest.json')]:
  directory=ROOT/'artifacts/superdex'/folder;metadata=json.loads((directory/manifest_name).read_text())
  expected={'orca-right.urdf':metadata['urdf_sha256']}
  expected.update(metadata.get('mesh_hashes',{}));expected.update({row['file']:row['sha256'] for row in metadata.get('files',[])})
  for filename,digest in expected.items():
   if hashlib.sha256((directory/filename).read_bytes()).hexdigest()!=digest:raise ValueError('Pinned collision asset mismatch: '+filename)
 frames=source['frames']
 if not 1<=len(frames)<=1800:raise ValueError('Expected 1..1800 physics samples')
 state=initial['achieved_before_first_sample'];vector(state['joint_positions'],17);vector(state['joint_velocities'],17);pose(state['base_pose'])
 vector(state['base_twist_world']['linear'],3);vector(state['base_twist_world']['angular'],3)
 obj=next(x for x in state['objects'] if x['id']=='block');pose(obj);vector(obj['linear_velocity'],3);vector(obj['angular_velocity'],3)
 events=episode['command_application_events'];previous=-1;stamp=-math.inf
 if not events:raise ValueError('Missing authoritative command-application times')
 submitted={row['command']['seq']:row['command'] for row in episode['commands']}
 for event in events:
  if type(event['seq'])!=int or event['seq']<=previous or event['first_applied_sim_time']<stamp:raise ValueError('Reordered command application events')
  previous=event['seq'];stamp=event['first_applied_sim_time'];vector(event['joint_position_targets'],17);pose(event['base_target'])
  sent=submitted.get(event['seq'])
  if sent and (sent['joint_position_targets']!=event['joint_position_targets'] or sent['base_target']!=event['base_target']):raise ValueError('Submitted/application command mismatch')
  if not sent and event['input_time_ms']>episode['clocks']['browser_record_start_ms']:raise ValueError('Applied command has no submitted provenance')
 return source,hashlib.sha256(raw).hexdigest()

def robot_spec(profile='source-default',gravity=True,free_base=True):
 if mujoco.__version__!='3.17.0':raise ValueError('MuJoCo 3.17.0 is pinned')
 if profile not in PROFILES:raise ValueError('Unknown frozen contact profile')
 spec=model_spec(gravity=gravity);spec.option.timestep=.0002
 disabled_environment=[]
 for geom in spec.geoms:
  body=geom.parent
  while body is not None and body.name not in ['world','right_mount']:body=body.parent
  if body is None or body.name!='right_mount':geom.contype=geom.conaffinity=0;disabled_environment.append(geom.name or geom.parent.name)
 if free_base:spec.body('right_mount').add_freejoint(name='superdex_base')
 spec.compiler.fusestatic=True
 removed=[(e.bodyname1,e.bodyname2) for e in spec.excludes]
 for exclusion in list(spec.excludes):spec.delete(exclusion)
 urdf=ET.parse(ROOT/'artifacts/superdex/orca-candidate/orca-right.urdf').getroot()
 parent={j.find('child').get('link'):(j.find('parent').get('link'),j.get('type')) for j in urdf.findall('joint')}
 def weld_root(name):
  while name in parent and parent[name][1]=='fixed':name=parent[name][0]
  return name
 groups={}
 for link in urdf.findall('link'):
  if link.find('collision') is not None:groups.setdefault(weld_root(link.get('name')),[]).append(link.get('name'))
 pairs=set()
 for child,(p,kind) in parent.items():
  if kind=='fixed':continue
  for a in groups.get(weld_root(child),[]):
   for b in groups.get(weld_root(p),[]):pairs.add(tuple(sorted([a,b])))
 for i,(a,b) in enumerate(sorted(pairs)):spec.add_exclude(name=f'native_neighbour_{i}',bodyname1=a,bodyname2=b)
 spec.add_exclude(name='native_palm_thumb_bearing',bodyname1='right_R-Carpals_8d1f1041',bodyname2='right_R-T-AP_a9723101')
 # Match native Coulomb sliding contact. These are simulator settings, not hardware measurements.
 for geom in spec.geoms:
  if geom.contype or geom.conaffinity:
   geom.friction=[1.2,0,0];geom.condim=3
   if profile in ['tray-clearance-3.5ms','tray-clearance-environment-3.5ms']:geom.margin=0;geom.gap=0
   if PROFILES[profile] is not None:geom.solref=[PROFILES[profile],1]
 return spec,dict(profile=profile,removed_upstream_excludes=removed,native_neighbour_excludes=sorted(pairs),disabled_source_environment=disabled_environment,
    contact_time_constant_override_s=PROFILES[profile],source_model_fingerprint=model_fingerprint())

def task_model(scene,profile='source-default'):
 spec,description=robot_spec(profile)
 def box(name,size,position):
  geom=spec.worldbody.add_geom(name=name,type=mujoco.mjtGeom.mjGEOM_BOX,size=np.array(size)/2,pos=position,friction=[1.2,0,0],condim=3)
  if PROFILES[profile] is not None:geom.solref=[PROFILES[profile],1]
  if profile=='tray-clearance-environment-3.5ms':geom.priority=1;geom.solref=[.0035,1];geom.margin=0;geom.gap=0
  return geom
 box('pickup',scene['pickup_size'],scene['pickup_position']);box('table',scene['table_size'],scene['table_position'])
 x,y=scene['tray_center'];z=scene['tray_floor_top'];w=scene['tray_internal_width'];h=scene['tray_wall_height']
 box('tray_floor',[w+.008,w+.008,.008],[x,y,z-.004])
 for sign in [-1,1]:
  box(f'tray_x_{sign}',[.004,w+.008,h],[x+sign*(w/2+.002),y,z+h/2]);box(f'tray_y_{sign}',[w,.004,h],[x,y+sign*(w/2+.002),z+h/2])
 block=spec.worldbody.add_body(name='replay_block');block.add_freejoint(name='replay_block_free')
 block.add_geom(name='replay_block_geom',type=mujoco.mjtGeom.mjGEOM_BOX,size=np.array(scene['block_size'])/2,mass=scene['block_mass_kg'],friction=[1.2,0,0],condim=3)
 if PROFILES[profile] is not None:spec.geom('replay_block_geom').solref=[PROFILES[profile],1]
 if profile in ['tray-only-3.5ms','tray-clearance-3.5ms','tray-clearance-environment-3.5ms']:
  for name in ['tray_floor','tray_x_-1','tray_x_1','tray_y_-1','tray_y_1']:
   spec.add_pair(name='calibrated_'+name,geomname1=name,geomname2='replay_block_geom',
                 condim=3,solref=[.0035,1],margin=0,gap=0,friction=[1.2,1.2,0,0,0])
 model=spec.compile()
 # Explicit external bounded PD replicates the declared native controller. Original position actuators emit zero torque.
 model.actuator_gainprm[:]=0;model.actuator_biasprm[:]=0
 overrides=dict(**description,timestep=.0002,free_base='right_mount; no added mass',fixed_body_fusion=True,
  collision_policy='Explicit native adjacent welded groups plus the native palm/thumb-bearing exclusion',
  sliding_friction=1.2,torsional_and_rolling_friction=0,controller='external bounded PD; upstream position actuator gains/biases disabled',
  penetration_ceiling_m=ENVELOPE,solver=int(model.opt.solver),iterations=int(model.opt.iterations),tolerance=float(model.opt.tolerance))
 overrides['tray_block_pair_time_constant_s']=.0035 if profile in ['tray-only-3.5ms','tray-clearance-3.5ms','tray-clearance-environment-3.5ms'] else None
 overrides['robot_contact_margin_override_m']=0 if profile in ['tray-clearance-3.5ms','tray-clearance-environment-3.5ms'] else None
 overrides['static_environment_contact_time_constant_s']=.0035 if profile=='tray-clearance-environment-3.5ms' else None
 return model,overrides

def initialize(model,state):
 data=mujoco.MjData(model);qp=np.array([model.jnt_qposadr[model.joint(n).id] for n in IDS]);dv=np.array([model.jnt_dofadr[model.joint(n).id] for n in IDS])
 data.qpos[qp]=vector(state['joint_positions'],17);data.qvel[dv]=vector(state['joint_velocities'],17)
 obj=next(x for x in state['objects'] if x['id']=='block')
 for joint_name,value in [('superdex_base',state['base_pose']),('replay_block_free',obj)]:
  adr=model.jnt_qposadr[model.joint(joint_name).id];p,q=pose(value);data.qpos[adr:adr+3]=p;data.qpos[adr+3:adr+7]=q
 mujoco.mj_forward(model,data)
 # Use measured world twists and the actual MuJoCo Jacobians, avoiding a guessed angular-velocity convention.
 for joint_name,body_name,linear,angular in [('superdex_base','right_mount',state['base_twist_world']['linear'],state['base_twist_world']['angular']),('replay_block_free','replay_block',obj['linear_velocity'],obj['angular_velocity'])]:
  body=model.body(body_name).id;jacp=np.zeros((3,model.nv));jacr=np.zeros((3,model.nv));mujoco.mj_jac(model,data,jacp,jacr,data.xpos[body],body)
  adr=model.jnt_dofadr[model.joint(joint_name).id];columns=np.arange(adr,adr+6);jac=np.vstack([jacp,jacr])[:,columns]
  data.qvel[columns]=np.linalg.solve(jac,np.r_[vector(linear,3),vector(angular,3)])
 mujoco.mj_forward(model,data);return data,qp,dv

def supported_lift(z,rest,opposed,hand_upward_force,external_support_load):
 # The approved task requires sustaining the lifted block, not holding its speed below an added threshold.
 return z>=rest+.05 and opposed and hand_upward_force>.4 and external_support_load<.05

def replay(source,profile='source-default',policy=None,duration_override=None):
 ep=source['episode'];initial=ep['initial_conditions'];ready=initial['model_and_scene'];scene=ready['scene']
 model,overrides=task_model(scene,profile);data,qp,dv=initialize(model,initial['achieved_before_first_sample']);gravity=mujoco.MjData(model)
 base=model.body('right_mount').id;block=model.body('replay_block').id;block_geom=model.geom('replay_block_geom').id
 hand_bodies=set()
 for i in range(model.nbody):
  ancestor=i
  while ancestor>0 and ancestor!=base:ancestor=int(model.body_parentid[ancestor])
  if ancestor==base:hand_bodies.add(i)
 start=initial['sim_time'];duration=source['frames'][-1]['state']['sim_time']-start if duration_override is None else duration_override;peak_contact=None
 if not 0<duration<=90:raise ValueError('Invalid simulation duration')
 events=[e for e in ep['command_application_events'] if e['first_applied_sim_time']>=start-1e-9] if policy is None else [];index=0;active=initial['active_action'];policy_tick=0
 gains=ready['base_gains'];rows=[];hold=longest=settle=0.;max_pen=0.;contact_seen=False;failure=None;next_sample=0.;joint_limit_violation=False;started=time.perf_counter()
 ranges=np.array([model.jnt_range[model.joint(n).id] for n in IDS]);rest=scene['block_position'][2]
 state=initial['achieved_before_first_sample'];initial_object=next(x for x in state['objects'] if x['id']=='block')
 initially_resting=abs(initial_object['position'][2]-rest)<=ENVELOPE and np.linalg.norm(initial_object['linear_velocity'])<.02
 for step in range(round(duration/model.opt.timestep)):
  if policy is not None and data.time+1e-9>=policy_tick/60:
   jacp=np.zeros((3,model.nv));jacr=np.zeros((3,model.nv));mujoco.mj_jac(model,data,jacp,jacr,data.xpos[base],base)
   base_pose=[*data.xpos[base],*data.xquat[base][[1,2,3,0]]]
   block_pose=[*data.xpos[block],*data.xquat[block][[1,2,3,0]]]
   tray_pose=[*scene['tray_center'],scene['tray_floor_top'],0,0,0,1]
   features=policy_observation(data.qpos[qp],data.qvel[dv],base_pose,jacp@data.qvel,jacr@data.qvel,block_pose,tray_pose)
   active=policy_action(policy(features),ranges);policy_tick+=1
  while index<len(events) and events[index]['first_applied_sim_time']<=start+data.time+1e-9:
   event=events[index];active={'joint_position_targets':event['joint_position_targets'],'base_target':event['base_target']};index+=1
  targets=vector(active['joint_position_targets'],17);desired_p,desired_q=pose(active['base_target'])
  if np.any(targets<ranges[:,0]-1e-6) or np.any(targets>ranges[:,1]+1e-6):raise ValueError('Position target outside pinned joint limits')
  gravity.qpos[:]=data.qpos;gravity.qvel[:]=0;mujoco.mj_forward(model,gravity)
  pd=np.clip(2*(targets-data.qpos[qp])-.02*data.qvel[dv],-1,1);tau=np.clip(pd+gravity.qfrc_bias[dv],-1,1)
  data.qfrc_applied[:]=0;data.qfrc_applied[dv]=tau
  jacp=np.zeros((3,model.nv));jacr=np.zeros((3,model.nv));mujoco.mj_jac(model,data,jacp,jacr,data.xpos[base],base)
  linear=jacp@data.qvel;angular=jacr@data.qvel;up=-model.body_subtreemass[base]*model.opt.gravity
  force=gains['position']*(desired_p-data.xpos[base])-gains['velocity']*linear+up
  torque=gains['rotation']*quaternion_error(desired_q,data.xquat[base])-gains['angular_velocity']*angular+np.cross(data.subtree_com[base]-data.xpos[base],up)
  requested=np.r_[force,torque];applied=requested.copy()
  for offset,limit in [(0,gains['force_limit']),(3,gains['torque_limit'])]:
   length=np.linalg.norm(applied[offset:offset+3])
   if length>limit:applied[offset:offset+3]*=limit/length
  mujoco.mj_applyFT(model,data,applied[:3],applied[3:],data.xpos[base],base,data.qfrc_applied)
  mujoco.mj_step(model,data)
  mujoco.mj_forward(model,data)
  if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():failure='nonfinite_physics_state';break
  joints=data.qpos[qp];joint_limit_violation|=bool(np.any(joints<ranges[:,0]-.02)|np.any(joints>ranges[:,1]+.02))
  forces={};tray=np.zeros(3);support=np.zeros(3)
  for contact_index,c in enumerate(data.contact):
   penetration=max(0.,-float(c.dist))
   if penetration>max_pen:
    max_pen=penetration;peak_contact=dict(time=float(data.time),penetration_m=penetration,bodies=[model.body(int(model.geom_bodyid[g])).name for g in [c.geom1,c.geom2]],geoms=[model.geom(g).name for g in [c.geom1,c.geom2]])
   if block_geom not in [c.geom1,c.geom2]:continue
   other=c.geom2 if c.geom1==block_geom else c.geom1;other_body=int(model.geom_bodyid[other]);wrench=np.empty(6);mujoco.mj_contactForce(model,data,contact_index,wrench)
   world_force=np.array(c.frame).reshape(3,3).T@wrench[:3]
   if c.geom1==block_geom:world_force=-world_force
   name=model.geom(other).name or ''
   if other_body in hand_bodies:forces[other_body]=forces.get(other_body,np.zeros(3))+world_force
   elif name.startswith('tray_'):tray+=world_force;support+=world_force
   else:support+=world_force
  loads=[f for f in forces.values() if np.linalg.norm(f)>.05]
  opposed=any(float(a@b)/(np.linalg.norm(a)*np.linalg.norm(b))<-.3 for i,a in enumerate(loads) for b in loads[i+1:]);hand_force=sum(forces.values(),start=np.zeros(3));hand_load=sum(np.linalg.norm(v) for v in forces.values());contact_seen|=hand_load>.1
  twist=np.empty(6);mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,block,twist,0);position=data.xpos[block].copy();dt=model.opt.timestep
  held=supported_lift(position[2],rest,opposed,hand_force[2],np.linalg.norm(support))
  hold=hold+dt if held else 0.;longest=max(longest,hold)
  extent=np.abs(data.xmat[block].reshape(3,3))@(np.array(scene['block_size'])/2)
  inside=np.all(np.abs(position[:2]-scene['tray_center'])+extent[:2]<=scene['tray_internal_width']/2+.001)
  landed=inside and abs(position[2]-extent[2]-scene['tray_floor_top'])<=ENVELOPE and tray[2]>.5 and hand_load<.05 and np.linalg.norm(twist[3:])<.02 and np.linalg.norm(twist[:3])<.1
  settle=settle+dt if landed else 0.
  if step%5000==0:print(f"Replay {data.time:.1f}/{duration:.1f}s block={position.round(4).tolist()} hold={longest:.2f}s penetration={max_pen*1000:.2f}mm",flush=True)
  if data.time+1e-9>=next_sample:
   rows.append(dict(time=float(data.time),source_sim_time=start+float(data.time),target=targets.tolist(),joints=joints.tolist(),base_target=active['base_target'],base_position=data.xpos[base].tolist(),base_quaternion_wxyz=data.xquat[base].tolist(),requested_wrench=requested.tolist(),applied_wrench=applied.tolist(),joint_torque=tau.tolist(),object_position=position.tolist(),object_quaternion_wxyz=data.xquat[block].tolist(),opposing_support=opposed,hand_load_n=float(hand_load),tray_force=tray.tolist(),max_penetration_m=max_pen));next_sample+=1/60
 gates=dict(initially_resting=initially_resting,finite=failure is None,supported_lift_hold=longest>=2,released_and_settled=settle>=1,penetration=max_pen<=ENVELOPE,joint_limits=not joint_limit_violation)
 return dict(passed=all(gates.values()),gates=gates,first_failing_stage=next((k for k,v in gates.items() if not v),None),hold_seconds=longest,settle_seconds=settle,max_penetration_m=max_pen,peak_contact=peak_contact,hand_contact_exercised=contact_seen,failure=failure,wall_seconds=time.perf_counter()-started,configuration=overrides,control_source='policy' if policy is not None else 'recorded_commands',policy_steps=policy_tick,frames=rows)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('episode',type=Path);p.add_argument('--output',type=Path,default=ROOT/'artifacts/superdex/mujoco-replay');p.add_argument('--diagnostic-uncalibrated',action='store_true');args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
 report={'validation_scope':'simulation_only','hardware_transfer':'unvalidated','policy_generalization':'unvalidated','source_path':str(args.episode),'source_immutable':True}
 try:
  profile='source-default';calibration_path=ROOT/'artifacts/superdex/mujoco-contact-calibration-v3.json'
  if not args.diagnostic_uncalibrated:
   calibration=json.loads(calibration_path.read_text())
   if not calibration.get('dynamic_base_contact_passed',False) or not calibration['passed'] or calibration['source_model_fingerprint']!=model_fingerprint() or calibration['asset_fingerprint']!=asset_fingerprint() or calibration['validator_sha256']!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest():raise ValueError('Independent joint/self/base contact calibration must pass against the pinned model, assets and validator first')
   profile=calibration['selected_profile'];report['calibration_sha256']=hashlib.sha256(calibration_path.read_bytes()).hexdigest()
  source,digest=load(args.episode);report.update(source_episode_id=source['episode']['id'],source_sha256=digest,diagnostic_uncalibrated=args.diagnostic_uncalibrated,contact_profile=profile)
  results=[replay(source,profile),replay(source,profile)]
  for i,result in enumerate(results):(args.output/f'trial-{i+1}.json').write_text(json.dumps(result,allow_nan=False,default=json_default))
  difference=max(np.linalg.norm(np.array(a['object_position'])-b['object_position']) for a,b in zip(results[0]['frames'],results[1]['frames'])) if results[0]['frames'] and len(results[0]['frames'])==len(results[1]['frames']) else math.inf
  report.update(pipeline_passed=True,task_passed=all(r['passed'] for r in results),repeatable=difference<=1e-9,max_repeat_difference_m=difference if math.isfinite(difference) else None,
   trials=[{k:v for k,v in r.items() if k!='frames'} for r in results],first_failing_stage=next((r['first_failing_stage'] for r in results if not r['passed']),None))
  report['physics_validated_episode_created']=False
  if report['task_passed'] and report['repeatable'] and not args.diagnostic_uncalibrated and not source['episode']['discontinuities']:
   derived={'schema':'orca.physics.validated.v1','validation_scope':'simulation_only','source_episode_id':source['episode']['id'],'source_sha256':digest,'calibration_sha256':report['calibration_sha256'],'profile':profile,'command_provenance':'source submitted commands applied at recorded simulation times','achieved_trajectory':results[0]['frames']}
   (args.output/'physics-validated-episode.json').write_text(json.dumps(derived,allow_nan=False,default=json_default))
   report['physics_validated_episode_created']=True
  report['remaining_validation']='Only repeatable calibrated task passes without recorded discontinuities produce a derived episode; no trajectory is corrected.'
 except Exception as error:report.update(pipeline_passed=False,task_passed=False,error=f'{type(error).__name__}: {error}')
 (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False,default=json_default));print(json.dumps(report,indent=2,default=json_default));return 0 if report.get('pipeline_passed') else 2

if __name__=='__main__':raise SystemExit(main())
