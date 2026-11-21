"""Explicit table-only diagnostic contract; frozen stand code and admission stay separate.

Selectively adapted from table-v1-integration 9139f3423a44ce3db2fdc3069031ba970a8f229b.
Only the table instance loses its legacy pickup collider. No recorded achieved pose
is used as a command, and diagnostic success never constitutes table calibration.
"""
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

TASK_VERSION='table-v1'
BASE_GAINS={'position':400,'velocity':60,'rotation':15,'angular_velocity':.1,'force_limit':80,'torque_limit':8}
FINGER_GAINS={'finger_kp':2,'wrist_kp':2,'kd':.02,'gravity_compensation':True,'torque_limit_nm':1,'source_kp':2}
REMAINING_GATES=['independent table static support and 2/5/10 cm drops','finger resistance and opening recovery',
    'forearm/table interaction and base saturation','separately hashed table model/controller/solver calibration',
    'repeatable complete human command replays and conservative fault-window admission','causal 60 Hz export and first-observation oracle']


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def scene_identity(scene):return hashlib.sha256(json.dumps(scene,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def finite_vector(value,count,name):
    if not isinstance(value,list) or len(value)!=count or any(type(v) not in (int,float) or not math.isfinite(v) for v in value):
        raise ValueError('Expected finite '+name)
    return value


def validate_scene(scene):
    if scene.get('task_version')!=TASK_VERSION or scene.get('pickup_support') is not False:
        raise ValueError('Requires explicit table-v1 scene with pickup_support false; stand/custom routing is separate')
    for key in ['table_position','table_size','block_position','block_size','pickup_position','pickup_size']:
        finite_vector(scene[key],3,key)
    finite_vector(scene['tray_center'],2,'tray_center')
    for key in ['table_size','block_size','pickup_size']:
        if any(v<=0 for v in scene[key]):raise ValueError('Nonpositive '+key)
    for key in ['block_mass_kg','tray_internal_width','tray_wall_height','tray_floor_top']:
        value=scene[key]
        if type(value) not in (int,float) or not math.isfinite(value):raise ValueError('Invalid '+key)
        if key!='tray_floor_top' and value<=0:raise ValueError('Nonpositive '+key)
    rest=scene['table_position'][2]+scene['table_size'][2]/2+scene['block_size'][2]/2
    if abs(scene['block_position'][2]-rest)>1e-6:
        raise ValueError('Nominal block rest height does not match tabletop; layout check is not a physical tolerance')
    for axis in [0,1]:
        if abs(scene['block_position'][axis]-scene['table_position'][axis])+scene['block_size'][axis]/2>scene['table_size'][axis]/2:
            raise ValueError('Nominal block footprint extends beyond table')


def episode_contract(source,validator):
    episode=source['episode'];initial=episode['initial_conditions'];ready=initial['model_and_scene'];scene=ready['scene'];validate_scene(scene)
    if episode.get('embodiment')!='orca-v2-right' or ready.get('embodiment')!='orca-v2-right':raise ValueError('Unsupported source embodiment')
    if ready.get('frame')!='right-handed-z-up' or ready.get('quaternion_order')!='xyzw':raise ValueError('Do not convert captured simulator coordinates twice')
    if ready['joint_ids']!=validator.IDS:raise ValueError('Canonical joint order changed')
    runtime_pin=json.loads((validator.ROOT/'public/superdex/orca-model.json').read_text())['runtime_urdf_sha256']
    if ready['model_sha256']!=runtime_pin or episode['model_sha256']!=runtime_pin:raise ValueError('Captured runtime model differs from the immutable local pin')
    pins=ready['model_manifest']
    if pins['xml_sha256']!=validator.model_fingerprint() or pins['collision_assets_sha256']!=validator.asset_fingerprint():
        raise ValueError('Captured source physics pins differ from actual frozen assets; regenerated substitution is forbidden')
    if ready['base_gains']!=BASE_GAINS or any(ready['controller'].get(k)!=v for k,v in FINGER_GAINS.items()):
        raise ValueError('Captured controller differs from the diagnostic controller contract')
    if ready['controller'].get('armature_source')!='pinned MuJoCo dof_armature':raise ValueError('Unknown armature semantics')
    for key in ['physics_dt','command_hz']:
        if type(ready[key]) not in (int,float) or not math.isfinite(ready[key]) or ready[key]<=0:raise ValueError('Invalid native '+key)
    geometry={k:scene[k] for k in ['task_version','pickup_support','table_position','table_size','block_size','block_mass_kg','tray_center','tray_floor_top','tray_internal_width','tray_wall_height']}
    state=initial['achieved_before_first_sample']
    return {'task_version':TASK_VERSION,'source_embodiment':ready['embodiment'],'source_runtime_model_sha256':ready['model_sha256'],
        'source_xml_sha256':pins['xml_sha256'],'source_collision_assets_sha256':pins['collision_assets_sha256'],
        'scene_sha256':scene_identity(scene),'geometry_sha256':scene_identity(geometry),'controller_sha256':scene_identity({'finger':ready['controller'],'base':ready['base_gains']}),
        'measured_initialization_sha256':scene_identity(state),'initial_sim_time':initial['sim_time'],'native_physics_dt':ready['physics_dt'],
        'native_command_hz':ready['command_hz'],'independent_physics_dt':.0002,'coordinate_conversion_applied':False,
        'initialization':'Captured measured joints, joint velocities, base pose/world twist and block pose/world twist; original Jacobian initialization',
        'source_session':ready.get('session'),'recorded_placement_id':scene.get('placement_id')}


def install_table_geometry(validator):
    if getattr(validator,'_table_geometry_installed',False):raise ValueError('Table adapter already installed on this validator instance')
    original=validator.task_model
    def table_model(scene,profile='source-default'):
        validate_scene(scene);model,configuration=original(scene,profile);pickup=model.geom('pickup').id
        model.geom_contype[pickup]=model.geom_conaffinity[pickup]=0;model.geom_rgba[pickup,3]=0
        configuration.update(task_version=TASK_VERSION,scene_sha256=scene_identity(scene),pickup_support=False,
            table_calibration='pending',table_adapter_sha256=digest(__file__))
        return model,configuration
    validator.task_model=table_model;validator._table_geometry_installed=True;return original


def pending_qualification():
    return {'qualification_status':'qualification_pending','physics_validated':False,'training_eligible':False,
        'table_calibration_passed':False,'remaining_gates':list(REMAINING_GATES),
        'admission_reason':'A stand controller/contact configuration or a diagnostic task pass is not an independently qualified table recipe'}


def load_frozen_validator(workspace,code_root):
    workspace=Path(workspace).resolve();code_root=Path(code_root).resolve();os.environ['WEI_WORKSPACE_ROOT']=str(workspace)
    lock=json.loads((Path(__file__).resolve().parents[1]/'experiments/milestone1_gpu_baseline/baseline-lock.json').read_text())
    for name,expected in lock['native_helper_sha256'].items():
        if digest(code_root/name)!=expected:raise ValueError('Frozen stand helper mismatch: '+name)
    sys.path[:0]=[str(code_root),str(workspace/'scripts'),str(workspace/'artifacts/orca_sim/src'),str(Path(__file__).resolve().parent)]
    spec=importlib.util.spec_from_file_location('table_trace_utility',code_root/'orca-trace-replay.py')
    utility=importlib.util.module_from_spec(spec);spec.loader.exec_module(utility);utility.ROOT=workspace;validator=utility.load_replay()
    if validator.ROOT.resolve()!=workspace:raise ValueError('Cached simulator helper belongs to a different workspace')
    if digest(sys.modules['orca_physics'].__file__)!=lock['native_helper_sha256']['orca_physics.py']:
        raise ValueError('Cached physics module is not the immutable stand implementation')
    profile,stand_calibration_sha256=utility.verify_calibration(validator)
    return validator,profile,stand_calibration_sha256
