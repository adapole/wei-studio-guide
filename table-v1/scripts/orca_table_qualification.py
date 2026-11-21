"""Hash-bound admission for the independently executed table component suite."""
import json
import gzip
from pathlib import Path
import numpy as np
from orca_table_task import digest,scene_identity


def verify_table_calibration(path,contract,*,validator_path,profile):
    path=Path(path);cal=json.loads(path.read_text());repo=Path(__file__).resolve().parents[1]
    if cal['schema']!='orca.table.calibration.v1' or not cal['complete'] or not cal['passed']:raise ValueError('Table component qualification incomplete')
    if cal['runner_sha256']!=digest(repo/'scripts/orca-table-calibrate.py') or cal['adapter_sha256']!=digest(repo/'scripts/orca_table_task.py'):raise ValueError('Table calibration code changed')
    if cal['validator_sha256']!=digest(validator_path) or cal['profile']!=profile:raise ValueError('Table validator/profile changed')
    if cal['parent_robot_calibration_sha256']!=digest(Path(validator_path).parents[1]/'artifacts/superdex/mujoco-contact-calibration-v3.json'):raise ValueError('Parent robot recipe changed')
    if cal['penetration_ceiling_m']!=.002 or cal['joint_limit_tolerance_rad']!=.02 or cal['physics_dt']!=.0002:raise ValueError('Physical thresholds changed')
    for key in ['geometry_sha256','controller_sha256','source_runtime_model_sha256','source_xml_sha256','source_collision_assets_sha256','source_embodiment']:
        if cal['contract'][key]!=contract[key]:raise ValueError('Table calibration contract mismatch: '+key)
    expected={*[f'static-{i:02}' for i in range(10)],'drop-02cm','drop-05cm','drop-10cm','finger-resistance-opening','forearm-table','base-saturation'}
    if {c['specification']['name'] for c in cal['cases']}!=expected or len(cal['cases'])!=16:raise ValueError('Incomplete table probe matrix')
    for case in cal['cases']:
        if not case['passed'] or not case['repeatable'] or len(case['trials'])!=2:raise ValueError('Failed component case')
        if len({Path(t['report']).resolve() for t in case['trials']})!=2:raise ValueError('A duplicate report is not a repeated experiment')
        results=[]
        for trial in case['trials']:
            recorded=Path(trial['report']); report=path.parent/recorded.relative_to(Path(cal['cases'][0]['trials'][0]['report']).parents[1])
            if digest(report)!=trial['sha256']:raise ValueError('Component report changed')
            row=json.loads(report.read_text());result=row['physical_result'];m=row['measurements'];g=result['gates']
            if not (row['passed'] and g['finite'] and g['joint_limits'] and g['penetration']) or result['max_penetration_m']>.002:raise ValueError('Component physical gate failed')
            if m['steps']!=round(case['specification']['duration']/.0002):raise ValueError('Incomplete all-physics-step observation')
            if result['configuration']['profile']!=profile or result['configuration']['penetration_ceiling_m']!=.002:raise ValueError('Component configuration changed')
            trace=report.with_suffix('.ndjson.gz')
            if digest(trace)!=row['trace_sha256']:raise ValueError('Component trace changed')
            kind=case['specification']['kind']
            if kind=='environment' and m['final_settled_seconds']<1:raise ValueError('Support/drop has not settled')
            if kind=='environment':
                first=None;last=None
                with gzip.open(trace,'rt') as stream:
                    for line in stream:
                        sample=json.loads(line)
                        if sample['type']=='sample':
                            if first is None:first=sample
                            last=sample
                initial=first['state_before']['objects'][0]['position'];final=last['state_after']['objects'][0]['position']
                if np.linalg.norm(np.array(final[:2])-case['specification']['position'])>.002 or abs(final[2]-(initial[2]-case['specification']['height']))>.002:raise ValueError('Block support position drifted')
            if kind=='finger' and not (m['hand_block_peak_n']>.1 and m['grip_resistance_error_rad']>.01 and m['opening_error_rad']<=.02):raise ValueError('Resistance/opening not exercised')
            if kind=='forearm' and m['forearm_table_peak_n']<=.1:raise ValueError('Forearm contact not exercised')
            if kind=='saturation':
                requested=np.array(m['requested_wrench']);applied=np.array(m['applied_wrench'])
                if not (np.linalg.norm(requested[:3])>80 and np.linalg.norm(requested[3:])>8 and np.linalg.norm(applied[:3])<=80+1e-9 and np.linalg.norm(applied[3:])<=8+1e-9):raise ValueError('Wrench saturation failed')
            results.append({k:v for k,v in result.items() if k!='wall_seconds'})
        if results[0]!=results[1]:raise ValueError('Component physics results differ')
    return cal,digest(path)


def install_qualified_table(validator,calibration_sha256):
    original=validator.task_model
    def model(scene,profile='source-default'):
        built,configuration=original(scene,profile)
        configuration.update(table_calibration='independent-component-suite-v1',table_calibration_sha256=calibration_sha256)
        return built,configuration
    validator.task_model=model


def inspect_continuity(source):
    ep=source['episode'];frames=source['frames'];inputs=ep['inputs'];times=[row['state']['sim_time'] for row in frames]
    if ep.get('input_classification')!='quest_webxr' or ep.get('assistance'):raise ValueError('Requires unassisted human Quest input')
    if not all(row.get('tracking_valid') is True for row in inputs):raise ValueError('Tracking fault requires a separately admitted conservative window')
    if not all(row['state']['status']=='running' for row in frames):raise ValueError('Non-running physics requires window admission')
    if any(b<=a or b-a>.05+1e-9 for a,b in zip(times,times[1:])):raise ValueError('Missing/nonmonotonic authoritative physics')
    if any(row['kind']!='command_clock_gap' for row in ep['discontinuities']):raise ValueError('A tracking/watchdog interruption cannot be bridged')
    events=ep['command_application_events'];event_times=[e['first_applied_sim_time'] for e in events]
    maximum=max((b-a for a,b in zip(event_times,event_times[1:])),default=0)
    if maximum>ep['initial_conditions']['model_and_scene']['watchdog_ms']/1000:raise ValueError('Command hold crosses the declared watchdog interval')
    return {'recipe':'held-command-with-continuous-authoritative-physics-v1','native_discontinuities':ep['discontinuities'],
        'tracking_valid_inputs':len(inputs),'physics_frames':len(frames),'maximum_physics_gap_s':max((b-a for a,b in zip(times,times[1:])),default=0),
        'maximum_application_gap_s':maximum,'interpretation':'Missed browser submission ticks with continuously running measured physics and valid tracking retain the previously submitted command; no tracking/watchdog fault is stitched.'}
