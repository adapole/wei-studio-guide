"""Full 15-second closed-loop table ACT evaluation; no recorded-command tail."""
import argparse
import gzip
import json
from pathlib import Path
import sys
import numpy as np
from orca_table_task import digest,episode_contract,install_table_geometry,load_frozen_validator
from orca_table_qualification import verify_table_calibration,install_qualified_table
from orca_table_policy import TableACTCheckpoint


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['workspace','bundle','checkpoint','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--split',choices=['train','validation'],required=True)
    p.add_argument('--temporal-ensemble',type=float,default=.01)
    p.add_argument('--no-temporal-ensemble',action='store_true')
    p.add_argument('--startup-seconds',type=float,default=0)
    p.add_argument('--original-training-only',action='store_true',help='Explicit diagnostic subset: original training episodes, excluding synthetic derivatives')
    a=p.parse_args();repo=Path(__file__).resolve().parents[1]
    r,profile,_=load_frozen_validator(a.workspace,repo/'experiments/milestone1_gpu_baseline/native-v2')
    import torch
    from orca_policy_command_adapter import JointCommands
    from orca_diagnostic_trace import DiagnosticTrace
    from orca_compressed_trace import CompressedTracePath
    from orca_trace_summary import summarize_trace
    bundle=json.loads((a.bundle/'manifest.json').read_text());marker=json.loads((a.checkpoint/'orca-table-representation.json').read_text())
    if a.original_training_only and a.split!='train':raise ValueError('Validation must always use its complete frozen split')
    selected_entries=[e for e in bundle['episodes'] if e['split']==a.split and (not a.original_training_only or not e.get('synthetic_augmentation'))]
    if not selected_entries:raise ValueError('Empty evaluation split')
    if not bundle['training_eligible'] or marker['bundle_manifest_sha256']!=digest(a.bundle/'manifest.json') or marker['adapter_sha256']!=bundle['table_adapter_sha256']:raise ValueError('Dataset/checkpoint identity mismatch')
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);torch.manual_seed(7);np.random.seed(7)
    from orca_table_representation import adapter
    model=adapter(bundle).TableACTCheckpoint(a.checkpoint,'cuda',temporal_ensemble_coeff=None if a.no_temporal_ensemble else a.temporal_ensemble)
    if a.startup_seconds:
        from orca_table_startup import MeasuredStartup
        policy=MeasuredStartup(model,a.startup_seconds)
    else:policy=model
    report={'schema':'orca.table.policy.evaluation.v1','task_version':'table-v1','scope':'simulation_only','split':a.split,'horizon_seconds':15,'command_hz':60,'physics_dt':.0002,
        'checkpoint_hashes':{f.name:digest(f) for f in a.checkpoint.iterdir() if f.is_file()},'runner_sha256':digest(__file__),'bundle_manifest_sha256':digest(a.bundle/'manifest.json'),
        'inference_recipe':model.inference_recipe,'startup_seconds':a.startup_seconds,'startup_adapter_sha256':digest(repo/'scripts/orca_table_startup.py') if a.startup_seconds else None,
        'raw_action_semantics':'Original decoded predictions and startup modifications retained separately; trace raw action is after startup and before bounded joint submission' if a.startup_seconds else 'Decoded model prediction before bounded joint submission',
        'calibration_sha256':bundle['calibration_sha256'],'all_step_physical_gates_unchanged':True,'held_out_accessed':False,'episodes':[],'complete':False,
        'original_training_only':a.original_training_only,'selected_episode_count':len(selected_entries)}
    def save():(a.output/'report.json').write_text(json.dumps(report,indent=2,default=r.json_default,allow_nan=False))
    save();original=r.policy_action;installed=False
    for entry in selected_entries:
        folder=a.output/Path(entry['file']).stem;folder.mkdir();row={'source_sha256':entry['source_sha256'],'session_id':entry['session_id'],'placement_id':entry['placement_id'],'task_passed':False,'pipeline_completed':False}
        commands=JointCommands(original,r.IDS,'bounded');r.policy_action=commands
        try:
            if digest(a.bundle/entry['fixture'])!=entry['fixture_sha256']:raise ValueError('Fixture changed')
            fixture=json.loads((a.bundle/entry['fixture']).read_text());source=fixture['replay_source']
            _,calhash=verify_table_calibration(a.bundle.parent/'calibration/report.json',episode_contract(source,r),validator_path=a.workspace/'scripts/orca-superdex-validate.py',profile=profile)
            if calhash!=bundle['calibration_sha256']:raise ValueError('Calibration changed')
            if not installed:install_table_geometry(r);install_qualified_table(r,calhash);installed=True
            policy.reset();trace=CompressedTracePath(folder/'trace.ndjson.gz')
            with DiagnosticTrace(r,source,trace,policy=True,control_source='learned_table_act_with_measured_startup' if a.startup_seconds else 'learned_table_act'):
                result=r.replay(source,profile,policy=policy,duration_override=15)
            with gzip.open(folder/'trajectory.json.gz','xt') as stream:json.dump(result,stream,default=r.json_default,allow_nan=False)
            row.update(pipeline_completed=True,task_passed=bool(result['passed']),result={k:v for k,v in result.items() if k!='frames'},trace_sha256=digest(trace.path),diagnostics=summarize_trace(trace,r.IDS))
        except Exception as exc:row['error']=type(exc).__name__+': '+str(exc)
        finally:r.policy_action=original
        row['commands']=commands.report();row['startup']=policy.report() if a.startup_seconds else None
        row['unmodified_policy_task_passed']=row['task_passed'] and not commands.events and not a.startup_seconds
        np.savez_compressed(folder/'raw-encoded-policy-actions.npz',action=model.raw_encoded)
        if a.startup_seconds:np.savez_compressed(folder/'raw-decoded-policy-actions.npz',action=policy.raw_native)
        (folder/'report.json').write_text(json.dumps(row,indent=2,default=r.json_default,allow_nan=False));report['episodes'].append(row);save()
        print(entry['file'],'pass',row['task_passed'],'error',row.get('error'),flush=True)
    report.update(complete=True,successes=sum(e['task_passed'] for e in report['episodes']));save()


if __name__=='__main__':main()
