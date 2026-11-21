"""Single-episode learning gate: oracle control, prediction errors, then closed loop."""
import argparse,gc,hashlib,importlib.util,json,os,subprocess,sys
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,default=Path.cwd());p.add_argument('--bundle',type=Path,required=True);p.add_argument('--fixture',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--oracle-only',action='store_true');p.add_argument('--steps',type=int,default=5000);p.add_argument('--joint-command-mode',choices=['strict','bounded'],default='strict');p.add_argument('--previous-checkpoint',type=Path);p.add_argument('--learning-rate',type=float,default=1e-5);a=p.parse_args()
    if not np.isfinite(a.learning_rate) or not 0<a.learning_rate<=.001:raise ValueError('Learning rate must be finite and in (0, 0.001]')
    root=a.workspace.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);os.environ['WEI_WORKSPACE_ROOT']=str(root);sys.path.insert(0,str(root/'scripts'));sys.path.insert(0,str(Path(__file__).resolve().parent))
    from orca_diagnostic_trace import DiagnosticTrace
    from orca_trace_summary import summarize_trace
    try:
        utility=module('traceutil',Path(__file__).with_name('orca-trace-replay.py'));utility.ROOT=root;r=utility.load_replay();profile,calhash=utility.verify_calibration(r)
    except Exception as exc:
        (out/'report.json').write_text(json.dumps({'scope':'simulation_only','stage':'preflight','milestone_1_complete':False,'error':f'{type(exc).__name__}: {exc}'},indent=2));raise
    report={'scope':'single-episode debugging simulation_only','milestone_1_complete':False,'hardware_transfer':'unvalidated','oracle_passed':False,'joint_command_mode':a.joint_command_mode,'fixture_sha256':sha(a.fixture),'bundle_manifest_sha256':sha(a.bundle/'manifest.json'),'calibration_sha256':calhash,'rollouts':[]}
    save=lambda: (out/'report.json').write_text(json.dumps(report,indent=2,default=r.json_default,allow_nan=False))
    try:
        import importlib.metadata
        report['dependency_versions']={n:importlib.metadata.version(n) for n in ['mujoco','numpy']}
        snapshot=out/'code-snapshot';snapshot.mkdir();report['code_sha256']={}
        for name in ['orca-single-episode-repair.py','orca_diagnostic_trace.py','orca_task_phases.py','orca_trace_summary.py','orca_prediction_metrics.py','orca_policy_command_adapter.py','orca_lerobot_policy.py','orca_physics.py']:
            path=Path(__file__).with_name(name)
            if not path.exists():path=root/'scripts'/name
            body=path.read_bytes();(snapshot/name).write_bytes(body);report['code_sha256'][name]=hashlib.sha256(body).hexdigest()
        report['pinned_validator_sha256']=sha(root/'scripts/orca-superdex-validate.py')
        fixture=json.loads(a.fixture.read_text());manifest=json.loads((a.bundle/'manifest.json').read_text());entry=next(e for e in manifest['episodes'] if e['split']=='train')
        if fixture['schema']!='orca.mujoco.initialization.v1' or fixture['calibration_sha256']!=calhash or fixture['source_sha256']!=entry['source_sha256'] or manifest.get('state_source')!='mujoco_simulation' or not manifest['training_eligible']:raise ValueError('Fixture or MuJoCo dataset provenance mismatch')
        # Independently verify the cached pinned XML/URDF/meshes using the existing source loader.
        r.load(root/'artifacts/orca-playtest-1794135408869.json')
        source=fixture['replay_source'];ready=source['episode']['initial_conditions']['model_and_scene'];pin=json.loads((root/'public/superdex/orca-model.json').read_text())
        if ready['joint_ids']!=r.IDS or ready['model_sha256']!=pin['runtime_urdf_sha256']:raise ValueError('Fixture robot mapping/model mismatch')
        with DiagnosticTrace(r,source,out/'oracle.ndjson') as observer:oracle=r.replay(source,profile)
        report['oracle_passed']=bool(oracle['passed']);report['oracle_result']={k:v for k,v in oracle.items() if k!='frames'};report['oracle_diagnostics']=summarize_trace(out/'oracle.ndjson',r.IDS)
        (out/'oracle-trajectory.json').write_text(json.dumps(oracle,default=r.json_default,allow_nan=False));save();gc.collect()
        if not report['oracle_passed']:raise ValueError('Oracle command replay failed; do not train around a controller/task mismatch')
        if a.oracle_only:return 0
        import torch
        from orca_lerobot_policy import ACTCheckpoint
        from orca_prediction_metrics import metrics
        from orca_policy_command_adapter import JointCommands
        if not torch.cuda.is_available():raise ValueError('Run learning on the existing CUDA machine')
        report['dependency_versions'].update({n:importlib.metadata.version(n) for n in ['torch','lerobot']})
        torch.manual_seed(7);np.random.seed(7)
        if a.previous_checkpoint:
            previous=ACTCheckpoint(a.previous_checkpoint,'cuda');samples=np.load(a.bundle/entry['file']);predictions=[]
            for state,environment in zip(samples['state'],samples['environment']):predictions.append(previous({'observation.state':state,'observation.environment_state':environment}))
            report['previous_checkpoint_teacher_forced']=metrics(predictions,samples['action'],r.IDS);report['previous_checkpoint_hashes']={f.name:sha(f) for f in a.previous_checkpoint.iterdir() if f.is_file()};np.savez_compressed(out/'previous-teacher-forced.npz',predicted=predictions,submitted=samples['action'],time=samples['time']);samples.close();del previous;gc.collect();torch.cuda.empty_cache();save()
        ds=out/'single-dataset';subprocess.run([sys.executable,str(Path(__file__).with_name('orca-lerobot-convert.py')),str(a.bundle.resolve()),str(ds),'--repo-id','local/orca-mujoco-single','--split','train','--max-segments','1'],check=True,cwd=root)
        cli=Path(sys.executable).parent/'lerobot-train';training=out/'training'
        cmd=[str(cli),'--dataset.repo_id=local/orca-mujoco-single',f'--dataset.root={ds}','--policy.type=act','--policy.device=cuda','--policy.push_to_hub=false','--policy.pretrained_backbone_weights=null','--policy.chunk_size=30','--policy.n_action_steps=1','--policy.dim_model=128','--policy.n_heads=4','--policy.dim_feedforward=512','--policy.n_encoder_layers=2','--policy.use_vae=false','--batch_size=4',f'--steps={a.steps}','--num_workers=0','--seed=7',f'--output_dir={training}',f'--save_freq={a.steps}']
        cmd.append(f'--policy.optimizer_lr={a.learning_rate}')
        report['training_command']=cmd;save()
        with (out/'training.log').open('w') as log:subprocess.run(cmd,cwd=root,check=True,stdout=log,stderr=subprocess.STDOUT)
        checkpoint=training/'checkpoints'/f'{a.steps:06d}'/'pretrained_model';report['checkpoint_hashes']={f.name:sha(f) for f in checkpoint.iterdir() if f.is_file()}
        data=np.load(a.bundle/entry['file']);model=ACTCheckpoint(checkpoint,'cuda');pred=[]
        for state,environment in zip(data['state'],data['environment']):pred.append(model({'observation.state':state,'observation.environment_state':environment}))
        report['teacher_forced']=metrics(pred,data['action'],r.IDS);np.savez_compressed(out/'teacher-forced.npz',predicted=pred,submitted=data['action'],time=data['time']);del model;gc.collect();torch.cuda.empty_cache();save()
        submit=r.policy_action
        for name,options in [('one-action',{'action_steps':1}),('ten-actions',{'action_steps':10}),('temporal-ensemble',{'action_steps':1,'temporal_ensemble_coeff':.01})]:
            torch.manual_seed(7);model=ACTCheckpoint(checkpoint,'cuda',**options);commands=JointCommands(submit,r.IDS,a.joint_command_mode);r.policy_action=commands;row={'recipe':name,'inference_recipe':model.inference_recipe,'task_passed':False}
            try:
                with DiagnosticTrace(r,source,out/(name+'.ndjson'),policy=True):result=r.replay(source,profile,policy=model)
                row['result']={k:v for k,v in result.items() if k!='frames'};row['task_passed']=bool(result['passed']);(out/(name+'-trajectory.json')).write_text(json.dumps(result,default=r.json_default,allow_nan=False))
            except Exception as exc:row['error']=f'{type(exc).__name__}: {exc}'
            finally:r.policy_action=submit
            row['commands']=commands.report();row['unmodified_policy_task_passed']=row['task_passed'] and not commands.events;row['diagnostics']=summarize_trace(out/(name+'.ndjson'),r.IDS);report['rollouts'].append(row);save();del model;gc.collect();torch.cuda.empty_cache();print(name,'task',row['task_passed'],'error',row.get('error'),flush=True)
    except Exception as exc:report['error']=f'{type(exc).__name__}: {exc}';save();raise
    finally:save()
    return 0
if __name__=='__main__':raise SystemExit(main())
