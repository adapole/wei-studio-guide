"""Fault-preserving instrumented replay of the immutable calibrated baseline."""
import argparse,gc,hashlib,importlib.util,json
from pathlib import Path
import numpy as np
from orca_diagnostic_trace import DiagnosticTrace
from orca_trace_summary import summarize_trace

ROOT=Path(__file__).resolve().parents[1]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load_replay():
    p=ROOT/'scripts/orca-superdex-validate.py';s=importlib.util.spec_from_file_location('replay',p);r=importlib.util.module_from_spec(s);s.loader.exec_module(r);return r
def verify_calibration(r):
    path=ROOT/'artifacts/superdex/mujoco-contact-calibration-v3.json';cal=json.loads(path.read_text())
    if not(cal['passed'] and cal['dynamic_base_contact_passed'] and cal['source_model_fingerprint']==r.model_fingerprint() and cal['asset_fingerprint']==r.asset_fingerprint() and cal['validator_sha256']==digest(ROOT/'scripts/orca-superdex-validate.py')):raise ValueError('Frozen calibration/model/validator mismatch')
    return cal['selected_profile'],digest(path)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('episode',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--seconds',type=float);p.add_argument('--checkpoint',type=Path);p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--verify-observer',action='store_true');p.add_argument('--reference-dir',type=Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    report={'schema':'orca.trace.report.v1','scope':'simulation_only','pipeline_passed':False,'task_passed':False,'state_source':'mujoco_simulation','observer_verified':False}
    try:
        r=load_replay();profile,calhash=verify_calibration(r);source,sourcehash=r.load(a.episode)
        report.update(source_sha256=sourcehash,calibration_sha256=calhash,profile=profile,code_sha256={n:digest(ROOT/'scripts'/n) for n in ['orca-superdex-validate.py','orca_diagnostic_trace.py','orca_task_phases.py','orca-trace-replay.py','orca_trace_summary.py','orca_physics.py']})
        def policy():
            if not a.checkpoint:return None
            from orca_lerobot_policy import ACTCheckpoint
            report['checkpoint_hashes']={f.name:digest(f) for f in a.checkpoint.iterdir() if f.is_file()}
            return ACTCheckpoint(a.checkpoint,a.device)
        baseline=r.replay(source,profile,policy=policy(),duration_override=a.seconds) if a.verify_observer else None
        if a.reference_dir:
            if a.checkpoint or a.seconds is not None:raise ValueError('Saved reference comparison requires full recorded-command replay')
            reference=json.loads((a.reference_dir/'report.json').read_text())
            if reference['source_sha256']!=sourcehash or reference['calibration_sha256']!=calhash or not reference['repeatable']:raise ValueError('Reference provenance mismatch')
            reference_path=a.reference_dir/'trial-1.json';baseline=json.loads(reference_path.read_text());report['reference_trajectory_sha256']=digest(reference_path)
        gc.collect()
        with DiagnosticTrace(r,source,a.output/'trace.ndjson',policy=bool(a.checkpoint)) as trace:
            result=r.replay(source,profile,policy=policy(),duration_override=a.seconds)
        (a.output/'trajectory.json').write_text(json.dumps(result,default=r.json_default,allow_nan=False))
        report.update(pipeline_passed=True,task_passed=bool(result['passed']),result={k:v for k,v in result.items() if k!='frames'},phases=trace.phases.report(),trace_samples=trace.frames)
        if baseline is not None:
            equal=len(result['frames'])==len(baseline['frames']) and all(x==y for x,y in zip(result['frames'],baseline['frames'])) and result['gates']==baseline['gates']
            report['observer_verified']=equal
            if not equal:raise ValueError('Observed trajectory differs from baseline')
    except Exception as exc:report['error']=f'{type(exc).__name__}: {exc}';report['pipeline_passed']=False
    if (a.output/'trace.ndjson').exists():
        try:report['diagnostics']=summarize_trace(a.output/'trace.ndjson',r.IDS)
        except Exception as exc:report['diagnostics_error']=f'{type(exc).__name__}: {exc}'
    (a.output/'report.json').write_text(json.dumps(report,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist(),allow_nan=False))
    print(json.dumps({k:v for k,v in report.items() if k!='result'},indent=2));return 0 if report['pipeline_passed'] else 2
if __name__=='__main__':raise SystemExit(main())
