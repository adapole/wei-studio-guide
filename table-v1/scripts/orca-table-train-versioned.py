"""Train any explicitly hash-bound table representation using qualified data only."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from orca_table_task import digest
from orca_table_representation import adapter


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['workspace','bundle','dataset','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--steps',type=int,default=20000);p.add_argument('--resume-from',type=Path)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--normalize-environment',action='store_true')
    p.add_argument('--repo-id',default='local/orca-table-initial-v3');a=p.parse_args();scripts=Path(__file__).parent;m=json.loads((a.bundle/'manifest.json').read_text());module=adapter(m)
    cmd=[sys.executable,str(scripts/'orca-train-qualified.py')]
    for key in ['workspace','bundle','dataset','output','seed','steps','batch_size']:cmd+=['--'+key.replace('_','-'),str(getattr(a,key))]
    cmd+=['--repo-id',a.repo_id]
    if a.resume_from:cmd+=['--resume-from',str(a.resume_from)]
    if a.normalize_environment:cmd+=['--normalize-environment']
    runner_sha=digest(__file__);completed=subprocess.run(cmd)
    if completed.returncode:raise SystemExit(completed.returncode)
    report=json.loads((a.output/'training-report.json').read_text());checkpoint=Path(report['checkpoint'])
    marker={'representation':module.REPRESENTATION,'adapter_sha256':m['table_adapter_sha256'],'bundle_manifest_sha256':digest(a.bundle/'manifest.json'),'calibration_sha256':m['calibration_sha256']}
    (checkpoint/'orca-table-representation.json').write_text(json.dumps(marker,indent=2))
    report.update(table_representation=module.REPRESENTATION,table_adapter_sha256=m['table_adapter_sha256'],table_runner_sha256=runner_sha,calibration_sha256=m['calibration_sha256'],checkpoint_hashes={f.name:digest(f) for f in checkpoint.iterdir() if f.is_file()})
    (a.output/'table-training-report.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
