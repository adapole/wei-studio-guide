"""Convert a prepared causal bundle using LeRobot's public Dataset writer on the training machine."""
import argparse,json,importlib.metadata
from pathlib import Path
import numpy as np

def convert(bundle,out,split,repo_id,allow_unqualified=False,max_segments=None):
 from lerobot.datasets.lerobot_dataset import LeRobotDataset
 manifest=json.loads((bundle/'manifest.json').read_text())
 if not manifest['training_eligible'] and not allow_unqualified:raise ValueError('Unqualified bundle is restricted to explicit smoke tests')
 features={key:{'dtype':'float32','shape':(len(names),),'names':names} for key,names in manifest['features'].items()}
 ds=LeRobotDataset.create(repo_id=repo_id,root=out,fps=60,robot_type='orca-v2-right',features=features,use_videos=False)
 selected=[e for e in manifest['episodes'] if e['split']==split]
 if max_segments:selected=selected[:max_segments]
 if not selected:raise ValueError('Empty split')
 try:
  for ep in selected:
   data=np.load(bundle/ep['file'])
   for state,environment,action in zip(data['state'],data['environment'],data['action']):ds.add_frame({'observation.state':state,'observation.environment_state':environment,'action':action,'task':'Lift block 5cm, hold two seconds, release into tray and settle'})
   ds.save_episode()
 finally:ds.finalize()
 (out/'orca-provenance.json').write_text(json.dumps({'bundle_manifest':manifest,'selected':selected,'lerobot_version':importlib.metadata.version('lerobot'),'smoke_only':not manifest['training_eligible']},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('bundle',type=Path);p.add_argument('output',type=Path);p.add_argument('--split',choices=['train','validation','test'],default='train');p.add_argument('--repo-id',default='local/orca-block');p.add_argument('--allow-unqualified',action='store_true');p.add_argument('--max-segments',type=int);a=p.parse_args();convert(a.bundle,a.output,a.split,a.repo_id,a.allow_unqualified,a.max_segments)
