"""Tested translation-relative table features/targets with a causal elapsed-policy clock."""
import hashlib
import json
from pathlib import Path
import numpy as np

REPRESENTATION='table-block-relative-xyz-and-elapsed-v1'


def features(raw,tick):
    state=np.asarray(raw['observation.state'],dtype=np.float32).copy();environment=np.asarray(raw['observation.environment_state'],dtype=np.float32).copy()
    if state.shape!=(47,) or environment.shape!=(14,) or not np.isfinite(np.r_[state,environment]).all():raise ValueError('Expected finite native table observations')
    if type(tick) is not int or tick<0:raise ValueError('Expected causal nonnegative policy tick')
    origin=environment[:3].copy();state[34:37]-=origin;environment[:3]-=origin;environment[7:10]-=origin
    return {'observation.state':np.r_[state,np.float32(tick/60)].astype(np.float32),'observation.environment_state':environment}


def encode(action,raw_environment):
    action=np.asarray(action,dtype=np.float64).copy();environment=np.asarray(raw_environment,dtype=np.float64)
    if action.shape!=(24,) or environment.shape!=(14,) or not np.isfinite(np.r_[action,environment]).all():raise ValueError('Invalid native action/environment')
    action[17:20]-=environment[:3];return action


def decode(action,raw_environment):
    action=np.asarray(action,dtype=np.float64).copy();environment=np.asarray(raw_environment,dtype=np.float64)
    if action.shape!=(24,) or environment.shape!=(14,) or not np.isfinite(np.r_[action,environment]).all():raise ValueError('Invalid encoded action/environment')
    action[17:20]+=environment[:3];return action


class TableACTCheckpoint:
    def __init__(self,checkpoint,device='cuda',**options):
        import torch
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors
        checkpoint=Path(checkpoint);marker=json.loads((checkpoint/'orca-table-representation.json').read_text())
        if marker['representation']!=REPRESENTATION or marker['adapter_sha256']!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest():raise ValueError('Table checkpoint representation/adapter mismatch')
        self.torch=torch;config=PreTrainedConfig.from_pretrained(checkpoint,local_files_only=True);config.device=device
        config.n_action_steps=options.get('action_steps',1);config.temporal_ensemble_coeff=options.get('temporal_ensemble_coeff',.01)
        if config.n_action_steps!=1:raise ValueError('Table inference requires one fresh causal observation per command')
        expected={'observation.state':(48,),'observation.environment_state':(14,)}
        if {k:tuple(v.shape) for k,v in config.input_features.items()}!=expected or tuple(config.output_features['action'].shape)!=(24,):raise ValueError('Checkpoint does not match table feature contract')
        self.policy=ACTPolicy.from_pretrained(checkpoint,config=config,local_files_only=True,strict=True);self.policy.eval()
        self.preprocess,self.postprocess=make_pre_post_processors(policy_cfg=config,pretrained_path=checkpoint,preprocessor_overrides={'device_processor':{'device':device}})
        self.inference_recipe={'n_action_steps':1,'temporal_ensemble_coeff':config.temporal_ensemble_coeff,'checkpoint_weights_modified':False};self.reset()
    def reset(self):self.tick=0;self.raw_encoded=[];self.policy.reset()
    def __call__(self,raw):
        batch={k:self.torch.as_tensor(v,dtype=self.torch.float32).unsqueeze(0) for k,v in features(raw,self.tick).items()}
        with self.torch.inference_mode():value=self.postprocess(self.policy.select_action(self.preprocess(batch))).detach().cpu().numpy().reshape(-1)
        self.tick+=1;self.raw_encoded.append(np.asarray(value).copy())
        return decode(value,raw['observation.environment_state'])
