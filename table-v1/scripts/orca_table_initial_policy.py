"""Episode-initial block frame: causal translation context, with no moving-object target feedback."""
import hashlib
import json
from pathlib import Path
import numpy as np

REPRESENTATION='table-initial-block-world-axes-and-elapsed-v3'


def features(raw,tick,anchor):
    state=np.asarray(raw['observation.state'],np.float32).copy();environment=np.asarray(raw['observation.environment_state'],np.float32).copy();anchor=np.asarray(anchor,np.float32)
    if state.shape!=(47,) or environment.shape!=(14,) or anchor.shape!=(3,) or not np.isfinite(np.r_[state,environment,anchor]).all():raise ValueError('Invalid native measurements or episode anchor')
    if type(tick) is not int or tick<0:raise ValueError('Expected causal nonnegative policy tick')
    state[34:37]-=anchor;environment[:3]-=anchor;environment[7:10]-=anchor
    return {'observation.state':np.r_[state,np.float32(tick/60)].astype(np.float32),'observation.environment_state':environment}


def encode(action,anchor):
    value=np.asarray(action,dtype=np.float64).copy();anchor=np.asarray(anchor,dtype=np.float64)
    if value.shape!=(24,) or anchor.shape!=(3,) or not np.isfinite(np.r_[value,anchor]).all():raise ValueError('Invalid native command or episode anchor')
    value[17:20]-=anchor;return value


def decode(action,anchor):
    value=np.asarray(action,dtype=np.float64).copy();anchor=np.asarray(anchor,dtype=np.float64)
    if value.shape!=(24,) or anchor.shape!=(3,) or not np.isfinite(np.r_[value,anchor]).all():raise ValueError('Invalid encoded command or episode anchor')
    value[17:20]+=anchor;return value


class EpisodeFrame:
    def __init__(self):self.anchor=None
    def features(self,raw,tick):
        if self.anchor is None:
            if tick!=0:raise ValueError('An episode anchor must come from the first observation')
            self.anchor=np.asarray(raw['observation.environment_state'],np.float32)[:3].copy()
        return features(raw,tick,self.anchor)
    def encode(self,action,_):
        if self.anchor is None:raise ValueError('Observe before encoding commands')
        return encode(action,self.anchor)


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
    def reset(self):self.tick=0;self.raw_encoded=[];self.frame=EpisodeFrame();self.policy.reset()
    def __call__(self,raw):
        batch={k:self.torch.as_tensor(v,dtype=self.torch.float32).unsqueeze(0) for k,v in self.frame.features(raw,self.tick).items()}
        with self.torch.inference_mode():value=self.postprocess(self.policy.select_action(self.preprocess(batch))).detach().cpu().numpy().reshape(-1)
        self.tick+=1;self.raw_encoded.append(np.asarray(value).copy());return decode(value,self.frame.anchor)
