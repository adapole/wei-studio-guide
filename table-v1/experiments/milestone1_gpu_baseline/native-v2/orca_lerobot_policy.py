"""Local ACT checkpoint and saved processor adapter; no downloads or hardware access."""
from pathlib import Path


class ACTCheckpoint:
    def __init__(self, checkpoint, device='cpu', *, action_steps=None, temporal_ensemble_coeff=None):
        import torch
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors

        path = Path(checkpoint).resolve()
        if not path.is_dir() or not (path / 'config.json').is_file():
            raise ValueError('Provide a local complete pretrained_model checkpoint directory')
        self.torch = torch
        self.device = device
        config = PreTrainedConfig.from_pretrained(path, local_files_only=True)
        config.device = device
        if action_steps is not None:
            if not 1<=action_steps<=config.chunk_size:raise ValueError('Action steps must be within checkpoint chunk size')
            config.n_action_steps=action_steps
        if temporal_ensemble_coeff is not None:
            if config.n_action_steps!=1:raise ValueError('Temporal ensembling requires one action per call')
            config.temporal_ensemble_coeff=temporal_ensemble_coeff
        self.inference_recipe={'n_action_steps':config.n_action_steps,'temporal_ensemble_coeff':config.temporal_ensemble_coeff,'checkpoint_weights_modified':False}
        self.policy = ACTPolicy.from_pretrained(path, config=config, local_files_only=True, strict=True)
        self.policy.eval()
        self.preprocess, self.postprocess = make_pre_post_processors(
            policy_cfg=config, pretrained_path=path,
            preprocessor_overrides={'device_processor': {'device': device}},
        )
        expected = {'observation.state': (47,), 'observation.environment_state': (14,)}
        if {key: tuple(value.shape) for key, value in config.input_features.items()} != expected:
            raise ValueError('Checkpoint input features do not match ORCA state contract')
        if tuple(config.output_features['action'].shape) != (24,):
            raise ValueError('Checkpoint action features do not match ORCA contract')
        self.policy.reset()

    def __call__(self, features):
        batch = {key: self.torch.as_tensor(value, dtype=self.torch.float32).unsqueeze(0)
                 for key, value in features.items()}
        with self.torch.inference_mode():
            result = self.postprocess(self.policy.select_action(self.preprocess(batch)))
        return result.detach().cpu().numpy().reshape(-1)
