"""Train one explicitly versioned qualified split; no held-out data or uploads."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['workspace', 'bundle', 'dataset', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--steps', type=int, default=5000)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--resume-from', type=Path)
    p.add_argument('--repo-id',default='local/orca-qualified-six-v1')
    p.add_argument('--normalize-environment',action='store_true')
    a = p.parse_args()
    m = json.loads((a.bundle / 'manifest.json').read_text())
    if not m['training_eligible'] or not all(e['qualified'] and e.get('oracle_verified') for e in m['episodes']):
        raise ValueError('Requires qualified and independently oracle-verified entries')
    provenance = json.loads((a.dataset / 'orca-provenance.json').read_text())
    if any(e['split'] != 'train' for e in provenance['selected']):
        raise ValueError('Training data includes a non-training entry')
    if a.batch_size<1:raise ValueError('Batch size must be positive')
    host_storage=Path(__file__).resolve().parents[1]
    if min(shutil.disk_usage(a.workspace).free,shutil.disk_usage(host_storage).free) < 512 * 1024**2:
        raise OSError('Less than 512 MB host headroom; no training started')
    a.output.mkdir(parents=True, exist_ok=True)
    path = a.output / 'training-report.json'
    if path.exists() or (a.output / 'training').exists():
        raise ValueError('Preserve existing training attempt')
    cli = Path(sys.executable).parent / 'lerobot-train'
    cmd = [str(cli), f'--dataset.repo_id={a.repo_id}', f'--dataset.root={a.dataset.resolve()}',
        '--policy.type=act', '--policy.device=cuda', '--policy.push_to_hub=false', '--policy.pretrained_backbone_weights=null',
        '--policy.chunk_size=30', '--policy.n_action_steps=1', '--policy.dim_model=128', '--policy.n_heads=4',
        '--policy.dim_feedforward=512', '--policy.n_encoder_layers=2', '--policy.use_vae=false', f'--batch_size={a.batch_size}',
        f'--steps={a.steps}', '--num_workers=0', f'--seed={a.seed}', f'--output_dir={(a.output / "training").resolve()}',
        f'--save_freq={a.steps}', '--policy.optimizer_lr=0.0001']
    normalization_mapping={'VISUAL':'MEAN_STD','STATE':'MEAN_STD','ENV':'MEAN_STD','ACTION':'MEAN_STD'} if a.normalize_environment else None
    if normalization_mapping:
        cmd.append('--policy.normalization_mapping='+json.dumps(normalization_mapping,separators=(',',':')))
    if a.resume_from:
        config = a.resume_from / 'train_config.json'
        if not config.exists() or not (a.resume_from.parent / 'training_state/training_step.json').exists():
            raise ValueError('Resume requires complete saved model/optimizer/RNG state')
        previous_policy=json.loads((a.resume_from/'config.json').read_text())
        if json.loads(config.read_text())['batch_size']!=a.batch_size:
            raise ValueError('Resume must preserve the saved batch size')
        if (previous_policy.get('normalization_mapping',{}).get('ENV')=='MEAN_STD')!=a.normalize_environment:
            raise ValueError('Resume must preserve the saved ENV normalization mode')
        cmd += [f'--config_path={config.resolve()}', '--resume=true']
    report = {'schema': 'orca.qualified.training.v1', 'scope': 'qualified native-state simulation_only',
        'planned_seven_three_ready': m['planned_seven_three_ready'], 'milestone_1_complete': False,
        'seed': a.seed, 'steps': a.steps, 'batch_size':a.batch_size,'host_storage_guard_path':str(host_storage),'command': cmd, 'selection_revision': m['selection_revision'],
        'bundle_manifest_sha256': hashlib.sha256((a.bundle / 'manifest.json').read_bytes()).hexdigest(),
        'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'dependencies': {n: importlib.metadata.version(n) for n in ['torch', 'mujoco', 'numpy', 'lerobot']},
        'training_episode_count': len(provenance['selected']), 'training_passed': False,
        'validation_used_for_training': False, 'held_out_accessed': False}
    report['action_representation']=m.get('action_representation','native-absolute-v1')
    report['observation_representation']=m.get('observation_representation','current-native-state-v1')
    report['environment_normalization_enabled']=a.normalize_environment
    report['normalization_mapping_requested']=normalization_mapping
    if a.resume_from:
        report['resume_from'] = str(a.resume_from.resolve())
        report['resume_checkpoint_hashes'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in a.resume_from.iterdir() if p.is_file()}
        report['declared_change'] = f'Training duration only; resume saved optimizer and RNG to a fixed total of {a.steps}'
        report['resume_training_state_hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (a.resume_from.parent/'training_state').iterdir() if p.is_file()}
    path.write_text(json.dumps(report, indent=2))
    start = time.monotonic()
    with (a.output / 'training.log').open('x') as log:
        result = subprocess.run(cmd, cwd=a.workspace, stdout=log, stderr=subprocess.STDOUT)
    report.update(returncode=result.returncode, wall_seconds=time.monotonic()-start, training_passed=result.returncode == 0)
    checkpoint = a.output / 'training/checkpoints' / f'{a.steps:06d}/pretrained_model'
    if result.returncode == 0:
        if m.get('action_representation'):
            (checkpoint/'orca-action-representation.json').write_text(json.dumps({
                'representation':m['action_representation'], 'adapter_sha256':m['action_adapter_sha256'],
                'bundle_manifest_sha256':report['bundle_manifest_sha256']},indent=2))
        if m.get('observation_representation'):
            (checkpoint/'orca-observation-representation.json').write_text(json.dumps({
                'representation':m['observation_representation'],'frames':m['observation_history_frames'],
                'adapter_sha256':m['observation_adapter_sha256'],'bundle_manifest_sha256':report['bundle_manifest_sha256']},indent=2))
        report['checkpoint'] = str(checkpoint)
        report['checkpoint_hashes'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in checkpoint.iterdir() if p.is_file()}
    path.write_text(json.dumps(report, indent=2))
    print('Training passed', report['training_passed'], 'wall_seconds', report['wall_seconds'])
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
