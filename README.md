# Wei Studio guide

Wei Studio records human demonstrations with Quest hand tracking. This guide starts with a prepared LeRobot dataset and ends with a robot hand acting under a learned policy in simulation.

## Train a LeRobot ACT policy and run it in MuJoCo

Use a Linux training machine with the Wei Studio simulation environment activated, including LeRobot, CUDA-enabled PyTorch, MuJoCo and the calibrated ORCA model assets. Run the commands from the Wei Studio **application checkout** containing `scripts/orca-policy-rollout.py`. This guide repository contains documentation. Supply your own prepared dataset and a compatible Wei recording for the initial scene. If you already have a checkpoint, skip training.

### Train from a prepared dataset

Set the local dataset directory and its LeRobot repository ID. Choose a fresh output directory. This example uses the compact state-based ACT architecture from the Wei pilot training script. The step count is an example training budget.

```bash
DATASET=/absolute/path/to/prepared-dataset
DATASET_ID=local/orca-qualified-pilot-train
TRAIN="$PWD/outputs/my-act-run"

lerobot-train \
  --dataset.repo_id="$DATASET_ID" --dataset.root="$DATASET" \
  --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
  --policy.pretrained_backbone_weights=null --policy.chunk_size=30 \
  --policy.n_action_steps=1 --policy.dim_model=128 --policy.n_heads=4 \
  --policy.dim_feedforward=512 --policy.n_encoder_layers=2 --policy.use_vae=false \
  --batch_size=4 --num_workers=0 --seed=7 --steps=5000 --save_freq=5000 \
  --output_dir="$TRAIN"

find "$TRAIN/checkpoints" -type d -name pretrained_model
```

### Run a checkpoint in MuJoCo

Choose the printed checkpoint directory containing the weights, `config.json` and saved processors. Wei's Python adapter loads it directly. MuJoCo supplies the simulated hand and object states, the adapter normalizes those inputs, ACT selects an action, and the adapter converts that action back to the controller's units. Each new observation comes from the running simulation. There is no checkpoint-to-XML conversion or desktop import step.

The supplied single-rollout adapter expects the Wei ORCA feature layout: 47 hand-state values, 14 environment-state values and 24 action values with the same ordering and units as the training data. Pair the recording with the model, task and calibration supported by your runner.

```bash
CHECKPOINT=/absolute/path/to/pretrained_model
EPISODE=/absolute/path/to/compatible-wei-recording.json
RUN="$PWD/outputs/my-act-rollout"

python scripts/orca-policy-rollout.py "$EPISODE" "$CHECKPOINT" \
  --device cuda --seconds 15 --output "$RUN"
cat "$RUN/report.json"
```

Use a new output directory for every run. The recording supplies the task setup. The checkpoint supplies the actions. The script writes a headless trajectory to `trajectory.json` on completion. In `report.json`, `pipeline_passed` means inference and simulation completed. Read `task_passed`, `gates` and `first_failing_stage` to see whether the grasp and placement succeeded. If initialization fails, read `error` and check the checkpoint, model assets and calibration fingerprints.

## Tabletop pick-and-place results

The table-v1 policy picked up the block and placed it in the tray in **84 of 100 closed-loop MuJoCo trials**. Training used two original human demonstrations recorded in Wei Studio and 14 qualified synthetic variations, for 16 training episodes. Two separate validation takes were excluded from training.

The evaluation used 20 previously unseen nominal block placements with five seeded variations per placement. Variations changed block orientation and applied small position or velocity perturbations. The table, tray, controller, standardized initial hand posture and 15-second time limit were fixed. The result measures simulated performance within that setup.

The command above demonstrates a single compatible rollout. Reproducing the reported aggregate requires the table-v1 evaluation runner, selected checkpoint and original trial schedule from the GPU experiment.
