# Wei Studio | Train an Orca hand and test it in MuJoCo

Start with a Wei Studio **Table-v1 block pick-and-place** LeRobot export. The scripts train a state-based ACT policy on whole demonstrations and test the checkpoint on a simulated right Orca hand. You can run this workflow locally after recording, without a headset or a running Studio server.

## Train a LeRobot ACT policy and run it in MuJoCo

### Reproduce the successful Table-v1 policy

Two held-out rollouts passed the 15-second MuJoCo pick-and-place test. The block rose 18.3 cm and 9.0 cm, respectively. Both trials used the same nominal block placement and different recorded starts. [Download the complete ACT checkpoint](table-v1/checkpoint.zip?raw=true) · [Watch the measured rollout](table-v1/held-out-rollout.mp4)

Run the commands below in a Linux or WSL2 shell with Git, tar, unzip and sha256sum installed. The successful run used Python 3.13, PyTorch 2.15.0, an NVIDIA RTX 4070 Laptop GPU, and CUDA. Install the corresponding CUDA-enabled PyTorch build and a compatible driver. The frozen evaluator also uses CUDA. The training and calibration archives are distributed separately. Obtain inputs.tar.gz and orca-runtime.tar.gz from the project owner and place them in table-v1/ before running prepare. The commands below verify the artifacts, extract the exact processed dataset and validation fixtures, then score the published weights.

```bash
git clone https://github.com/adapole/wei-studio-guide.git
cd wei-studio-guide
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r table-v1/requirements.txt
python -c 'import torch; assert torch.cuda.is_available(), "CUDA is required"'
git clone https://github.com/orcahand/orca_sim.git models/orca_sim
git -C models/orca_sim checkout --detach 05e00dafd0a6eb2186d69d4e351c252a7bcffadd
bash table-v1/run.sh prepare
bash table-v1/run.sh evaluate
```

To train from scratch and score the resulting checkpoint, continue in the same environment. Use separate output directories for the two evaluations.

```bash
bash table-v1/run.sh train
bash table-v1/run.sh evaluate "$PWD/table-v1/outputs/train/training/checkpoints/020000/pretrained_model" "$PWD/table-v1/outputs/retrained-validation"
```

The 60 Hz dataset contains two original demonstrations and 14 qualified variations for training. The two validation demonstrations are excluded. The training adapter expresses positions relative to the initial block position and includes elapsed time. ACT trained with seed 7, batch size 32, learning rate 0.0001, environment normalization, and 20,000 optimizer steps. The evaluator requests one action per inference with temporal ensembling and blends from the measured initial hand pose for 0.2 seconds. It scores contact-supported lift, sustained hold, release and settling, joint limits, and penetration. Read the per-trial gates in the output directory's report.json. [Reference scores and checkpoint hashes](table-v1/reference-results.json) are included for comparison. The reference model.safetensors SHA-256 is `a55228f140ce6f12edc18c89cf26534c7b76a798e0b57bba0ab1b17619e8a74b`. Different hardware or software can change the trained weights.

### Train on your own Studio export

The following workflow uses a state-only 50 Hz Studio export. Its checkpoint uses a different feature layout from the successful 60 Hz recipe above.

#### Install on Linux or WSL2

Run these commands from the root of this guide. Install Git, Python 3.12 or 3.13 with venv support, and FFmpeg (for MP4 output). For headless video, install an EGL graphics driver or Mesa EGL. Training and headless simulation can both use CPU. CUDA training requires a compatible PyTorch installation and driver.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

git clone https://github.com/orcahand/orca_sim.git models/orca_sim
git -C models/orca_sim checkout --detach 05e00dafd0a6eb2186d69d4e351c252a7bcffadd
```

### Inspect the export

In Studio Settings, select evaluated takes for the same task and robot, export the LeRobot dataset, and download the ZIP. Extract the Studio dataset ZIP. Set DATA to the directory containing wei-export.json, wei-manifest.json, meta/ and data/. Keep the original ZIP for provenance.

```bash
mkdir -p data/my-export
unzip my-studio-export.zip -d data/my-export
DATA="$(dirname "$(find "$PWD/data/my-export" -type f -name wei-export.json -print -quit)")"
python scripts/wei.py inspect "$DATA"
```

Inspect checks the manifest and feature layout and lists the recording outcomes. The supported observation has 47 hand-state and 14 environment values. The action contains 24 values. These widths apply to the Table-v1 block rollout. For apple or can training, export each object separately and use the same inspect and train commands below. For connector training, use a state-only export that includes both hands in the observation and action vectors and both connector poses in the environment state. Inspect the feature names in wei-export.json before training. ACT reads the widths from the export. If those channels are missing, use the native recording workflow below with a take containing both simulated hands. Closed-loop scoring currently covers the right-hand Table-v1 block task.

### Train ACT

```bash
python scripts/wei.py train "$DATA" --device cpu --steps 5000 --output outputs/act-01
```

For a quick installation check, use --steps 2 and the printed checkpoint path. Use --device cuda for GPU training. By default the trainer selects the episodes marked passed in the export. To choose demonstrations explicitly, append e.g. --episodes 0 2. For an independent evaluation, export held-out demonstrations separately. LeRobot uses the normalization statistics of the training export. Keep variations of each source demonstration in the same split. Pass the held-out dataset directory to evaluate. Training writes training-input.json with source episode metadata, the input hash and the command, and prints the checkpoint path. Keep the complete pretrained_model directory including its processors.

### Test in closed loop

```bash
CHECKPOINT="$PWD/outputs/act-01/training/checkpoints/005000/pretrained_model"
MUJOCO_GL=egl python scripts/wei.py evaluate "$DATA" --episode 0 \
  --checkpoint "$CHECKPOINT" --seconds 15 --video --output outputs/rollout-01
cat outputs/rollout-01/report.json
```

Choose a new output directory for each run. Omit --video to simulate without a graphics context or FFmpeg. The episode supplies the initial hand and block state and the tray pose. The block starts at rest because the export omits its velocity. Every policy decision uses a fresh simulated observation at the export frame rate. Commands are bounded by the exported joint ranges. A per-take VR restriction must be present in those ranges to carry into this test.

Read pipeline_passed for execution status and task_passed for the manipulation outcome. Inspect gates and trajectory.json to diagnose contact, lift, release and final placement. rollout.mp4 visualizes the measured trajectory, including misses. Testing the same single demonstration used for training is a fit-and-rollout diagnostic. To measure generalization, evaluate held-out starts and count every completed trial.

### Train from an apple, can or connector recording

If Studio supplies a native episode JSON instead of a LeRobot ZIP, use the recorded-task converter. Select takes of the same task. Read state.object.id and state.custom_objects[].id in the first frame for the object identifiers. Set PLUG_ID and SOCKET_ID to the two connector IDs. For apple or can, use --object with the manipulated object's ID and add --object tray for the placement target, omit --two-hands for a right-hand take and choose a separate output directory for each task.

```bash
python -c 'import json; s=json.load(open("connector-take.json"))["frames"][0]["state"]; print([o["id"] for o in [s["object"], *s.get("custom_objects", [])]])'
# Set PLUG_ID and SOCKET_ID to the printed IDs before running:
python scripts/recorded_task.py connector-take.json --two-hands \
  --object "$PLUG_ID" --object "$SOCKET_ID" --task "Insert the plug into the socket" \
  --output data/connector
python -m lerobot.scripts.lerobot_train --dataset.repo_id=local/wei-recorded-task \
  --dataset.root="$PWD/data/connector" --policy.type=act --policy.device=cpu \
  --policy.push_to_hub=false --policy.pretrained_backbone_weights=null \
  --policy.chunk_size=30 --policy.n_action_steps=1 --batch_size=4 --num_workers=0 \
  --steps=5000 --output_dir="$PWD/outputs/connector-act"
```

The converter pairs each achieved hand and object observation with the next recorded applied command. It samples on simulation time, keeps each demonstration in its own episode and records source hashes. Select demonstrations you want ACT to imitate. This path does not assign success labels. A connector rollout requires a separate bimanual MuJoCo environment and a seating test after release.

### Verified export walkthrough

A real Quest block take passed the Studio evaluator and both native replays. The Studio writer exported 593 frames at 50 Hz and verified every frame against the source. Training the example ACT configuration on that one demonstration for 5,000 CPU steps produced a checkpoint. Its 15-second closed-loop test ran 750 policy decisions and missed the lift. This single-demonstration result is separate from the multi-demonstration benchmark below.

## Tabletop pick-and-place results

The table-v1 policy picked up the block and placed it in the tray in **84 of 100 closed-loop MuJoCo trials**. Training used two original human demonstrations recorded in Wei Studio and 14 qualified synthetic variations, for 16 training episodes. Two separate validation takes were excluded from training.

The evaluation used 20 previously unseen nominal block placements with five seeded variations per placement. Variations changed block orientation and applied small position or velocity perturbations. The table, tray, controller, standardized initial hand posture and 15-second time limit were fixed. The result measures simulated performance within that setup.

The command above demonstrates a single compatible rollout. Reproducing the reported aggregate requires the table-v1 evaluation runner, selected checkpoint and original trial schedule from the GPU experiment.
