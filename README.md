# Wei Studio | Train an Orca hand and test it in MuJoCo

Start with a Wei Studio **Table-v1 block pick-and-place** LeRobot export. The scripts train a state-based ACT policy on whole demonstrations and test the checkpoint on a simulated right Orca hand. You can run this workflow locally after recording, without a headset or a running Studio server.

## Install on Linux or WSL2

Run these commands from the root of this guide. Install Git, Python 3.12 or 3.13 with venv support, and FFmpeg (for MP4 output). For headless video, install an EGL graphics driver or Mesa EGL. Training and headless simulation can both use CPU. CUDA training requires a compatible PyTorch installation and driver.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

git clone https://github.com/orcahand/orca_sim.git models/orca_sim
git -C models/orca_sim checkout --detach 05e00dafd0a6eb2186d69d4e351c252a7bcffadd
```

## Inspect the export

Extract the Studio dataset ZIP. Set DATA to the directory containing wei-export.json, wei-manifest.json, meta/ and data/. Keep the original ZIP for provenance.

```bash
mkdir -p data/my-export
unzip my-studio-export.zip -d data/my-export
find data/my-export -name wei-export.json
DATA="$PWD/data/my-export"  # adjust if the archive contains a wrapper directory
python scripts/wei.py inspect "$DATA"
```

Inspect checks the manifest and feature layout and lists the recording outcomes. The supported observation has 47 hand-state and 14 environment values. The action contains 24 values. Apple, bag, connector, two-hand and custom-scene exports require their own dynamics and task scoring.

## Train ACT

```bash
python scripts/wei.py train "$DATA" --device cpu --steps 5000 --output outputs/act-01
```

Use --device cuda for GPU training. By default the trainer selects the episodes marked passed in the export. To choose demonstrations explicitly, append e.g. --episodes 0 2. For an independent evaluation, export held-out demonstrations separately. LeRobot uses the normalization statistics of the training export. Keep variations of each source demonstration in the same split. Pass the held-out dataset directory to evaluate. Training writes training-input.json with source episode metadata, the input hash and the command, and prints the checkpoint path. Keep the complete pretrained_model directory including its processors.

## Test in closed loop

```bash
CHECKPOINT="$PWD/outputs/act-01/training/checkpoints/005000/pretrained_model"
MUJOCO_GL=egl python scripts/wei.py evaluate "$DATA" --episode 0 \
  --checkpoint "$CHECKPOINT" --seconds 15 --video --output outputs/rollout-01
cat outputs/rollout-01/report.json
```

Choose a new output directory for each run. Omit --video to simulate without a graphics context or FFmpeg. The episode supplies the initial hand and block state and the tray pose. The block starts at rest because the export omits its velocity. Every policy decision uses a fresh simulated observation at the export frame rate. Commands are bounded by the exported joint ranges. A per-take VR restriction must be present in those ranges to carry into this test.

Read pipeline_passed for execution status and task_passed for the manipulation outcome. Inspect gates and trajectory.json to diagnose contact, lift, release and final placement. rollout.mp4 visualizes the measured trajectory, including misses. Testing the same single demonstration used for training is a fit-and-rollout diagnostic. To measure generalization, evaluate held-out starts and count every completed trial.

## Tabletop pick-and-place results

The table-v1 policy picked up the block and placed it in the tray in **84 of 100 closed-loop MuJoCo trials**. Training used two original human demonstrations recorded in Wei Studio and 14 qualified synthetic variations, for 16 training episodes. Two separate validation takes were excluded from training.

The evaluation used 20 previously unseen nominal block placements with five seeded variations per placement. Variations changed block orientation and applied small position or velocity perturbations. The table, tray, controller, standardized initial hand posture and 15-second time limit were fixed. The result measures simulated performance within that setup.

The command above demonstrates a single compatible rollout. Reproducing the reported aggregate requires the table-v1 evaluation runner, selected checkpoint and original trial schedule from the GPU experiment.
