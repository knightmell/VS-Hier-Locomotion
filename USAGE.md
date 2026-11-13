# Installation and usage

## Environment

The reference runtime is Linux, Python 3.9 and PyTorch 2.4.0. Create a separate
environment, install an appropriate CUDA build of PyTorch, then run from this
repository's root:

```bash
python -m pip install -r requirements.txt
python -m pip install -e vendor/stable-baselines3
python -m pytest
```

D4RL `*-medium-v2` uses legacy Gym and `mujoco-py`. Install its MuJoCo 2.1 runtime
and native build dependencies following the [DPPO MuJoCo installation guide](https://github.com/irom-princeton/dppo/blob/main/installation/install_mujoco.md).
The modern `mujoco` Python package alone does not replace this legacy dependency.
For headless execution, configure the renderer for the available platform.

## Pretrained assets

Download the original locomotion diffusion checkpoints and normalization files
from the [DSRL publisher's asset folder](https://drive.google.com/drive/folders/1kzC49RRFOE7aTnJh_7OvJ1K5XaDmtuh1).
Keep its original `gym-pretrain/` and `gym/` layout in a local folder, then run:

```bash
python scripts/prepare_assets.py --source /path/to/downloaded/folder
python scripts/prepare_assets.py --check
```

`assets_manifest.json` specifies the six expected files and their SHA-256 hashes.
The importer verifies each file before copying it into the ignored
`assets/pretrained/` directory. Checkpoints and training data are not bundled
with the source release and retain their original asset terms.

## Train

```bash
python train.py --config-name hopper seed=1
python train.py --config-name walker seed=1
python train.py --config-name halfcheetah seed=1
```

Each reference configuration explicitly specifies 800k online chunk transitions:
300k base adaptation followed by 500k refinement, with the configured 50k hold
and 50k ramp. Each chunk nominally executes four primitive actions. Prefill,
evaluation and actually executed primitive actions are accounted for separately.
The generator is frozen; the latent actor continues to learn during refinement.
Teacher sampling is Gaussian with four proposals per state, and latent actor
gradient clipping is disabled. Read the resolved config and run manifest for
the complete update settings:

```bash
python train.py --config-name hopper --cfg job --resolve
```

The old internal schedule-profile identifier remains for checkpoint compatibility;
the explicit phase lengths and 800k budget in the released YAML are authoritative.
Core algorithm, replay and schedule implementations are retained. Environment
dispatch, source paths and package exports are adapted for this standalone release.
The paper's scores are historical experimental results; this packaging pass does
not constitute a new multi-seed reproduction of those scores.

The matched DSRL runner is also included. For example:

```bash
python train.py --config-name hopper algorithm=dsrl_na_control p6.algorithm_label=dsrl_na_control name=hopper_s1_noclip_k4_300k_800k_basecontrol
```

This is the flat matched-DSRL baseline, not a same-checkpoint Base View or an
independently trained hierarchical Base Control. Avoid conflating these comparisons.

## Evaluate

```bash
python evaluate.py --task hopper --checkpoint runs/hopper_s1_noclip_k4_300k_800k_full/checkpoints/final_model.zip --episodes 100 --output results/hopper_s1.json
```

The default evaluates Full and Base View using environment seeds 10000 onward
and policy seeds 20000 onward. Base View disables residual execution in the same
trained checkpoint; it is not an independent training control. `--mode full` and
`--mode base` select individual views. `--device cpu` enables CPU inference.
Output contains per-episode records and aggregate metrics; existing files are
not overwritten. D4RL normalized scores are not success percentages.

## Checkpoints and resume

Periodic complete bundles contain model, replay and runtime/RNG state. Resume
uses a new environment reset boundary rather than pretending to restore an
unfinished simulator episode. Supply a complete bundle path explicitly:

```bash
python train.py --config-name hopper p6.resume_bundle_path=/path/to/complete/bundle
```

Use the same task, seed, phase settings, assets and output directory as the source
run. Model-only `.zip` files are suitable for inference, not exact training resume.
