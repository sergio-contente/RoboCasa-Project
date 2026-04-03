# RoboCasa — Open Electric Kettle Lid

Behavioral Cloning, GAIL, and RL experiments on the `OpenElectricKettleLid` atomic task in [RoboCasa](https://robocasa.ai), using the **PandaOmron** robot.

## Methods

### Imitation Learning — Behavioral Cloning

Two BC variants were trained on 50 human expert episodes (5,146 transitions):

| Method | Policy | Loss |
|---|---|---|
| Simple BC | Deterministic MLP | MSE |
| Imitation BC | ActorCritic (SB3) | NLL + entropy |

### Imitation Learning — GAIL

A **Generative Adversarial Imitation Learning** approach using the `imitation` library. A PPO generator is trained adversarially against a discriminator that classifies (obs, action) pairs as expert or learner. Unlike BC, GAIL interacts with the environment during training, making it more robust to distribution shift.

| Method | Policy | Reward signal |
|---|---|---|
| GAIL | PPO + MLP [256, 256] | Discriminator |

### Reinforcement Learning — PPO + Curriculum *(separate branch)*

A PPO agent trained **from scratch** (no demonstrations) with dense reward shaping and a 3-stage curriculum. Available on the `ppo-curriculum` branch.

| Stage | Layout | Style |
|---|---|---|
| 0 | Fixed | Fixed |
| 1 | Random | Fixed |
| 2 | Random | Random |

## Results

All approaches achieved **0% success rate (except for GAIL with action chunking - SR = 10%)** in evaluation. The task requires pressing a 2mm button, which demands extreme precision none of the policies could consistently achieve.

But we did get some good runs on other approaches...

![demo](demo.gif)

## How to run

### Simple BC
```python
# Train
python train_bc.py --episodes 50 --epochs 100 --batch-size 256 --lr 3e-4

# Evaluate
from common import load_bc_checkpoint
policy = load_bc_checkpoint('results/bc_model.pt')
```

### Imitation BC

Open and run `Colab_ImitationBC.ipynb` on Google Colab (T4 GPU recommended).

```python
# Reload trained policy
import torch
from stable_baselines3.common.policies import ActorCriticPolicy

ckpt = torch.load('results/imitation_bc_policy.pt', map_location='cpu')
policy = ActorCriticPolicy(
    observation_space=env.observation_space,
    action_space=env.action_space,
    lr_schedule=lambda _: 3e-4,
    net_arch=[256, 256],
)
policy.load_state_dict(ckpt['state_dict'])
```

### GAIL

Open and run `Colab_BC.ipynb` on Google Colab (T4/H100 GPU recommended).

```python
# Reload trained generator
from stable_baselines3 import PPO
model = PPO.load('results/gail_model')
action, _ = model.predict(obs, deterministic=True)
```

### PPO + Curriculum Learning

Switch to the `ppo-curriculum` branch and follow the instructions there.

```bash
git checkout ppo-curriculum
python train_ppo.py --timesteps 500000
```

## Installation

```bash
# robosuite
git clone https://github.com/ARISE-Initiative/robosuite.git
cd robosuite && git checkout aaa8b9b && pip install -e .

# robocasa
git clone https://github.com/robocasa/robocasa.git
cd robocasa && git checkout 0f59111 && pip install -e .
python -m robocasa.scripts.setup_macros
python -m robocasa.scripts.download_kitchen_assets
```

## Output files

| File | Description |
|---|---|
| `results/bc_model.pt` | Simple BC weights |
| `results/imitation_bc_policy.pt` | Imitation BC weights + metadata |
| `results/bc_losses.json` | Simple BC training losses |
| `results/imitation_bc_eval.json` | Evaluation metrics |
| `results/gail_model.zip` | GAIL PPO generator (SB3 format) |
| `results/gail_eval.json` | GAIL evaluation metrics |
