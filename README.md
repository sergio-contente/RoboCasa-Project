# RoboCasa — Open Electric Kettle Lid

Behavioral Cloning experiments on the `OpenElectricKettleLid` atomic task in [RoboCasa](https://robocasa.ai), using the **PandaOmron** robot.

## Methods

Two BC variants were trained on 50 human expert episodes (5,146 transitions):

| Method | Policy | Loss |
|---|---|---|
| Simple BC | Deterministic MLP | MSE |
| Imitation BC | ActorCritic (SB3) | NLL + entropy |

## Results

Both models converged in training but achieved **0% success rate** on 20 evaluation episodes due to covariate shift.

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
