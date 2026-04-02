# RoboCasa-Project
Project for CSC_5IA05_TA-Apprentissage pour la robotique

## Which files does what ?
**TODO:** This part must be filled before we send the project to the teacher so she isn't lost.

## How to setup from git ?
After you have clone the repository, you will need to download `roboacasa` and `robosuite`.
For that, please run the following commands in the root folder of this project:
```bash
git submodule init
git submodule update --remote
```

Then, to setup the virtual environment you have two choices.
- Either you use [UV](https://docs.astral.sh/uv/), in that case:
  + If you're planning to use a CPU environment for pytorch, comment these lines in `pyproject.toml`:
    ```toml
    { index = "pytorch-cu130" }
    ```
    and
    ```toml
    [[tool.uv.index]]
    name = "pytorch-cu130"
    url = "https://download.pytorch.org/whl/cu130"
    explicit = true
    ```
    And uncomment these lines:
    ```toml
    { index = "pytorch-cpu" }
    ```
    and
    ```toml
    [[tool.uv.index]]
    name = "pytorch-cpu"
    url = "https://download.pytorch.org/whl/cpu"
    explicit = true
    ```
  + And then, please run:
```bash
uv sync

# Install the robocasa package and download assets
uv run python -m robocasa.scripts.setup_macros              # Set up system variables.
uv run python -m robocasa.scripts.download_kitchen_assets   # Caution: Assets to be downloaded are around 10GB.
```
- Or if you use `conda`, please run:
```bash
# Create the environment
conda create -c conda-forge -n robocasa python=3.11
conda activate robocasa

# Install robosuite
cd deps/robosuite
pip install -e .

# Install robocasa
cd ../robocasa
pip install -e .
pip install pre-commit; pre-commit install           # Optional: set up code formatter.

# Install the robocasa package and download assets
python -m robocasa.scripts.setup_macros              # Set up system variables.
python -m robocasa.scripts.download_kitchen_assets   # Caution: Assets to be downloaded are around 10GB.
```

To make sure your environment is properly set up, please run the sample code
provided by the robocasa team:
```py
import gymnasium as gym
import robocasa
from robocasa.utils.env_utils import run_random_rollouts

env = gym.make(
    "robocasa/PickPlaceCounterToCabinet",
    split="pretrain", # use 'pretrain' or 'target' kitchen scenes and objects
    seed=0 # seed environment as needed. set seed=None to run unseeded
)

# run rollouts with random actions and save video
run_random_rollouts(
    env, num_rollouts=3, num_steps=100, video_path="test.mp4"
)
```

If everything works, you should have a video named `test.mp4`
of a robot shaking in various environments.
