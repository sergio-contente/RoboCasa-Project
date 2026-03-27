from environment_transformer import ActionObservationTransformer
import json
from pathlib import Path

import numpy as np
import pandas as pd
import robocasa
import robocasa.utils.dataset_registry
import robocasa.utils.lerobot_utils as LU

class DatasetManager:
    dataset_path: Path
    episodes_path: list[Path]
    env_metadata: dict

    def __init__(self, env_name: str, split: str, source: str):
        print(f"Loading the dataset")
        dataset_metadata = robocasa.utils.dataset_registry.get_ds_meta(
            env_name,
            split="pretrain",
            source="human",
            demo_fraction=1.0,
        )
        assert dataset_metadata is not None
        self.dataset_path = Path(dataset_metadata["path"])

        self.episodes_path = LU.get_episodes(self.dataset_path)
        self.env_metadata = LU.get_env_metadata(self.dataset_path)
        self.modality = json.load(
            open(self.dataset_path / "meta" / "modality.json", "r")
        ) 

    def get_episode_actions(self, index: int) -> tuple[dict, np.ndarray, str, list[dict[str, np.ndarray]]]:
        ## Cap the number of episodes to load
        assert index >= 0 and index < len(self.episodes_path)

        data_files = list(self.dataset_path.glob(f"data/*/episode_{index:06d}.parquet"))
        if not data_files:
            raise FileNotFoundError(f"No parquet file found for episode {index}")
        
        df = pd.read_parquet(data_files[0])

        output = []
        for _, row in df.iterrows():
            action = {}
            for action_name, action_info in self.modality["action"].items():
                key = action_info["original_key"]
                action[f"{key}.{action_name}"] = (
                    row[key][action_info["start"]: action_info["end"]]
                )
            output.append(action)
        
        return (
            LU.get_episode_meta(self.dataset_path, index),
            LU.get_episode_states(self.dataset_path, index)[0],
            LU.get_episode_model_xml(self.dataset_path, index),
            output
        )

def reset_based_on_episode(env: ActionObservationTransformer, ep_metadata: dict, model: str, initial_state_flatten: np.ndarray):
    # == Based on playback_dataset.do_reset function ==
    env.get_wrapper_attr("set_ep_meta")(ep_metadata)
    observation, _ = env.reset()

    xml = env.get_wrapper_attr("edit_model_xml")(model)
    env.get_wrapper_attr("reset_from_xml_string")(xml)
    
    sim = env.get_wrapper_attr("sim")
    sim.reset()
    
    sim.set_state_from_flattened(initial_state_flatten)
    sim.forward()
    # == ==
