from collections import defaultdict
from typing import Optional

import numpy as np
import torch
#import tqdm
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule

from torchrl.data import BoundedTensorSpec, CompositeSpec, UnboundedContinuousTensorSpec
from torchrl.envs import (
    CatTensors,
    EnvBase,
    Transform,
    TransformedEnv,
    UnsqueezeTransform,
)
from torchrl.envs.transforms.transforms import _apply_to_composite
from torchrl.envs.utils import check_env_specs, step_mdp

class CustomEnv(EnvBase):
    def __init__(self, observation_space, action_space, device="cpu"):
        # Call the parent class's constructor to ensure proper initialization
        super().__init__(device=device, batch_size=[])
        
        # Custom initialization
        self.observation_space = observation_space
        self.action_space = action_space
        self.state = None

    def _reset(self):
        # Define the initial state of the environment
        self.state = torch.zeros(self.observation_space.shape)
        
        # Create a TensorDict to return the initial observation
        obs = TensorDict({"observation": self.state}, batch_size=[1])
        
        return obs

    def _step(self, action):
        # Define the logic to update the state based on the action
        self.state += action
        
        # Define the reward
        reward = torch.tensor([1.0])
        
        # Define whether the episode is done
        done = torch.tensor([False])
        
        # Create a TensorDict for the next state, reward, and done flag
        obs = TensorDict({
            "observation": self.state,
            "reward": reward,
            "done": done
        }, batch_size=[1])
        
        return obs

    def _set_seed(self, seed:Optional[int]):
        rng = torch.manual_seed(seed)
        self.rng = rng
        
    def render(self):
        # Optional: Define a render method to visualize the environment
        print(f"Current state: {self.state}")

    def close(self):
        # Optional: Define a close method to clean up resources
        pass

observationSpace = torch.tensor([0, 0], dtype=torch.float32)
actionSpace = torch.tensor([0], dtype=torch.int64)
env = CustomEnv(observation_space=observationSpace, action_space=actionSpace)
check_env_specs(env)
print("observation_spec:", env.observation_spec)
print("state_spec:", env.state_spec)
print("reward_spec:", env.reward_spec)