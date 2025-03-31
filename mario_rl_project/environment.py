import gym
import numpy as np
import torch
from gym import spaces
from gym.wrappers import FrameStack
from nes_py.wrappers import JoypadSpace
from typing import Tuple, Any

import torchvision.transforms as T

import gym_super_mario_bros

class SkipFrame(gym.Wrapper):
    def __init__(self, env: gym.Env, skip: int = 4) -> None:
        super().__init__(env)
        self._skip = skip

    def step(self, action: int) -> Tuple[Any, float, bool, bool, dict]:
        total_reward = 0.0
        done = False
        trunc = False
        info = {}
      
        # Repeat the action for 'skip' frames
        for _ in range(self._skip):
            obs, reward, done, trunc, info = self.env.step(action)
            total_reward += reward
          
            if done:
                break
              
        return obs, total_reward, done, trunc, info


# Wrapper to convert RGB observations to grayscale.
class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        # Set the observation space to one channel (grayscale)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation: np.ndarray) -> torch.Tensor:
        # Change observation from [H, W, C] to [C, H, W]
        observation = np.transpose(observation, (2, 0, 1))
        return torch.tensor(observation.copy(), dtype=torch.float)

    def observation(self, observation: np.ndarray) -> torch.Tensor:
        observation = self.permute_orientation(observation)
        
        transform = T.Grayscale()
        observation = transform(observation)
      
        return observation


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, shape: int | Tuple[int, int] = 84) -> None:
        super().__init__(env)
        # Accept an integer or a tuple for the new shape
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = shape

        # Update observation space with new shape
        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation: torch.Tensor) -> torch.Tensor:
        transforms = T.Compose([
            T.Resize(self.shape, antialias=True),
            T.Normalize(0, 255)
        ])
      
        observation = transforms(observation).squeeze(0)
      
        return observation


def create_environment() -> gym.Env:
    """
    Create and return the Super Mario environment with all necessary wrappers.
    """
    # Use new_step_api or render_mode based on gym version
    if gym.__version__ < '0.26':
        env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", new_step_api=True)
    else:
        env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", render_mode='rgb', apply_api_compatibility=True)

    # Limit actions to "right" and "right + jump"
    env = JoypadSpace(env, [["right"], ["right", "A"]])

    # Apply wrappers for skipping frames, grayscaling, and resizing.
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)

    # Stack 4 consecutive frames for temporal context.
    if gym.__version__ < '0.26':
        env = FrameStack(env, num_stack=4, new_step_api=True)
    else:
        env = FrameStack(env, num_stack=4)

    return env
