import datetime
import os
from pathlib import Path
from typing import Tuple, Optional

import torch
import torch.optim as optim
import numpy as np

from network import MarioNet

from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage


class Mario:
    """
    Mario RL Agent class.
    Implements action selection, experience replay, learning, and checkpointing.
    """
    def __init__(self, state_dim: Tuple[int, int, int], action_dim: int, save_dir: Path) -> None:
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir

        self.net = MarioNet(self.state_dim, self.action_dim).float().to(self.device)

        # Epsilon-greedy exploration parameters.
        self.exploration_rate: float = 1.0
        self.exploration_rate_decay: float = 0.99999975
        self.exploration_rate_min: float = 0.1
        self.curr_step: int = 0

        self.save_every: int = int(5e5)
        self.burnin: int = int(1e4)
        self.learn_every: int = 3
        self.sync_every: int = int(1e4)

        self.optimizer = optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()

        # Initialize replay memory.
        self.memory = TensorDictReplayBuffer(storage=LazyMemmapStorage(100000, device=torch.device("cpu")))
        self.batch_size: int = 32

    def act(self, state: torch.Tensor) -> int:
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)
        else:
            # Convert state to tensor if necessary.
            if isinstance(state, tuple):
                state = state[0]
              
            state_array = state.__array__() if hasattr(state, '__array__') else state
          
            state_tensor = torch.tensor(state_array, device=self.device).unsqueeze(0)
          
            action_values = self.net(state_tensor, model="online")
          
            action_idx = int(torch.argmax(action_values, dim=1).item())

        # Decay exploration rate.
        self.exploration_rate = max(self.exploration_rate * self.exploration_rate_decay, self.exploration_rate_min)
        self.curr_step += 1
      
        return action_idx

    def cache(self, state: torch.Tensor, next_state: torch.Tensor, action: int, reward: float, done: bool) -> None:
        """
        Store an experience in the replay buffer.
        """
        def first_if_tuple(x):
            return x[0] if isinstance(x, tuple) else x
          
        state_np = first_if_tuple(state).__array__()
        next_state_np = first_if_tuple(next_state).__array__()

        state_tensor = torch.tensor(state_np)
        next_state_tensor = torch.tensor(next_state_np)
        action_tensor = torch.tensor([action])
        reward_tensor = torch.tensor([reward])
        done_tensor = torch.tensor([done])

        experience = TensorDict({
            "state": state_tensor,
            "next_state": next_state_tensor,
            "action": action_tensor,
            "reward": reward_tensor,
            "done": done_tensor
        }, batch_size=[])
      
        self.memory.add(experience)

    def recall(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a batch of experiences from memory.
        """
    
        batch = self.memory.sample(self.batch_size).to(self.device)
        state = batch.get("state")
        next_state = batch.get("next_state")
        action = batch.get("action")
        reward = batch.get("reward")
        done = batch.get("done")
      
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def td_estimate(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute the TD estimate using the online network.
        """
        batch_indices = np.arange(0, self.batch_size)
        q_values = self.net(state, model="online")
        return q_values[batch_indices, action]

    @torch.no_grad()
    def td_target(self, reward: torch.Tensor, next_state: torch.Tensor,
                  done: torch.Tensor) -> torch.Tensor:
        """
        Compute the TD target using the target network.
        """
        next_q_values = self.net(next_state, model="online")
                    
        best_actions = torch.argmax(next_q_values, dim=1)
                    
        batch_indices = np.arange(0, self.batch_size)
                    
        next_q_target = self.net(next_state, model="target")[batch_indices, best_actions]
                    
        return (reward + (1 - done.float()) * 0.9 * next_q_target).float()

    def update_Q_online(self, td_est: torch.Tensor, td_tgt: torch.Tensor) -> float:
        """
        Update the online network based on the TD error.
        """
        loss = self.loss_fn(td_est, td_tgt)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self) -> None:
        """
        Sync the target network with the online network.
        """
        self.net.target.load_state_dict(self.net.online.state_dict())

    def save(self) -> None:
        """
        Save the current model checkpoint.
        """
        checkpoint = {
            "model_state_dict": self.net.state_dict(),
            "exploration_rate": self.exploration_rate,
            "curr_step": self.curr_step
        }
        save_path = self.save_dir / f"mario_net_{int(self.curr_step // self.save_every)}.chkpt"
        torch.save(checkpoint, save_path)
        print(f"[INFO] MarioNet saved to {save_path} at step {self.curr_step}")

    def learn(self) -> Tuple[Optional[float], Optional[float]]:
        """
        Perform a learning step if enough experiences have been gathered.
        """
        # Sync target network periodically.
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            self.save()

        # Wait until a minimum number of experiences have been collected.
        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        # Sample a batch and compute TD estimate and target.
        state, next_state, action, reward, done = self.recall()
        td_est = self.td_estimate(state, action)
        td_tgt = self.td_target(reward, next_state, done)

        loss = self.update_Q_online(td_est, td_tgt)
        mean_q_value = td_est.mean().item()
        return mean_q_value, loss
