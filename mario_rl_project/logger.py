import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List

class MetricLogger:
    def __init__(self, save_dir: Path) -> None:
        self.save_dir = save_dir
        self.save_log = self.save_dir / "log.txt"

        with open(self.save_log, "w") as f:
            f.write(f"{'Episode':>8}{'Step':>8}{'Epsilon':>10}{'MeanReward':>15}"
                    f"{'MeanLength':>15}{'MeanLoss':>15}{'MeanQValue':>15}"
                    f"{'TimeDelta':>15}{'Time':>20}\n")


        self.ep_rewards_plot = self.save_dir / "reward_plot.jpg"
        self.ep_lengths_plot = self.save_dir / "length_plot.jpg"
        self.ep_avg_losses_plot = self.save_dir / "loss_plot.jpg"
        self.ep_avg_qs_plot = self.save_dir / "q_plot.jpg"

        self.ep_rewards: List[float] = []
        self.ep_lengths: List[int] = []
        self.ep_avg_losses: List[float] = []
        self.ep_avg_qs: List[float] = []

        self.moving_avg_ep_rewards: List[float] = []
        self.moving_avg_ep_lengths: List[float] = []
        self.moving_avg_ep_avg_losses: List[float] = []
        self.moving_avg_ep_avg_qs: List[float] = []

        self.init_episode()
        self.record_time = time.time()

    def init_episode(self) -> None:
        """
        Initialize current episode metrics.
        """
        self.curr_ep_reward: float = 0.0
        self.curr_ep_length: int = 0
        self.curr_ep_loss: float = 0.0
        self.curr_ep_q: float = 0.0
        self.curr_ep_loss_length: int = 0

    def log_step(self, reward: float, loss: float, q: float) -> None:
        """
        Record metrics for a single step.
        """
        self.curr_ep_reward += reward
        self.curr_ep_length += 1
      
        if loss is not None:
            self.curr_ep_loss += loss
            self.curr_ep_q += q
            self.curr_ep_loss_length += 1

    def log_episode(self) -> None:
        """
        Store metrics at the end of an episode.
        """
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)
      
        if self.curr_ep_loss_length == 0:
            ep_avg_loss = 0.0
            ep_avg_q = 0.0
        else:
            ep_avg_loss = np.round(self.curr_ep_loss / self.curr_ep_loss_length, 5)
            ep_avg_q = np.round(self.curr_ep_q / self.curr_ep_loss_length, 5)
          
        self.ep_avg_losses.append(ep_avg_loss)
        self.ep_avg_qs.append(ep_avg_q)
      
        self.init_episode()

    def record(self, episode: int, epsilon: float, step: int) -> None:
        """
        Calculate moving averages and log overall metrics.
        """
        mean_ep_reward = np.round(np.mean(self.ep_rewards[-100:]), 3)
        mean_ep_length = np.round(np.mean(self.ep_lengths[-100:]), 3)
        mean_ep_loss = np.round(np.mean(self.ep_avg_losses[-100:]), 3)
        mean_ep_q = np.round(np.mean(self.ep_avg_qs[-100:]), 3)

        self.moving_avg_ep_rewards.append(mean_ep_reward)
        self.moving_avg_ep_lengths.append(mean_ep_length)
        self.moving_avg_ep_avg_losses.append(mean_ep_loss)
        self.moving_avg_ep_avg_qs.append(mean_ep_q)

        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time - last_record_time, 3)

        log_str = (f"Episode {episode} - Step {step} - Epsilon {epsilon:.3f} - "
                   f"Mean Reward {mean_ep_reward} - Mean Length {mean_ep_length} - "
                   f"Mean Loss {mean_ep_loss} - Mean Q Value {mean_ep_q} - "
                   f"Time Delta {time_since_last_record} - "
                   f"Time {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}")
      
        print(log_str)
      
        with open(self.save_log, "a") as f:
            f.write(f"{episode:8d}{step:8d}{epsilon:10.3f}"
                    f"{mean_ep_reward:15.3f}{mean_ep_length:15.3f}"
                    f"{mean_ep_loss:15.3f}{mean_ep_q:15.3f}"
                    f"{time_since_last_record:15.3f}"
                    f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n")

        self._update_plot(self.moving_avg_ep_rewards, self.ep_rewards_plot, "Mean Reward")
        self._update_plot(self.moving_avg_ep_lengths, self.ep_lengths_plot, "Mean Length")
        self._update_plot(self.moving_avg_ep_avg_losses, self.ep_avg_losses_plot, "Mean Loss")
        self._update_plot(self.moving_avg_ep_avg_qs, self.ep_avg_qs_plot, "Mean Q Value")

    def _update_plot(self, data: list, save_path: Path, label: str) -> None:
        """
        Helper method to generate and save a plot.
        """
        plt.clf()
        plt.plot(data, label=label)
        plt.legend()
        plt.savefig(save_path)
