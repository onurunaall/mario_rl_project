import datetime
from pathlib import Path
import torch

from environment import create_environment
from agent import Mario
from logger import MetricLogger


def main() -> None:
    env = create_environment()

    use_cuda = torch.cuda.is_available()
    print(f"[INFO] Using CUDA: {use_cuda}\n")

    save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir.mkdir(parents=True, exist_ok=True)

    state_dim = (4, 84, 84)  # as provided by the FrameStack wrapper
    action_dim = env.action_space.n
    mario = Mario(state_dim=state_dim, action_dim=action_dim, save_dir=save_dir)

    logger = MetricLogger(save_dir)

    episodes = 40
    for e in range(episodes):
        state = env.reset()

        while True:
            action = mario.act(state)
            next_state, reward, done, trunc, info = env.step(action)

            mario.cache(state, next_state, action, reward, done)
          
            q, loss = mario.learn()

            logger.log_step(reward, loss if loss is not None else 0.0, q if q is not None else 0.0)

            state = next_state

            if done or info.get("flag_get", False):
                break

        logger.log_episode()

        if e % 20 == 0 or e == episodes - 1:
            logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)

    env.close()


if __name__ == "__main__":
    main()
