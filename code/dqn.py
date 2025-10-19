import os
import time
import math
from dataclasses import dataclass, asdict
from typing import Deque, Dict, List, Optional, Tuple
from collections import deque

import numpy as np
import gymnasium as gym

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
import numpy as np

@dataclass
class DQNConfig:
    env_id: str = "CartPole-v1"
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    buffer_capacity: int = 50000
    learning_starts: int = 1000
    train_freq: int = 1
    target_update_freq: int = 1000
    max_episodes: int = 500
    max_steps_per_episode: int = 500 # la consigna dice 1000, pero en https://gymnasium.farama.org/environments/classic_control/cart_pole/ dice 500
    seed: Optional[int] = 0
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_rate: float = 0.001
    log_dir: str = "runs/dqn_cartpole"
    checkpoint_path: str = "dqn_cartpole.pt"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ReplayBuffer:
    """
    FIFO replay buffer
    Stores transitions (obs, action, reward, next_obs, done) 
    Has finite capacity
    Samples random batches with uniform probability
    """
    def __init__(self, capacity: int, obs_shape: Tuple[int, ...]):
        self.capacity = capacity
        self.obs_buf = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.next_obs_buf = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.act_buf = np.zeros((capacity,), dtype=np.int64)
        self.rew_buf = np.zeros((capacity,), dtype=np.float32)
        self.done_buf = np.zeros((capacity,), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def push(self, obs, action: int, reward: float, next_obs, done: bool):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = action
        self.rew_buf[self.ptr] = reward
        self.done_buf[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            self.obs_buf[idx],
            self.act_buf[idx],
            self.rew_buf[idx],
            self.next_obs_buf[idx],
            self.done_buf[idx],
        )

    def __len__(self):
        return self.size


class MLP(nn.Module):
    """ Simple MLP """
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        # Uso la arquitectura que dan en la consigna
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )
    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(self, cfg: DQNConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.env = gym.make(cfg.env_id)
        if cfg.seed is not None:
            self.env.reset(seed=cfg.seed)
            np.random.seed(cfg.seed)
            torch.manual_seed(cfg.seed)

        obs_shape = self.env.observation_space.shape
        n_actions = self.env.action_space.n

        # Inicializamos las dos redes
        self.policy_net = MLP(input_dim=obs_shape[0], output_dim=n_actions).to(self.device)
        self.target_net = MLP(input_dim=obs_shape[0], output_dim=n_actions).to(self.device)

        # Copiamos los pesos de la red de politica a la red objetivo
        self.target_net.load_state_dict(self.policy_net.state_dict())
        """
        state_dict() --> Devuelve un diccionario con todos los parámetros entrenables de la red (pesos y biases de cada capa) y buffers (como running_mean/var en batchnorms).
        load_state_dict(state_dict)	--> Carga ese diccionario en la red, reemplazando sus pesos actuales por los nuevos.
        """
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.lr)
        self.loss_fn = nn.SmoothL1Loss() # Huber Loss: Es MSE para errores chicos y L1 lineal para errores grandes

        self.buffer = ReplayBuffer(cfg.buffer_capacity, obs_shape)

        ts = time.strftime("%Y%m%d-%H%M%S") 
        log_dir = os.path.join(cfg.log_dir, ts)
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir) # SummaryWriter para TensorBoard
        print("TensorBoard logdir:", self.writer.log_dir)

        self.global_step = 0
        self.best_mean_reward = -float("inf")

    def epsilon_by_step(self, step: int) -> float:
        """ Epsilon decay schedule """
        cfg = self.cfg
        eps = cfg.epsilon_end + (cfg.epsilon_start - cfg.epsilon_end) * math.exp(-cfg.epsilon_decay_rate * step)
        return float(max(cfg.epsilon_end, min(cfg.epsilon_start, eps)))

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, epsilon: float) -> int:
        """ Epsilon-greedy action selection """
        # Con probabilidad epsilon tomamos una acción aleatoria
        if np.random.rand() < epsilon:
            return self.env.action_space.sample()
        
        # Sino tomamos la acción greedy según la policy net
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0) 
        q_values = self.policy_net(obs_t)
        return int(torch.argmax(q_values, dim=1).item()) # Devuelve la acción con mayor Q value 

    def optimize(self) -> Optional[float]:
        cfg = self.cfg

        # Si no hay suficientes datos (len(buffer) < batch_size) o todavia estamos en warmup (global_step < learning_starts), devuelve None
        if len(self.buffer) < cfg.batch_size or self.global_step < cfg.learning_starts:
            return None

        # Sampleamos mini-batch de transiciones del replay buffer
        obs, act, rew, next_obs, done = self.buffer.sample(cfg.batch_size)

        # convertimos a tensores
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        act_t = torch.as_tensor(act, dtype=torch.int64, device=self.device).unsqueeze(1)
        rew_t = torch.as_tensor(rew, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        done_t = torch.as_tensor(done, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Calculamos los Q values actuales 
        q_pred = self.policy_net(obs_t).gather(1, act_t)

        # Calculamos los Q values target usando la target net
        with torch.no_grad():
            q_next = self.target_net(next_obs_t).max(dim=1, keepdim=True)[0]
            q_target = rew_t + self.cfg.gamma * q_next * (1.0 - done_t)

        loss = self.loss_fn(q_pred, q_target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        return float(loss.item())

    def maybe_update_target(self):
        """ Actualiza la red target cada target_update_freq pasos """
        # Cada target_update_freq pasos globales, copia pesos de policy_net -> target_net con load_state_dict
        if self.global_step % self.cfg.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_checkpoint(self, path: str):
        torch.save({
            "policy_state_dict": self.policy_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": asdict(self.cfg),
        }, path)

    def train(self) -> Dict[str, float]:
        cfg = self.cfg
        episode_rewards: List[float] = []
        losses_window: Deque[float] = deque(maxlen=1000)

        for ep in range(cfg.max_episodes):
            obs, _ = self.env.reset()
            total_reward = 0.0

            for t in range(cfg.max_steps_per_episode):
                epsilon = self.epsilon_by_step(self.global_step)
                action = self.select_action(obs, epsilon)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                self.buffer.push(obs, action, reward, next_obs, done)

                if self.global_step % cfg.train_freq == 0:
                    loss = self.optimize()
                    if loss is not None:
                        losses_window.append(loss)
                        # Guardamos loss para tensorboard
                        self.writer.add_scalar("loss/td_loss", loss, self.global_step)

                self.maybe_update_target()

                total_reward += reward
                obs = next_obs
                self.global_step += 1
                self.writer.add_scalar("train/epsilon", epsilon, self.global_step)

                if done:
                    break

            episode_rewards.append(total_reward)
            mean_r = np.mean(episode_rewards[-100:])

            # Guardamos los datos para tensorboard
            self.writer.add_scalar("reward/episode", total_reward, ep)
            self.writer.add_scalar("reward/mean_100", mean_r, ep)

            # Si mean_r supera el mejor historico, guardamos checkpoint
            if mean_r > self.best_mean_reward:
                self.best_mean_reward = mean_r
                self.save_checkpoint(self.cfg.checkpoint_path)

            print(f"Ep {ep+1}/{cfg.max_episodes} | Reward ={total_reward:.1f} | mean-R (100) ={mean_r:.2f} "
                  f"| eps={epsilon:.3f} | buffer={len(self.buffer)}")

        self.writer.flush()
        return {
            "episodes": cfg.max_episodes,
            "best_mean_reward": float(self.best_mean_reward),
            "final_mean_100": float(np.mean(episode_rewards[-100:])),
            "total_steps": int(self.global_step),
            "rewards": episode_rewards, 
        }

    @torch.no_grad()
    def evaluate(self, episodes: int = 10, epsilon_eval: float = 0.0) -> Dict[str, float]:
        """ Evalua la politica durante varios episodios y devuelve la recompensa media y std """
        rewards = []
        for _ in range(episodes):
            obs, _ = self.env.reset()
            total = 0.0
            for _ in range(self.cfg.max_steps_per_episode):
                action = self.select_action(obs, epsilon_eval)
                obs, r, terminated, truncated, _ = self.env.step(action)
                total += r
                if terminated or truncated:
                    break
            rewards.append(total)
        return {"mean_reward": float(np.mean(rewards)), "std_reward": float(np.std(rewards))}


def train_dqn_cartpole(config_overrides: Optional[dict] = None):
    cfg = DQNConfig()

    if config_overrides:
        for k, v in config_overrides.items():
            setattr(cfg, k, v)

    agent = DQNAgent(cfg)
    summary = agent.train()
    eval_res = agent.evaluate(episodes=20, epsilon_eval=0.0)
    print(f"Eval (greedy): mean={eval_res['mean_reward']:.1f} ± {eval_res['std_reward']:.1f}")
    return agent, {**summary, **{f'eval_{k}': v for k, v in eval_res.items()}}


if __name__ == "__main__":
    agent, summary = train_dqn_cartpole(
        config_overrides={
            "env_id" : "CartPole-v1",
            "gamma" : 0.99,
            "lr" : 1e-3,
            "batch_size" : 64,
            "buffer_capacity" : 50000,
            "learning_starts" : 1000,
            "train_freq" : 1,
            "target_update_freq" : 1000,
            "max_episodes" : 1000,
            "max_steps_per_episode" : 500, # la consigna dice 1000, pero en https://gymnasium.farama.org/environments/classic_control/cart_pole/ dice 500
            "seed" : 0,
            "epsilon_start" : 1.0,
            "epsilon_end" : 0.05,
            "epsilon_decay_rate" : 0.001,
            "log_dir" : "runs/dqn_cartpole",
            "checkpoint_path" : "dqn_cartpole.pt",
            "device" : "cuda" if torch.cuda.is_available() else "cpu"
        })
    print("Summary:", summary)
    rewards = np.array(summary["rewards"], dtype=np.float32)
    plt.plot(rewards)
    plt.xlabel("Episodio"); plt.ylabel("Reward"); plt.grid(True, alpha=0.3)
    plt.title("DQN – Reward por episodio")
    plt.tight_layout(); plt.show()
