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

from arquitectures import MLP, CNN
from environments import SimpleFrameStack

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
        """ Sample a batch of transitions uniformly """
        # idx es un array de índices aleatorios
        idx = np.random.randint(0, self.size, size=batch_size) # tamano=batch_size, valores entre 0 y self.size-1
        return (
            self.obs_buf[idx],
            self.act_buf[idx],
            self.rew_buf[idx],
            self.next_obs_buf[idx],
            self.done_buf[idx],
        )

    def __len__(self):
        return self.size


class DQNAgent:
    def __init__(self, cfg: DQNConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.env = gym.make(cfg.env_id)
        if cfg.seed is not None:
            self.env.reset(seed=cfg.seed)
            np.random.seed(cfg.seed)
            torch.manual_seed(cfg.seed)

        # miramos una observación para decidir si es imagen
        obs0, _ = self.env.reset()
        is_image = (isinstance(obs0, np.ndarray) and obs0.ndim == 3)

        # Si es imagen, aplicamos framestack (k=4) y volvemos a pedir obs
        if is_image:
            self.env = SimpleFrameStack(self.env, k=4)
            obs0, _ = self.env.reset()

        obs_shape = obs0.shape                   # (D,)  o  (C*k, H, W)
        n_actions = self.env.action_space.n

        # Elegimos MLP o CNN
        if is_image:
            C, H, W = obs_shape
            self.policy_net = CNN(C, (H, W), n_actions).to(self.device)
            self.target_net = CNN(C, (H, W), n_actions).to(self.device)
        else:
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
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if obs_t.ndim == 1:
            obs_t = obs_t.unsqueeze(0)                 
        elif obs_t.ndim == 3:
            obs_t = obs_t.unsqueeze(0) 

        q_values = self.policy_net(obs_t)
        return int(torch.argmax(q_values, dim=1).item()) # Devuelve la acción con mayor Q value 

    def optimize(self) -> Optional[float]:
        cfg = self.cfg

        # Si no hay suficientes datos (len(buffer) < batch_size) o todavia estamos en warmup (global_step < learning_starts), devuelve None
        if len(self.buffer) < cfg.batch_size or self.global_step < cfg.learning_starts:
            return None

        # 4) Sampleamos mini-batch de transiciones del replay buffer
        obs, act, rew, next_obs, done = self.buffer.sample(cfg.batch_size)

            # convertimos a tensores
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        act_t = torch.as_tensor(act, dtype=torch.int64, device=self.device).unsqueeze(1)
        rew_t = torch.as_tensor(rew, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        done_t = torch.as_tensor(done, dtype=torch.float32, device=self.device).unsqueeze(1)

        # 5) Calculo yi (output objetivo) usando la red target
        with torch.no_grad():
            # yi = ri + γ * max_a' Q_target(s', a') 
            q_next = self.target_net(next_obs_t).max(dim=1, keepdim=True)[0]
            q_target = rew_t + self.cfg.gamma * q_next * (1.0 - done_t) # si done=1, no sumamos valor futuro. El valor de un estado terminal es solo el reward inmediato

            # Calculo el valor de Q(s, a) usando la red de mi politica
        q_pred = self.policy_net(obs_t).gather(1, act_t)

        # 6) Calculo loss y optimizo
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

                # 1) seleccionamos una accion e-greedy
                epsilon = self.epsilon_by_step(self.global_step)
                action = self.select_action(obs, epsilon)

                # 2) ejecutamos la accion y observamos
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                # 3) la guardamos en el buffer
                self.buffer.push(obs, action, reward, next_obs, done)

                # 4) samplear minibatch  -  5) calcular yi  -  6) optimizar
                loss = self.optimize()
                if loss is not None:
                    losses_window.append(loss)
                    # Guardamos loss para tensorboard
                    self.writer.add_scalar("loss/td_loss", loss, self.global_step)

                # 7) Cada target_update_freq pasos, actualizamos la red target 
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
            "global_steps": self.global_step,
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


def train_dqn_cartpole():
    cfg = DQNConfig()

    config_overrides={
            "env_id" : "CartPole-v1",
            "gamma" : 0.99,
            "lr" : 1e-3,
            "batch_size" : 64,
            "buffer_capacity" : 50000,
            "learning_starts" : 1000,
            "train_freq" : 1,
            "target_update_freq" : 1000,
            "max_episodes" : 4000,
            "max_steps_per_episode" : 1000, # la consigna dice 1000, pero en https://gymnasium.farama.org/environments/classic_control/cart_pole/ dice 500
            "seed" : 0,
            "epsilon_start" : 1.0,
            "epsilon_end" : 0.01,
            "epsilon_decay_rate" : 0.001,
            "log_dir" : "runs/dqn_cartpole",
            "checkpoint_path" : "dqn_cartpole.pt",
            "device" : "cuda" if torch.cuda.is_available() else "cpu"
    }

    for k, v in config_overrides.items():
        setattr(cfg, k, v)

    agent = DQNAgent(cfg)
    summary = agent.train()
    eval_res = agent.evaluate(episodes=20, epsilon_eval=0.0)
    print(f"Eval (greedy): mean={eval_res['mean_reward']:.1f} ± {eval_res['std_reward']:.1f}")
    return agent, {**summary, **{f'eval_{k}': v for k, v in eval_res.items()}}


def train_dqn_minatar_breakout():
    cfg = DQNConfig()
    defaults = {
        "env_id": "MinAtar/Breakout-v0",
        "gamma": 0.99,
        "lr": 2.5e-4,
        "batch_size": 32,
        "buffer_capacity": 200_000,
        "learning_starts": 5_000,
        "train_freq": 4,
        "target_update_freq": 2_000,        # en pasos
        "max_episodes": 30000,
        "max_steps_per_episode": 2_000,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay_rate": 1.0/200_000,  # decaimiento suave a ~200k steps
        "log_dir": "runs/dqn_minatar_breakout",
        "checkpoint_path": "dqn_minatar_breakout.pt",
    }
    for k, v in defaults.items():
        setattr(cfg, k, v)

    agent = DQNAgent(cfg)
    summary = agent.train()
    eval_res = agent.evaluate(episodes=20, epsilon_eval=0.0)
    print(f"Eval Breakout (greedy): mean={eval_res['mean_reward']:.2f} ± {eval_res['std_reward']:.2f}")
    print("Global steps:", summary["global_steps"])
    return agent, {**summary, **{f"eval_{k}": v for k, v in eval_res.items()}}

def plot_rewards(summary: Dict[str, float], title: str):
    rewards = np.array(summary["rewards"], dtype=np.float32)
    plt.plot(rewards)
    plt.xlabel("Episodio"); plt.ylabel("Reward"); plt.grid(True, alpha=0.3)
    plt.title(title)
    plt.tight_layout()
    plt.show()
    plt.savefig(f"{title}.png")


def plot_moving_average_rewards(summary: Dict[str, float], title: str, window: int = 100, ):
    rewards = np.asarray(summary["rewards"], dtype=np.float32)
    if rewards.size < window:
        print(f"Hay solo {rewards.size} episodios; el moving average de ventana {window} necesita ≥ {window}.")
        return
    ma = np.convolve(rewards, np.ones(window, dtype=np.float32)/window, mode="valid")
    x = np.arange(window-1, window-1 + ma.size)  # alinear con el último episodio incluido en cada promedio

    plt.figure()
    plt.plot(x, ma, label=f"Mean {window}-ep")
    plt.xlabel("Episodio"); plt.ylabel("Reward promedio"); plt.grid(True, alpha=0.3)
    plt.title(f"{title} - Moving Average ({window} episodios)")
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig(f"{title}.png")


if __name__ == "__main__":
    agent, summary = train_dqn_cartpole()
    # agent, summary = train_dqn_minatar_breakout()
    # print("Summary:", summary)
    plot_rewards(summary, title="DQN_CartPole_rewards")
    plot_moving_average_rewards(summary, title="DQN_CartPole_ma_rewards", window=100)
