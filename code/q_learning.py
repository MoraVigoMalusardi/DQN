import numpy as np
import gymnasium as gym
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import math
import matplotlib.pyplot as plt
from environments import ConstantRewardEnv, RandomObsBinaryRewardEnv, TwoStepDelayedRewardEnv
import pandas as pd

@dataclass
class QLearningConfig:
    env_name: str = "FrozenLake-v1"
    map_name: str = "4x4"
    is_slippery: bool = True
    gamma: float = 0.99
    alpha: float = 0.8
    epsilon: float = 1.0
    min_epsilon: float = 0.01
    max_epsilon: float = 1.0
    decay_rate: float = 0.001  # used if use_decay=True
    episodes: int = 5000
    max_steps_per_episode: int = 100 # The length of the episode is 100 for FrozenLake4x4, 200 for FrozenLake8x8. (lo saque de https://gymnasium.farama.org/environments/toy_text/frozen_lake/)
    seed: Optional[int] = 0
    use_decay: bool = True
    log_every: int = 100  # for moving average plotting


envs = {
        "ConstantRewardEnv": ConstantRewardEnv,
        "RandomObsBinaryRewardEnv": RandomObsBinaryRewardEnv,
        "TwoStepDelayedRewardEnv": TwoStepDelayedRewardEnv,
        "FrozenLake-v1" : lambda: gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False)
        # is slippery: If true the player will move in intended 
        # direction with probability specified by the success_rate else will move 
        # in either perpendicular direction with equal probability in both directions. (lo saque de https://gymnasium.farama.org/environments/toy_text/frozen_lake/)
}

class QLearningAgent:
    def __init__(self, config: QLearningConfig):
        self.config = config
        self.env = envs[config.env_name]()

        if config.seed is not None:
            self.env.reset(seed=config.seed)
            np.random.seed(config.seed)
        
        # Usamos cantidad de estados x cantidad de acciones 
        nS = self.env.observation_space.n
        nA = self.env.action_space.n
        # para inicializar la tabla Q en ceros
        self.Q = np.zeros((nS, nA), dtype=np.float32)

        self.episode_rewards: List[float] = []
        self.epsilons: List[float] = []

    def _epsilon_for_episode(self, episode: int) -> float:
        """ Calculates epsilon for given episode, using decay if configured."""
        cfg = self.config
        if not cfg.use_decay:
            return cfg.epsilon
        return cfg.min_epsilon + (cfg.max_epsilon - cfg.min_epsilon) * math.exp(-cfg.decay_rate * episode)

    def choose_action(self, state: int, epsilon: float) -> int:
        """ Epsilon-greedy action selection. """
        if np.random.rand() < epsilon:
            return self.env.action_space.sample()
        q_row = self.Q[state] # valores de Q(s, a1), Q(s, a2), ...
        max_q = np.max(q_row) # valor maximo de Q(s, ai)
        best_actions = np.flatnonzero(q_row == max_q) # indices de las acciones con valor maximo
        return int(np.random.choice(best_actions)) # si hay mas de una con valor maximo, elijo una aleatoriamente

    def update(self, s: int, a: int, r: float, s_next: int, terminated: bool):
        """ Implements the Q-learning update for a given transition. """
        # Aplica la regla de actualizacion de Q-Learning
        cfg = self.config
        best_next = 0.0 if terminated else np.max(self.Q[s_next])
        td_target = r + cfg.gamma * best_next
        td_error = td_target - self.Q[s, a]
        self.Q[s, a] += cfg.alpha * td_error

    def train(self) -> Dict[str, float]:
        """ Trains the agent for a number of episodes set in config """
        cfg = self.config
        self.episode_rewards.clear()
        self.epsilons.clear()

        for ep in range(cfg.episodes):
            state, _ = self.env.reset()
            total_reward = 0.0
            epsilon = self._epsilon_for_episode(ep)
            self.epsilons.append(epsilon)

            for _ in range(cfg.max_steps_per_episode):
                action = self.choose_action(state, epsilon) # epsilon-greedy action selection
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                self.update(state, action, reward, next_state, terminated) # actualizar Q-table. Gamma se usa en esa funcion
                total_reward += reward
                state = next_state
                if terminated or truncated:
                    break

            self.episode_rewards.append(total_reward)

            if (ep + 1) % cfg.log_every == 0:  # si el numero del episodio es multiplo de log_every, loggeamos
                window = self.episode_rewards[-cfg.log_every:]
                print(f"Episode {ep+1}/{cfg.episodes} | avg_reward(last {cfg.log_every})={np.mean(window):.3f} | epsilon={epsilon:.3f}")

        return {"avg_reward": float(np.mean(self.episode_rewards[-cfg.log_every:] if self.episode_rewards else [0.0]))}

    def evaluate_success_rate(self, episodes: int = 100, epsilon: float = 0.0) -> float:
        """ Evaluates the agent's success rate over a number of episodes using given epsilon for action selection. """
        # Usamos epsilon=0.0 para evaluar la politica greedy y ver como performa el agente. 
        successes = 0
        for _ in range(episodes):
            state, _ = self.env.reset()
            for _ in range(self.config.max_steps_per_episode):
                action = self.choose_action(state, epsilon)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                state = next_state
                if terminated or truncated:
                    if reward > 0:
                        successes += 1
                    break
        return 100.0 * successes / episodes

    def plot_rewards_moving_average(self, window: int = 100, title: str = "q_learning_rewards"):
        arr = np.array(self.episode_rewards, dtype=np.float32)
        moving = np.convolve(arr, np.ones(window)/window, mode="valid") # calculamos la media movil de las recompensas para que sea mas facil de visualizar y mas suave 
        plt.figure()
        plt.plot(moving)
        plt.title(f"{title} - media movil (window={window})")
        plt.xlabel("Bloques de episodios")
        plt.ylabel("Reward promedio")
        plt.grid()
        plt.savefig(f"{title}.png")
        plt.show()

    def summary(self) -> Dict[str, float]:
        """ Returns a summary of the training results. """
        return {
            **asdict(self.config),
            "final_avg_reward_last_window": float(np.mean(self.episode_rewards[-self.config.log_every:])) if self.episode_rewards else 0.0,
        }

def plot_several_rewards(agents: List[QLearningAgent], window: int = 100, title: str = "q_learning_multiple_rewards"):
    plt.figure()
    for agent in agents:
        arr = np.array(agent.episode_rewards, dtype=np.float32)
        moving = np.convolve(arr, np.ones(window)/window, mode="valid")
        plt.plot(moving, label=f"α={agent.config.alpha}, ε={agent.config.epsilon}")
    plt.title(f"Comparacion de hiperparams en FrozenLake - (window={window})")
    plt.xlabel("Bloques de episodios")
    plt.ylabel("Reward promedio")
    plt.legend()
    plt.grid()
    plt.savefig(f"{title}.png")
    plt.show()

def run_FrozenLake_hyperparams():
    alphas = [0.1, 0.5, 0.8]
    epsilons = [0.1, 0.3, 0.5]
    agents = []
    for alpha in alphas:
        for epsilon in epsilons:
            print(f"Running Q-Learning with alpha={alpha}, epsilon={epsilon}")
            cfg = QLearningConfig(
                env_name="FrozenLake-v1",
                episodes=5000,
                alpha=alpha,
                gamma=0.99,
                epsilon=epsilon,
                min_epsilon=0.01,
                max_epsilon=1.0,
                decay_rate=0,
                is_slippery=False,
                use_decay=False,
                log_every=500,
                seed=0,
            )
            agent = QLearningAgent(cfg)
            agent.train()
            agents.append(agent)
    plot_several_rewards(agents, window=100, title="FrozenLake_QLearning_hyperparams_rewards")
    return agents

def run_FrozenLake():
    cfg = QLearningConfig(
        env_name="FrozenLake-v1",
        episodes=5000,
        alpha=0.1,
        gamma=0.99,
        epsilon=0.1,
        min_epsilon=0.01,
        max_epsilon=1.0,
        decay_rate=0,
        is_slippery=False,
        use_decay=False,
        log_every=200,
        seed=0,
    )
    agent = QLearningAgent(cfg)
    agent.train()
    sr_greedy = agent.evaluate_success_rate(episodes=100, epsilon=0.0)
    sr_eps01 = agent.evaluate_success_rate(episodes=100, epsilon=0.1)
    print(f"Success rate (greedy): {sr_greedy:.1f}%")
    print(f"Success rate (epsilon=0.1): {sr_eps01:.1f}%")
    try:
        agent.plot_rewards_moving_average(window=100, title="FrozenLake_QLearning_rewards")
    except Exception:
        pass
    return agent

def run_custom_envs():
    for env_name in ["ConstantRewardEnv", "RandomObsBinaryRewardEnv", "TwoStepDelayedRewardEnv"]:
        print(f"Running Q-Learning on {env_name}")
        cfg = QLearningConfig(
            env_name=env_name,
            episodes=1000,
            alpha=0.8,
            gamma=0.99,
            epsilon=0.5,
            min_epsilon=0.01,
            max_epsilon=1.0,
            decay_rate=0.001,
            use_decay=True,
            log_every=100,
            seed=0,
        )
        agent = QLearningAgent(cfg)
        agent.train()
        # Muestro la Q-table 
        print(f"Tabla de valores Q para {env_name}:")
        df_q = pd.DataFrame(agent.Q, columns=[f"A{a}" for a in range(agent.Q.shape[1])])
        df_q.index = [f"S{s}" for s in range(agent.Q.shape[0])]
        print(df_q)

        avg_reward = np.mean(agent.episode_rewards)
        print(f"Average reward over training: {avg_reward:.3f}")
        try:
            agent.plot_rewards_moving_average(window=50, title=f"{env_name}_rewards")
        except Exception:
            pass
    return

if __name__ == "__main__":
    # run_custom_envs() 
    # run_FrozenLake()
    run_FrozenLake_hyperparams()
