import numpy as np
import matplotlib.pyplot as plt
from typing import List
from q_learning import QLearningAgent, QLearningConfig
import pandas as pd

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
    plt.savefig(f"plots/{title}.png")
    plt.show()

def plot_decay_nodecay(decayAgent, nodecayAgent, window: int = 100, title: str = "FrozenLake_decay_vs_nodecay_rewards"):
    plt.figure()
    arr1 = np.array(decayAgent.episode_rewards, dtype=np.float32)
    moving1 = np.convolve(arr1, np.ones(window)/window, mode="valid")
    plt.plot(moving1, label=f"Decay")

    arr2 = np.array(nodecayAgent.episode_rewards, dtype=np.float32)
    moving2 = np.convolve(arr2, np.ones(window)/window, mode="valid")
    plt.plot(moving2, label=f"No decay")

    plt.title(f"Decay vs No Decay en FrozenLake - (window={window})")
    plt.xlabel("Bloques de episodios")
    plt.ylabel("Reward promedio")
    plt.legend()
    plt.grid()
    plt.savefig(f"plots/{title}.png")
    plt.show()

def run_FrozenLake_hyperparams():
    alphas = [0.1, 0.5, 0.8]
    epsilons = [0.05, 0.1, 0.3, 0.5]
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

def run_FrozenLake(cfg = None):
    if cfg is None:
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

    # Experimento 2: TUNEO DE HIPERPARÁMETROS
    # run_FrozenLake_hyperparams()

    # Experimento 3: EVALUAR AL AGENTE LUEGO DEL ENTRENAMIENTO, DURANTE 100 EPISODIOS
    # agent = run_FrozenLake()
    # sr_greedy = agent.evaluate_success_rate(episodes=100, epsilon=0.0)
    # sr_eps01 = agent.evaluate_success_rate(episodes=100, epsilon=0.1)
    # print(f"Success rate (greedy): {sr_greedy:.1f}%")
    # print(f"Success rate (epsilon=0.1): {sr_eps01:.1f}%")

    # Experimento 4: Decay de epsilon 
    # cfg = QLearningConfig(
    #     env_name="FrozenLake-v1",
    #     episodes=6000,
    #     alpha=0.1,
    #     gamma=0.99,
    #     epsilon=1,
    #     min_epsilon=0.01,
    #     max_epsilon=1.0,
    #     decay_rate=0.001, 
    #     is_slippery=False,
    #     use_decay=True,
    #     log_every=200,
    #     seed=0,
    # )
    # agent = run_FrozenLake(cfg)
    # agent.plot_rewards_moving_average(title="FrozenLake_epsilon_decay")

    # Experimento 5: Comparar decay vs no decay
    agent_nodecay = run_FrozenLake() # sin decay
    cfg_decay = QLearningConfig(
        env_name="FrozenLake-v1",
        episodes=5000,
        alpha=0.1,
        gamma=0.99,
        epsilon=1,
        min_epsilon=0.01,
        max_epsilon=1.0,
        decay_rate=0.001,
        is_slippery=False,
        use_decay=True,
        log_every=200,
        seed=0,
    )
    agent_decay = run_FrozenLake(cfg_decay)
    plot_decay_nodecay(agent_decay, agent_nodecay, window=100, title="FrozenLake_decay_vs_nodecay_rewards")

