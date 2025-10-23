from dqn import DQNAgent, DQNConfig
import torch


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
            "max_steps_per_episode" : 500, # la consigna dice 1000, pero en https://gymnasium.farama.org/environments/classic_control/cart_pole/ dice 500
            "seed" : 0,
            "epsilon_start" : 1.0,
            "epsilon_end" : 0.01,
            "epsilon_decay_rate" : 0.0001,
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
        "target_update_freq": 2_000,         # en pasos
        "max_episodes": 30000,
        "max_steps_per_episode": 2_000,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay_rate": 1.0/200_000, 
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

def run_custom_envs():
    for env_name in ["ConstantRewardEnv", "RandomObsBinaryRewardEnv", "TwoStepDelayedRewardEnv"]:
        print(f"Running DQN on {env_name}")
        cfg = DQNConfig(
            env_id=env_name,
            max_episodes=1000,
            gamma=0.99,
            lr=1e-3,
            batch_size=64,
            buffer_capacity=5000,
            learning_starts=500,
            train_freq=1,
            target_update_freq=100,
            epsilon_start=1.0,
            epsilon_end=0.1,
            epsilon_decay_rate=0.001,
            log_dir=f"runs/dqn_{env_name}",
            seed=0,
        )
        agent = DQNAgent(cfg)
        agent.train()
        agent.plot_moving_average_rewards(window=50, title=f"DQN_{env_name}_Moving_Average_Rewards")


if __name__ == "__main__":
    # 3.1: Entrenar CartPole-v1 con DQN hasta que converja
    # agent_cartpole, results_cartpole = train_dqn_cartpole()
    # print("Training results:")
    # for k, v in results_cartpole.items():
    #     print(f"  {k}: {v}")
    # agent_cartpole.plot_rewards(title="DQN_CartPole_Rewards")
    # agent_cartpole.plot_moving_average_rewards(window=100, title="DQN_CartPole_Moving_Average_Rewards")
    # # Me dio --> eval_mean_reward: 474.35 ± 43.75

    # 3.2: Entrenar sobre Minatar-Breakout 
    # agent_minatar, results_minatar = train_dqn_minatar_breakout()
    # print("Training results:")
    # for k, v in results_minatar.items():
    #     print(f"  {k}: {v}")
    # agent_minatar.plot_rewards(title="DQN_Minatar_Breakout_Rewards")
    # agent_minatar.plot_moving_average_rewards(window=100, title="DQN_Minatar_Breakout_Moving_Average_Rewards")

    # Experimento 1: Correr custom environments 
    # run_custom_envs()  --> TODAVIA NO ME FUNCIONA

    pass