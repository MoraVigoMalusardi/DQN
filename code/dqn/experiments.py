from dqn import DQNAgent, DQNConfig
import torch
import numpy as np
from pathlib import Path
import sys
import importlib
import importlib.util
import matplotlib.pyplot as plt

# --- FIX: asegurar que se importe environments.py de PolicyGradient, no el de DQN ---
PG_CODE_DIR = Path(__file__).resolve().parents[3] / "PolicyGradient" / "code"

# 1) Poner PG_CODE_DIR al inicio de sys.path
if str(PG_CODE_DIR) in sys.path:
    sys.path.remove(str(PG_CODE_DIR))
sys.path.insert(0, str(PG_CODE_DIR))
importlib.invalidate_caches()

# 2) Si ya existe un módulo llamado 'environments' (probablemente el de DQN),
#    eliminarlo de sys.modules para forzar que se importe el de PolicyGradient.
if "environments" in sys.modules:
    try:
        del sys.modules["environments"]
    except KeyError:
        pass

# 3) Intentar importar main desde PG_CODE_DIR. Si falla por conflictos, cargar por ruta.
try:
    from main import train_env
except Exception as e:
    # fallback: cargar main.py directamente desde PG_CODE_DIR con un nombre exclusivo
    main_path = PG_CODE_DIR / "main.py"
    if not main_path.exists():
        raise ImportError(f"No se encontró {main_path}") from e
    spec = importlib.util.spec_from_file_location("pg_main", str(main_path))
    pg_main = importlib.util.module_from_spec(spec)
    sys.modules["pg_main"] = pg_main
    spec.loader.exec_module(pg_main)
    train_env = getattr(pg_main, "train_env")


def compare_target_network_vs_no_target(
    env_id: str = "CartPole-v1",
    episodes: int = 500,
    runs: int = 3,
    seeds: list | None = None,
    target_update_freq: int = 1000,
    window: int = 50,
    out_path: str = "plots/compare_target_vs_no_target.png",
):
    """
    Experimento: compara DQN con target network frente a DQN SIN target network (usamos la policy_net como target).
    Ejecutamos varias veces (runs veces), cada vez usando una seed distinta.
    Usamos el promedio de todas las ejecuciones. 
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    if seeds is None:
        seeds = list(range(runs))

    results = {"with_target": [], "no_target": []}

    for seed in seeds[:runs]:
        overrides = {
            "env_id": env_id,
            "max_episodes": episodes,
            "max_steps_per_episode": 500,
            "seed": int(seed),
            "target_update_freq": int(target_update_freq),
            "buffer_capacity": 50000,
            "batch_size": 64,
            "learning_starts": 1000,
            "lr": 1e-3,
            "epsilon_start": 1.0,
            "epsilon_end": 0.01,
            "epsilon_decay_rate": 0.0001,
            "checkpoint_path": f"dqn_{env_id}_seed{seed}.pt",
            "log_dir": f"runs/dqn_{env_id}_seed{seed}",
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }

        # ------------------ Con target network ------------------
        cfg = DQNConfig()
        for k, v in overrides.items():
            setattr(cfg, k, v)
        agent = DQNAgent(cfg)
        summary = agent.train()
        rewards = list(summary.get("rewards", agent.episode_rewards))
        results["with_target"].append(np.asarray(rewards, dtype=np.float32))

        # ------------------ Sin target network: hacemos target_net = a policy_net ------------------
        cfg_nt = DQNConfig()
        for k, v in overrides.items():
            setattr(cfg_nt, k, v)
        agent_nt = DQNAgent(cfg_nt)
        # forzamos que target_net apunte a policy_net (usa la misma instancia)
        agent_nt.target_net = agent_nt.policy_net 
        summary_nt = agent_nt.train()
        rewards_nt = list(summary_nt.get("rewards", agent_nt.episode_rewards))
        results["no_target"].append(np.asarray(rewards_nt, dtype=np.float32))

    min_len_with = min(r.shape[0] for r in results["with_target"])
    min_len_no = min(r.shape[0] for r in results["no_target"])
    L = min(min_len_with, min_len_no)

    arr_with = np.vstack([r[:L] for r in results["with_target"]])
    arr_no = np.vstack([r[:L] for r in results["no_target"]])

    mean_with = arr_with.mean(axis=0)
    std_with = arr_with.std(axis=0)
    mean_no = arr_no.mean(axis=0)
    std_no = arr_no.std(axis=0)

    def moving_avg(x, w):
        if x.size < w:
            return x
        return np.convolve(x, np.ones(w)/w, mode="valid")

    ma_mean_with = moving_avg(mean_with, window)
    ma_mean_no = moving_avg(mean_no, window)
    ma_std_with = moving_avg(std_with, window)
    ma_std_no = moving_avg(std_no, window)

    x_with = np.arange(ma_mean_with.size)
    x_no = np.arange(ma_mean_no.size)

    plt.figure(figsize=(10, 6))
    plt.plot(x_with, ma_mean_with, label="DQN con target network", c="tab:blue")
    plt.fill_between(x_with, ma_mean_with - ma_std_with, ma_mean_with + ma_std_with, color="tab:blue", alpha=0.2)
    plt.plot(x_no, ma_mean_no, label="DQN sin target network", c="tab:orange")
    plt.fill_between(x_no, ma_mean_no - ma_std_no, ma_mean_no + ma_std_no, color="tab:orange", alpha=0.2)
    plt.xlabel("Episodios (media móvil)")
    plt.ylabel("Reward promedio")
    plt.title(f"Comparación: target network vs no target ({env_id})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.show()
    plt.close()

    return {
        "with_target": results["with_target"],
        "no_target": results["no_target"],
        "mean_with": mean_with,
        "mean_no": mean_no,
        "out_path": out_path,
    }


def compare_replay_vs_no_replay(env_id: str = "CartPole-v1", episodes: int = 500, runs: int = 3, seeds: list | None = None, window: int = 50, out_path: str = "plots/compare_replay_vs_no_replay.png"):
    """
    Compara DQN usando replay buffer vs no usarlo
    (batch_size=1, buffer_capacity=1, learning_starts=0)
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    if seeds is None:
        seeds = list(range(runs))

    results = {"with_replay": [], "no_replay": []}

    for seed in seeds[:runs]:
        base_overrides = {
            "env_id": env_id,
            "max_episodes": episodes,
            "max_steps_per_episode": 500,
            "seed": int(seed),
            "buffer_capacity": 50000,
            "batch_size": 64,
            "learning_starts": 1000,
            "train_freq": 1,
            "target_update_freq": 1000,
            "lr": 1e-3,
            "epsilon_start": 1.0,
            "epsilon_end": 0.01,
            "epsilon_decay_rate": 0.0001,
            "checkpoint_path": f"dqn_{env_id}_replay_seed{seed}.pt",
            "log_dir": f"runs/dqn_{env_id}_replay_seed{seed}",
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }

        # With replay (no cambio nada)
        cfg = DQNConfig()
        for k, v in base_overrides.items():
            setattr(cfg, k, v)
        agent = DQNAgent(cfg)
        summary = agent.train()
        rewards = list(summary.get("rewards", agent.episode_rewards))
        results["with_replay"].append(np.asarray(rewards, dtype=np.float32))

        # Sin replay : batch_size=1, buffer_capacity=1, learning_starts=0
        no_r_overrides = {**base_overrides,
                          "buffer_capacity": 1,
                          "batch_size": 1,
                          "learning_starts": 0,
                          "checkpoint_path": f"dqn_{env_id}_noreplay_seed{seed}.pt",
                          "log_dir": f"runs/dqn_{env_id}_noreplay_seed{seed}"}
        cfg_nr = DQNConfig()
        for k, v in no_r_overrides.items():
            setattr(cfg_nr, k, v)
        agent_nr = DQNAgent(cfg_nr)
        summary_nr = agent_nr.train()
        rewards_nr = list(summary_nr.get("rewards", agent_nr.episode_rewards))
        results["no_replay"].append(np.asarray(rewards_nr, dtype=np.float32))

    min_len_with = min(r.shape[0] for r in results["with_replay"])
    min_len_no = min(r.shape[0] for r in results["no_replay"])
    L = min(min_len_with, min_len_no)

    arr_with = np.vstack([r[:L] for r in results["with_replay"]])
    arr_no = np.vstack([r[:L] for r in results["no_replay"]])

    mean_with = arr_with.mean(axis=0)
    std_with = arr_with.std(axis=0)
    mean_no = arr_no.mean(axis=0)
    std_no = arr_no.std(axis=0)

    def moving_avg(x, w):
        if x.size < w:
            return x
        return np.convolve(x, np.ones(w)/w, mode="valid")

    ma_mean_with = moving_avg(mean_with, window)
    ma_mean_no = moving_avg(mean_no, window)
    ma_std_with = moving_avg(std_with, window)
    ma_std_no = moving_avg(std_no, window)

    x = np.arange(ma_mean_with.size)

    plt.figure(figsize=(10, 6))
    plt.plot(x, ma_mean_with, label="DQN with replay", c="tab:blue")
    plt.fill_between(x, ma_mean_with - ma_std_with, ma_mean_with + ma_std_with, color="tab:blue", alpha=0.2)
    plt.plot(x, ma_mean_no, label="DQN no replay (online)", c="tab:orange")
    plt.fill_between(x, ma_mean_no - ma_std_no, ma_mean_no + ma_std_no, color="tab:orange", alpha=0.2)
    plt.xlabel("Episodios (media móvil)")
    plt.ylabel("Reward promedio")
    plt.title(f"Replay buffer vs No-replay ({env_id})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)  
    plt.show()
    plt.close()

    return {
        "with_replay": results["with_replay"],
        "no_replay": results["no_replay"],
        "mean_with": mean_with,
        "mean_no": mean_no,
        "out_path": out_path,
    }


def CartPole_DQN_vs_REINFORCE():
    """ Compara el desempeño de DQN y REINFORCE en CartPole-v1 """
    config_overrides={
        "env_id" : "CartPole-v1",
        "gamma" : 0.99,
        "lr" : 1e-3,
        "batch_size" : 64,
        "buffer_capacity" : 50000,
        "learning_starts" : 1000,
        "train_freq" : 1,
        "target_update_freq" : 1000,
        "max_episodes" : 500,
        "max_steps_per_episode" : 500, # la consigna dice 1000, pero en https://gymnasium.farama.org/environments/classic_control/cart_pole/ dice 500
        "seed" : 0,
        "epsilon_start" : 1.0,
        "epsilon_end" : 0.01,
        "epsilon_decay_rate" : 0.0001,
        "log_dir" : "runs/dqn_cartpole",
        "checkpoint_path" : "dqn_cartpole.pt",
        "device" : "cuda" if torch.cuda.is_available() else "cpu"
    }
    # Entrenar DQN
    dqn_agent, dqn_results = train_dqn_cartpole(config_overrides=config_overrides)

    # Entrenar REINFORCE
    reinforce_rewards = train_env("CartPole-v1", episodes=500, obs=np.array([0.0, 0.0, 0.0, 0.0]), batch_size=10, early_stop=False)

    # Resultados
    dqn_mean_reward = np.mean(dqn_agent.episode_rewards)
    dqn_std_reward = np.std(dqn_agent.episode_rewards)
    reinforce_mean_reward = np.mean(reinforce_rewards)
    reinforce_std_reward = np.std(reinforce_rewards)

    plt.figure(figsize=(10, 6))
    plt.plot(dqn_agent.episode_rewards, label='DQN Rewards', alpha=0.7)
    plt.plot(reinforce_rewards, label='REINFORCE Rewards', alpha=0.7)
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.title('DQN vs REINFORCE on CartPole-v1')
    plt.legend()
    plt.grid()
    plt.savefig("plots/dqn_vs_reinforce_cartpole.png")
    plt.show()
    plt.close()

    print(f"DQN CartPole-v1: mean={dqn_mean_reward:.2f} ± {dqn_std_reward:.2f}")
    print(f"REINFORCE CartPole-v1: mean={reinforce_mean_reward:.2f} ± {reinforce_std_reward:.2f}")



def train_dqn_cartpole(config_overrides=None):
    cfg = DQNConfig()

    if not config_overrides:
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

    # 4.1: Graficar comparacion DQN vs REINFORCE en CartPole-v1
    # CartPole_DQN_vs_REINFORCE()

    # Experimento 2: Comparar DQN con y sin target network
    # compare_target_network_vs_no_target()

    # Experimento 3: Comparar DQN con replay buffer vs sin replay (actualización online)
    compare_replay_vs_no_replay()

    pass