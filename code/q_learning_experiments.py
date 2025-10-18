import numpy as np
import matplotlib.pyplot as plt
import csv
import os
from q_learning import QLearningAgent, QLearningConfig


def moving_average(arr, window: int = 100):
    arr = np.asarray(arr, dtype=np.float32)
    if len(arr) < window:
        return np.array([], dtype=np.float32)
    return np.convolve(arr, np.ones(window)/window, mode="valid")

def run_experiment(cfg: QLearningConfig):
    """
    Corre un experimento con una config dada, devuelve dict con:
      - 'cfg' (la config)
      - 'rewards' (lista recompensas por episodio)
      - 'epsilons' (lista epsilons por episodio)
      - 'sr_greedy' (% exito en 100 ep con epsilon=0.0)
      - 'summary'   (dict con resumen)
    """
    agent = QLearningAgent(cfg)
    agent.train()
    sr_greedy = agent.evaluate_success_rate(episodes=100, epsilon=0.0)
    return {
        "cfg": cfg,
        "rewards": agent.episode_rewards[:],
        "epsilons": agent.epsilons[:],
        "sr_greedy": sr_greedy,
        "summary": agent.summary(),
    }

def _label_from_cfg(cfg: QLearningConfig):
    return f"α={cfg.alpha}, γ={cfg.gamma}, decay={cfg.decay_rate}, slip={cfg.is_slippery}"

def compare_hyperparams(
    configs: list[QLearningConfig],
    window: int = 100,
    savefig: str = "q_learning_comparison.png",
    outdir: str = "experiments_q_learning",
):
    """
    Corre multiples configuraciones, grafica la media móvil de rewards,
    y exporta un CSV con el resumen de cada corrida.
    """
    os.makedirs(outdir, exist_ok=True)
    results = []

    plt.figure()
    for i, cfg in enumerate(configs, start=1):
        print(f"\n=== Experimento {i}/{len(configs)}: {_label_from_cfg(cfg)} ===")
        res = run_experiment(cfg)
        results.append(res)

        ma = moving_average(res["rewards"], window=window)
        if ma.size > 0:
            plt.plot(ma, label=_label_from_cfg(cfg))
        else:
            # Caso borde: muy pocos episodios para la ventana
            plt.plot(res["rewards"], label=_label_from_cfg(cfg))

        # Guardamos recompensas por episodio en CSV individual
        rewards_path = os.path.join(outdir, f"rewards_{i}.csv")
        with open(rewards_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["episode", "reward"])
            for ep, r in enumerate(res["rewards"], start=1):
                w.writerow([ep, r])

    plt.title(f"Comparación de recompensas (media móvil, window={window})")
    plt.xlabel("Episodios (suavizados)")
    plt.ylabel("Reward promedio")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, savefig))
    plt.show()

    # CSV con resumen final de cada config
    summary_path = os.path.join(outdir, "summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "label","episodes","alpha","gamma","epsilon_init","min_epsilon","max_epsilon",
            "decay_rate","use_decay","is_slippery","map_name",
            "final_avg_reward_last_window","sr_greedy_100"
        ])
        for res in results:
            cfg = res["cfg"]
            summ = res["summary"]
            w.writerow([
                _label_from_cfg(cfg),
                cfg.episodes,
                cfg.alpha,
                cfg.gamma,
                cfg.epsilon,
                cfg.min_epsilon,
                cfg.max_epsilon,
                cfg.decay_rate,
                cfg.use_decay,
                cfg.is_slippery,
                cfg.map_name,
                summ.get("final_avg_reward_last_window", 0.0),
                res["sr_greedy"]
            ])

    # Tambn imprimimos por la terminal
    print("\n===== Resumen de experimentos =====")
    for res in results:
        cfg = res["cfg"]
        summ = res["summary"]
        print(f"{_label_from_cfg(cfg)} | "
              f"final_avg(last_window)={summ.get('final_avg_reward_last_window', 0.0):.3f} | "
              f"SR_greedy(100)={res['sr_greedy']:.1f}% | "
              f"SR_eps0.1(100)={res['sr_eps01']:.1f}%")

    return results


if __name__ == "__main__":
    config1 = QLearningConfig(
        episodes=5000,
        alpha=0.8,
        gamma=0.99,
        epsilon=1.0,
        min_epsilon=0.01,
        max_epsilon=1.0,
        decay_rate=0.001,
        is_slippery=True,
        use_decay=True,
        log_every=200,
        seed=0,
        map_name="4x4",
        # max_steps_per_episode para 4x4
        max_steps_per_episode=100,
    )

    config2 = QLearningConfig(
        episodes=5000,
        alpha=0.6,
        gamma=0.99,
        epsilon=1.0,
        min_epsilon=0.01,
        max_epsilon=1.0,
        decay_rate=0.001,
        is_slippery=True,
        use_decay=True,
        log_every=200,
        seed=0,
        map_name="4x4",
        # max_steps_per_episode para 4x4
        max_steps_per_episode=100,
    )

    config3 = QLearningConfig(
        episodes=5000,
        alpha=0.8,
        gamma=0.99,
        epsilon=1.0,
        min_epsilon=0.01,
        max_epsilon=1.0,
        decay_rate=0.002,
        is_slippery=True,
        use_decay=True,
        log_every=200,
        seed=0,
        map_name="4x4",
        # max_steps_per_episode para 4x4
        max_steps_per_episode=100,
    )
    # Armamos algunas variantes para comparar
    configs = [config1, config2, config3]

    compare_hyperparams(configs, window=100, savefig="comparison_4x4.png")
