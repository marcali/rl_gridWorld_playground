"""Hyperparameter grid search - Find optimal settings"""

import numpy as np
import pandas as pd
import itertools
from datetime import datetime
from pathlib import Path
from environment import Environment
from agents import QLearningAgent
import config
from run import train, evaluate
import matplotlib.pyplot as plt


def train_and_evaluate(alpha, gamma, epsilon_decay, eval_epsilon, tolerance, n_episodes, env):
    """
    Train agent with given hyperparameters and evaluate using run.py functions

    Returns:
        Dictionary with results
    """
    # Train agent using run.py train function in silent mode
    agent, logger, training_success_rate = train(
        n_episodes=n_episodes,
        epsilon_start=config.EPSILON_START,
        epsilon_end=config.EPSILON_END,
        epsilon_decay=epsilon_decay,
        alpha=alpha,
        gamma=gamma,
        experiment_name=None,
        silent=True,
        env=env,
    )

    # Evaluate trained agent using run.py evaluate function with custom epsilon and tolerance
    eval_results = evaluate(
        agent,
        None,
        env,
        n_episodes=config.GRID_SEARCH_N_EVAL,
        silent=True,
        eval_epsilon=eval_epsilon,
        tolerance=tolerance,
    )

    return {
        "alpha": alpha,
        "gamma": gamma,
        "epsilon_decay": epsilon_decay,
        "eval_epsilon": eval_epsilon,
        "tolerance": tolerance,
        "training_success_rate": training_success_rate,
        "eval_success_rate": eval_results["success_rate"],
        "eval_avg_reward": eval_results["avg_reward"],
        "eval_std_reward": eval_results["std_reward"],
        "eval_avg_steps": eval_results["avg_steps"],
        "eval_std_steps": eval_results["std_steps"],
    }


def run_grid_search():
    """
    Run grid search over hyperparameters
    """
    print("=" * 80)
    print("HYPERPARAMETER GRID SEARCH")
    print("=" * 80)

    # Create parameter grid
    param_grid = list(
        itertools.product(
            config.GRID_SEARCH_ALPHA,
            config.GRID_SEARCH_GAMMA,
            config.GRID_SEARCH_EPSILON_DECAY,
            config.GRID_SEARCH_EVAL_EPSILON,
            config.GRID_SEARCH_TOLERANCE,
        )
    )

    total_configs = len(param_grid)

    # Create environment (reused for all experiments)
    env = Environment()

    # Store results
    results = []

    # Run grid search
    for idx, (alpha, gamma, epsilon_decay, eval_epsilon, tolerance) in enumerate(param_grid, 1):
        print(
            f"\n[{idx}/{total_configs}] Testing: alpha={alpha}, gamma={gamma}, epsilon_decay={epsilon_decay}, eval_epsilon={eval_epsilon}, tolerance={tolerance}"
        )

        try:
            result = train_and_evaluate(
                alpha=alpha,
                gamma=gamma,
                epsilon_decay=epsilon_decay,
                eval_epsilon=eval_epsilon,
                tolerance=tolerance,
                n_episodes=config.GRID_SEARCH_N_EPISODES,
                env=env,
            )
            results.append(result)

        except Exception as e:
            print(f"    ✗ ERROR: {e}")
            continue

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Save results
    output_dir = Path("results/grid_search")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"grid_search_{timestamp}.csv"
    df.to_csv(csv_path, index=False)

    # Sort by eval success rate (primary), then avg steps (secondary)
    df_sorted = df.sort_values(by=["eval_success_rate", "eval_avg_steps"], ascending=[False, True])

    # Print formatted table
    print(
        f"{'Rank':<6} {'Alpha':<7} {'Gamma':<7} {'Eps.Decay':<11} {'Eval.Eps':<10} {'Tol':<7} | "
        f"{'Train%':<8} {'Eval%':<8} {'Steps':<8} {'Reward':<10}"
    )
    print("-" * 97)

    for idx, row in enumerate(df_sorted.itertuples(), 1):
        print(
            f"{idx:<6} {row.alpha:<7.2f} {row.gamma:<7.2f} {row.epsilon_decay:<11.3f} {row.eval_epsilon:<10.2f} {row.tolerance:<7.3f} | "
            f"{row.training_success_rate:<8.1f} {row.eval_success_rate:<8.1f} "
            f"{row.eval_avg_steps:<8.1f} {row.eval_avg_reward:<10.1f}"
        )

    best = df_sorted.iloc[0]

    print(f"\n🏆 BEST HYPERPARAMETERS:")
    print(f"   alpha = {best['alpha']}")
    print(f"   gamma = {best['gamma']}")
    print(f"   epsilon_decay = {best['epsilon_decay']}")
    print(f"   eval_epsilon = {best['eval_epsilon']}")
    print(f"   tolerance = {best['tolerance']}")

    print(f"\n📊 EXPECTED PERFORMANCE:")
    print(f"   Success Rate: {best['eval_success_rate']:.1f}%")
    print(f"   Avg Steps to Goal: {best['eval_avg_steps']:.1f}")
    print(f"   Avg Reward: {best['eval_avg_reward']:.1f}")

    # Create visualization
    visualize_grid_search_results(df_sorted, output_dir / f"grid_search_{timestamp}.png")

    return df_sorted


def visualize_grid_search_results(df, save_path):
    """Create visualization of grid search results"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Hyperparameter Grid Search Results", fontsize=16, fontweight="bold")

    # 1. Success Rate by Alpha
    ax = axes[0, 0]
    for gamma in df["gamma"].unique():
        data = df[df["gamma"] == gamma].groupby("alpha")["eval_success_rate"].mean()
        ax.plot(data.index, data.values, marker="o", label=f"γ={gamma}", linewidth=2)
    ax.set_xlabel("Alpha (Learning Rate)", fontsize=12)
    ax.set_ylabel("Eval Success Rate (%)", fontsize=12)
    ax.set_title("Success Rate vs Learning Rate", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Success Rate by Gamma
    ax = axes[0, 1]
    for alpha in df["alpha"].unique():
        data = df[df["alpha"] == alpha].groupby("gamma")["eval_success_rate"].mean()
        ax.plot(data.index, data.values, marker="o", label=f"α={alpha}", linewidth=2)
    ax.set_xlabel("Gamma (Discount Factor)", fontsize=12)
    ax.set_ylabel("Eval Success Rate (%)", fontsize=12)
    ax.set_title("Success Rate vs Discount Factor", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Success Rate by Epsilon Decay
    ax = axes[1, 0]
    for alpha in df["alpha"].unique():
        data = df[df["alpha"] == alpha].groupby("epsilon_decay")["eval_success_rate"].mean()
        ax.plot(data.index, data.values, marker="o", label=f"α={alpha}", linewidth=2)
    ax.set_xlabel("Epsilon Decay", fontsize=12)
    ax.set_ylabel("Eval Success Rate (%)", fontsize=12)
    ax.set_title("Success Rate vs Epsilon Decay", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Steps vs Success Rate (scatter)
    ax = axes[1, 1]
    scatter = ax.scatter(
        df["eval_avg_steps"],
        df["eval_success_rate"],
        c=df["eval_avg_reward"],
        s=100,
        cmap="RdYlGn",
        alpha=0.7,
        edgecolors="black",
    )
    ax.set_xlabel("Average Steps", fontsize=12)
    ax.set_ylabel("Success Rate (%)", fontsize=12)
    ax.set_title("Efficiency vs Success (color = avg reward)", fontweight="bold")
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Avg Reward", rotation=270, labelpad=20)

    # Mark best point
    best_idx = df["eval_success_rate"].idxmax()
    best_row = df.loc[best_idx]
    ax.scatter(
        best_row["eval_avg_steps"],
        best_row["eval_success_rate"],
        s=300,
        marker="*",
        c="gold",
        edgecolors="black",
        linewidths=2,
        label="Best",
        zorder=10,
    )
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n✓ Grid search visualization saved to: {save_path}")
    plt.close(fig)


def print_selection_criteria():
    """Explain how to select best hyperparameters"""
    print("\n" + "=" * 80)
    print("HOW TO PICK BEST HYPERPARAMETERS")
    print("=" * 80)

    print(
        """
For this GridWorld navigation task, hyperparameters should be selected based on:

1. PRIMARY: Evaluation Success Rate (highest)
   ⭐ Most important: Can the agent reach the goal?
   - Target: >70% for good performance
   - <30% = Poor learning
   - 30-70% = Acceptable
   - >70% = Good

2. SECONDARY: Average Steps (lowest, among high success configs)
   ⚡ Efficiency: How quickly does it reach the goal?
   - Lower is better (optimal path ≈ 18 steps for 10x10 grid)
   - Penalize wandering behavior

3. TERTIARY: Average Reward (highest)
   💰 Combined metric of success and efficiency
   - Success + few steps = high reward
   - No success or many steps = low reward

4. VALIDATION: Training Success Rate
   ✓ Should be similar to eval success rate
   - Large gap = Overfitting to training obstacles
   - Similar values = Good generalization

5. VALIDATION: Generalization
   🎯 Does agent perform consistently?
   - Similar train/eval rates = Good generalization
   - Large gap = Overfitting to training obstacles

DECISION PROCESS:
1. Filter configurations with eval success rate >70%
2. Among these, pick the one with lowest average steps
3. If no configs reach 70%, pick highest success rate
4. Validate train/eval rates are similar for selected config

WHY THIS PRIORITY?
- Success rate matters most (reaching goal is primary objective)
- Steps matter for efficiency (shorter paths are better)
- Consistent performance validates that learning generalizes well
    """
    )


if __name__ == "__main__":
    print_selection_criteria()

    results_df = run_grid_search()
