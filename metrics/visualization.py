"""Visualization utilities for training metrics"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path


def plot_training_metrics(csv_path, save_dir=None):
    """
    Create comprehensive training visualization

    Args:
        csv_path: Path to training metrics CSV file
        save_dir: Directory to save plots (if None, only display)
    """
    # Load data
    df = pd.read_csv(csv_path)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Training Metrics", fontsize=16, fontweight="bold")

    # Plot 1: Episode Rewards
    axes[0, 0].plot(df["episode"], df["total_reward"], alpha=0.6, label="Episode Reward")
    axes[0, 0].plot(
        df["episode"],
        df["avg_reward_last_100"],
        color="red",
        linewidth=2,
        label="Avg Last 100",
    )
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Total Reward")
    axes[0, 0].set_title("Reward per Episode")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Steps per Episode
    axes[0, 1].plot(df["episode"], df["steps"], color="green", alpha=0.6)
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Steps")
    axes[0, 1].set_title("Steps per Episode")
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Success Rate Over Time
    if "success" in df.columns:
        # Calculate rolling success rate
        window = 50
        success_rate = df["success"].rolling(window=window, min_periods=1).mean() * 100
        axes[1, 0].plot(df["episode"], success_rate, color="orange", linewidth=2)
        axes[1, 0].set_xlabel("Episode")
        axes[1, 0].set_ylabel("Success Rate (%)")
        axes[1, 0].set_title(f"Success Rate Over Time (MA{window})")
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim(0, 100)
    else:
        axes[1, 0].text(
            0.5,
            0.5,
            "Success data not available",
            ha="center",
            va="center",
            transform=axes[1, 0].transAxes,
        )
        axes[1, 0].set_title("Success Rate Over Time")

    # Plot 4: Reward Distribution (last 500 episodes)
    last_n = min(500, len(df))
    axes[1, 1].hist(
        df["total_reward"].tail(last_n), bins=30, color="purple", alpha=0.7, edgecolor="black"
    )
    axes[1, 1].set_xlabel("Total Reward")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].set_title(f"Reward Distribution (Last {last_n} Episodes)")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save or display
    if save_dir:
        save_path = Path(save_dir) / "training_metrics.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    plt.close()


def plot_evaluation_metrics(csv_path, save_dir=None):
    """
    Create evaluation visualization

    Args:
        csv_path: Path to evaluation metrics CSV file
        save_dir: Directory to save plots (if None, only display)
    """
    # Load data
    df = pd.read_csv(csv_path)

    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Evaluation Metrics", fontsize=16, fontweight="bold")

    # Plot 1: Rewards
    axes[0].bar(df["episode"], df["total_reward"], color="skyblue", edgecolor="black")
    axes[0].axhline(df["total_reward"].mean(), color="red", linestyle="--", label="Mean")
    axes[0].set_xlabel("Evaluation Episode")
    axes[0].set_ylabel("Total Reward")
    axes[0].set_title("Evaluation Rewards")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis="y")

    # Plot 2: Steps
    axes[1].bar(df["episode"], df["steps"], color="lightgreen", edgecolor="black")
    axes[1].axhline(df["steps"].mean(), color="red", linestyle="--", label="Mean")
    axes[1].set_xlabel("Evaluation Episode")
    axes[1].set_ylabel("Steps")
    axes[1].set_title("Steps per Episode")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")

    # Plot 3: Success Rate
    success_rate = df["success"].mean() * 100
    axes[2].bar(
        ["Success", "Failure"],
        [df["success"].sum(), len(df) - df["success"].sum()],
        color=["green", "red"],
        edgecolor="black",
        alpha=0.7,
    )
    axes[2].set_ylabel("Count")
    axes[2].set_title(f"Success Rate: {success_rate:.1f}%")
    axes[2].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    # Save or display
    if save_dir:
        save_path = Path(save_dir) / "evaluation_metrics.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    plt.close()


def plot_agent_comparison(trained_results, random_results, save_dir=None):
    """
    Create comparison visualization between trained and random agents

    Args:
        trained_results: Dictionary with trained agent metrics
        random_results: Dictionary with random agent metrics
        save_dir: Directory to save plots (if None, only display)
    """
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Agent Performance Comparison", fontsize=16, fontweight="bold")

    # Data for comparison
    agents = ["Trained Agent", "Random Agent"]
    success_rates = [trained_results["success_rate"], random_results["success_rate"]]
    avg_rewards = [trained_results["avg_reward"], random_results["avg_reward"]]
    avg_steps = [trained_results["avg_steps"], random_results["avg_steps"]]

    # Colors
    colors = ["#2E8B57", "#DC143C"]  # Sea Green and Crimson

    # Plot 1: Success Rate Comparison
    bars1 = axes[0].bar(agents, success_rates, color=colors, alpha=0.7, edgecolor="black")
    axes[0].set_ylabel("Success Rate (%)")
    axes[0].set_title("Success Rate Comparison")
    axes[0].set_ylim(0, 100)
    axes[0].grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, value in zip(bars1, success_rates):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Plot 2: Average Reward Comparison
    bars2 = axes[1].bar(agents, avg_rewards, color=colors, alpha=0.7, edgecolor="black")
    axes[1].set_ylabel("Average Reward")
    axes[1].set_title("Average Reward Comparison")
    axes[1].grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, value in zip(bars2, avg_rewards):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(avg_rewards) * 0.01,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Plot 3: Average Steps Comparison
    bars3 = axes[2].bar(agents, avg_steps, color=colors, alpha=0.7, edgecolor="black")
    axes[2].set_ylabel("Average Steps")
    axes[2].set_title("Average Steps Comparison\n(Lower is Better)")
    axes[2].grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, value in zip(bars3, avg_steps):
        axes[2].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(avg_steps) * 0.01,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()

    # Save or display
    if save_dir:
        save_path = Path(save_dir) / "agent_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    plt.close()


def plot_learning_curve(csv_path, save_dir=None, window=100):
    """
    Create a focused learning curve plot

    Args:
        csv_path: Path to training metrics CSV file
        save_dir: Directory to save plot
        window: Window size for moving average
    """
    df = pd.read_csv(csv_path)

    plt.figure(figsize=(12, 6))

    # Plot raw rewards with transparency
    plt.plot(
        df["episode"],
        df["total_reward"],
        alpha=0.3,
        color="blue",
        label="Episode Reward",
    )

    # Plot moving average
    plt.plot(
        df["episode"],
        df["avg_reward_last_100"],
        color="red",
        linewidth=2,
        label=f"Moving Average ({window} episodes)",
    )

    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Total Reward", fontsize=12)
    plt.title("Learning Curve", fontsize=14, fontweight="bold")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save or display
    if save_dir:
        save_path = Path(save_dir) / "learning_curve.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    plt.close()


def plot_convergence_metrics(csv_path, save_dir=None):
    """
    Create convergence and Q-value evolution visualization

    Args:
        csv_path: Path to training metrics CSV file
        save_dir: Directory to save plots (if None, only display)
    """
    # Load data
    df = pd.read_csv(csv_path)

    # Check if convergence columns exist
    convergence_cols = [
        "mean_q_value",
        "std_q_value",
        "max_q_value",
        "min_q_value",
        "q_value_range",
        "explored_states",
        "convergence_rate",
    ]

    if not all(col in df.columns for col in convergence_cols):
        print("Warning: Convergence data not found in CSV file")
        return

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Algorithm Convergence and Q-Value Evolution", fontsize=16, fontweight="bold")

    # Plot 1: Q-Value Evolution
    axes[0, 0].plot(
        df["episode"], df["mean_q_value"], label="Mean Q-Value", color="blue", linewidth=2
    )
    axes[0, 0].fill_between(
        df["episode"],
        df["mean_q_value"] - df["std_q_value"],
        df["mean_q_value"] + df["std_q_value"],
        alpha=0.3,
        color="blue",
        label="±1 Std Dev",
    )
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Q-Value")
    axes[0, 0].set_title("Q-Value Evolution")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Q-Value Range (Max - Min)
    axes[0, 1].plot(df["episode"], df["q_value_range"], color="red", linewidth=2)
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Q-Value Range")
    axes[0, 1].set_title("Q-Value Range Over Time")
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Convergence Rate
    axes[0, 2].plot(df["episode"], df["convergence_rate"], color="green", linewidth=2)
    axes[0, 2].set_xlabel("Episode")
    axes[0, 2].set_ylabel("Convergence Rate")
    axes[0, 2].set_title("Algorithm Convergence Rate")
    axes[0, 2].grid(True, alpha=0.3)

    # Plot 4: Explored States
    axes[1, 0].plot(df["episode"], df["explored_states"], color="orange", linewidth=2)
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Number of Explored States")
    axes[1, 0].set_title("State Exploration Progress")
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 5: Q-Value Distribution (last 100 episodes)
    last_100 = df.tail(100)
    axes[1, 1].hist(last_100["mean_q_value"], bins=20, color="purple", alpha=0.7, edgecolor="black")
    axes[1, 1].set_xlabel("Mean Q-Value")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].set_title("Q-Value Distribution (Last 100 Episodes)")
    axes[1, 1].grid(True, alpha=0.3)

    # Plot 6: Convergence vs Performance
    ax6 = axes[1, 2]
    scatter = ax6.scatter(
        df["convergence_rate"], df["total_reward"], c=df["episode"], cmap="viridis", alpha=0.6, s=20
    )
    ax6.set_xlabel("Convergence Rate")
    ax6.set_ylabel("Episode Reward")
    ax6.set_title("Convergence vs Performance")
    ax6.grid(True, alpha=0.3)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax6)
    cbar.set_label("Episode")

    plt.tight_layout()

    # Save or display
    if save_dir:
        save_path = Path(save_dir) / "convergence_metrics.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    plt.close()


def plot_state_value_heatmap(agent, save_dir=None, grid_size=10, env=None):
    """
    Create a comprehensive heatmap showing the learned state values

    Args:
        agent: QLearningAgent instance
        save_dir: Directory to save plots (if None, only display)
        grid_size: Size of the grid (assumes square grid)
        env: Environment instance to show obstacles and goal
    """
    # Get state values (max Q-value for each state)
    state_values = agent.get_state_value_map()

    # Reshape to grid
    value_grid = state_values.reshape(grid_size, grid_size)

    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle("Q-Value Analysis: Learned State Values", fontsize=16, fontweight="bold")

    # Plot 1: State Value Heatmap with obstacles
    im1 = axes[0].imshow(
        value_grid, cmap="Reds", interpolation="nearest", vmin=0, vmax=value_grid.max()
    )

    # Add obstacles if environment is provided
    if env is not None:
        for i in range(grid_size):
            for j in range(grid_size):
                if env.grid[i, j] == 1:  # Static obstacle
                    axes[0].add_patch(
                        plt.Rectangle(
                            (j - 0.5, i - 0.5),
                            1,
                            1,
                            facecolor="black",
                            alpha=0.8,
                            edgecolor="white",
                        )
                    )
                elif env.grid[i, j] == 2:  # Random obstacle
                    axes[0].add_patch(
                        plt.Rectangle(
                            (j - 0.5, i - 0.5), 1, 1, facecolor="gray", alpha=0.8, edgecolor="white"
                        )
                    )

        # Mark start and goal positions
        start_row, start_col = env.start_state
        goal_row, goal_col = env.goal_state
        axes[0].plot(
            start_col,
            start_row,
            "go",
            markersize=15,
            markeredgecolor="white",
            markeredgewidth=2,
            label="Start",
        )
        axes[0].plot(
            goal_col,
            goal_row,
            "bo",
            markersize=15,
            markeredgecolor="white",
            markeredgewidth=2,
            label="Goal",
        )
        axes[0].legend(loc="upper right")

    axes[0].set_title("State Values (Max Q-Value per State)\nRed = High Value, White = Low Value")
    axes[0].set_xlabel("Column")
    axes[0].set_ylabel("Row")

    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=axes[0])
    cbar1.set_label("State Value (Max Q-Value)")

    # Add value annotations for important states
    for i in range(grid_size):
        for j in range(grid_size):
            if not np.isnan(value_grid[i, j]):
                # Only show values for states with significant Q-values
                if value_grid[i, j] > np.percentile(value_grid, 80):  # Top 20% of values
                    color = "white" if value_grid[i, j] > np.median(value_grid) else "black"
                    axes[0].text(
                        j,
                        i,
                        f"{value_grid[i, j]:.1f}",
                        ha="center",
                        va="center",
                        color=color,
                        fontsize=8,
                        fontweight="bold",
                    )

    # Plot 2: Q-Value Distribution and Statistics
    axes[1].hist(state_values, bins=30, color="red", alpha=0.7, edgecolor="black")
    axes[1].axvline(
        np.mean(state_values),
        color="blue",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {np.mean(state_values):.2f}",
    )
    axes[1].axvline(
        np.median(state_values),
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Median: {np.median(state_values):.2f}",
    )
    axes[1].axvline(
        np.max(state_values),
        color="orange",
        linestyle="--",
        linewidth=2,
        label=f"Max: {np.max(state_values):.2f}",
    )
    axes[1].set_xlabel("State Value")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Distribution of State Values")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Add statistics text
    stats_text = f"""Q-Value Statistics:
Total States: {len(state_values)}
Explored States: {np.count_nonzero(state_values > 0)}
Mean Value: {np.mean(state_values):.3f}
Std Dev: {np.std(state_values):.3f}
Range: {np.min(state_values):.3f} to {np.max(state_values):.3f}"""

    axes[1].text(
        0.02,
        0.98,
        stats_text,
        transform=axes[1].transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        fontsize=10,
        fontfamily="monospace",
    )

    plt.tight_layout()

    # Save or display
    if save_dir:
        save_path = Path(save_dir) / "q_value_heatmap_analysis.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Q-value heatmap saved to: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_action_value_heatmaps(agent, save_dir=None, grid_size=10, env=None):
    """
    Create separate heatmaps for each action showing Q-values

    Args:
        agent: QLearningAgent instance
        save_dir: Directory to save plots (if None, only display)
        grid_size: Size of the grid (assumes square grid)
        env: Environment instance to show obstacles and goal
    """
    action_names = ["UP", "DOWN", "LEFT", "RIGHT"]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "Q-Values by Action: How Good is Each Action in Each State", fontsize=16, fontweight="bold"
    )

    # Get Q-table
    q_table = agent.q_table
    q_grids = []

    # Reshape Q-values for each action
    for action in range(4):
        action_q_values = q_table[:, action].reshape(grid_size, grid_size)
        q_grids.append(action_q_values)

    # Create heatmaps for each action
    for action in range(4):
        row = action // 2
        col = action % 2

        # Get Q-values for this action
        q_grid = q_grids[action]

        # Create heatmap
        im = axes[row, col].imshow(
            q_grid, cmap="Reds", interpolation="nearest", vmin=0, vmax=q_table.max()
        )

        # Add obstacles if environment is provided
        if env is not None:
            for i in range(grid_size):
                for j in range(grid_size):
                    if env.grid[i, j] == 1:  # Static obstacle
                        axes[row, col].add_patch(
                            plt.Rectangle(
                                (j - 0.5, i - 0.5),
                                1,
                                1,
                                facecolor="black",
                                alpha=0.8,
                                edgecolor="white",
                            )
                        )
                    elif env.grid[i, j] == 2:  # Random obstacle
                        axes[row, col].add_patch(
                            plt.Rectangle(
                                (j - 0.5, i - 0.5),
                                1,
                                1,
                                facecolor="gray",
                                alpha=0.8,
                                edgecolor="white",
                            )
                        )

            # Mark start and goal positions
            start_row, start_col = env.start_state
            goal_row, goal_col = env.goal_state
            axes[row, col].plot(
                start_col,
                start_row,
                "go",
                markersize=12,
                markeredgecolor="white",
                markeredgewidth=2,
            )
            axes[row, col].plot(
                goal_col, goal_row, "bo", markersize=12, markeredgecolor="white", markeredgewidth=2
            )

        axes[row, col].set_title(f"Q-Values for Action: {action_names[action]}")
        axes[row, col].set_xlabel("Column")
        axes[row, col].set_ylabel("Row")

        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[row, col])
        cbar.set_label("Q-Value")

        # Add value annotations for high Q-values
        for i in range(grid_size):
            for j in range(grid_size):
                if not np.isnan(q_grid[i, j]) and q_grid[i, j] > np.percentile(q_grid, 85):
                    color = "white" if q_grid[i, j] > np.median(q_grid) else "black"
                    axes[row, col].text(
                        j,
                        i,
                        f"{q_grid[i, j]:.1f}",
                        ha="center",
                        va="center",
                        color=color,
                        fontsize=8,
                        fontweight="bold",
                    )

    plt.tight_layout()

    # Save or display
    if save_dir:
        save_path = Path(save_dir) / "action_q_value_heatmaps.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Action Q-value heatmaps saved to: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_optimal_policy_heatmap(agent, save_dir=None, grid_size=10, env=None):
    """
    Create a heatmap showing the optimal policy (best action for each state)

    Args:
        agent: QLearningAgent instance
        save_dir: Directory to save plots (if None, only display)
        grid_size: Size of the grid (assumes square grid)
        env: Environment instance to show obstacles and goal
    """
    # Get Q-table
    q_table = agent.q_table

    # Find best action for each state
    best_actions = np.argmax(q_table, axis=1)
    best_actions_grid = best_actions.reshape(grid_size, grid_size)

    # Get Q-values for best actions
    best_q_values = np.max(q_table, axis=1)
    best_q_values_grid = best_q_values.reshape(grid_size, grid_size)

    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Learned Optimal Policy", fontsize=16, fontweight="bold")

    # Plot 1: Best Actions
    action_names = ["UP", "DOWN", "LEFT", "RIGHT"]
    action_colors = ["red", "blue", "green", "orange"]

    # Create a custom colormap for actions
    from matplotlib.colors import ListedColormap

    action_cmap = ListedColormap(["white", "red", "blue", "green", "orange"])

    # Add 1 to actions so 0-3 becomes 1-4 for colormap
    best_actions_display = best_actions_grid + 1

    im1 = axes[0].imshow(
        best_actions_display, cmap=action_cmap, interpolation="nearest", vmin=0, vmax=4
    )

    # Add obstacles if environment is provided
    if env is not None:
        for i in range(grid_size):
            for j in range(grid_size):
                if env.grid[i, j] == 1:  # Static obstacle
                    axes[0].add_patch(
                        plt.Rectangle(
                            (j - 0.5, i - 0.5),
                            1,
                            1,
                            facecolor="black",
                            alpha=0.8,
                            edgecolor="white",
                        )
                    )
                elif env.grid[i, j] == 2:  # Random obstacle
                    axes[0].add_patch(
                        plt.Rectangle(
                            (j - 0.5, i - 0.5), 1, 1, facecolor="gray", alpha=0.8, edgecolor="white"
                        )
                    )

        # Mark start and goal positions
        start_row, start_col = env.start_state
        goal_row, goal_col = env.goal_state
        axes[0].plot(
            start_col,
            start_row,
            "go",
            markersize=15,
            markeredgecolor="white",
            markeredgewidth=2,
            label="Start",
        )
        axes[0].plot(
            goal_col,
            goal_row,
            "bo",
            markersize=15,
            markeredgecolor="white",
            markeredgewidth=2,
            label="Goal",
        )
        axes[0].legend(loc="upper right")

    axes[0].set_title("Learned Optimal Policy\n(Red=UP, Blue=DOWN, Green=LEFT, Orange=RIGHT)")
    axes[0].set_xlabel("Column")
    axes[0].set_ylabel("Row")

    # Add action labels
    for i in range(grid_size):
        for j in range(grid_size):
            if best_actions_grid[i, j] >= 0:  # Valid action
                axes[0].text(
                    j,
                    i,
                    action_names[best_actions_grid[i, j]][0],
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=12,
                    fontweight="bold",
                )

    # Plot 2: Q-Values for Best Actions
    im2 = axes[1].imshow(
        best_q_values_grid, cmap="Reds", interpolation="nearest", vmin=0, vmax=best_q_values.max()
    )

    # Add obstacles if environment is provided
    if env is not None:
        for i in range(grid_size):
            for j in range(grid_size):
                if env.grid[i, j] == 1:  # Static obstacle
                    axes[1].add_patch(
                        plt.Rectangle(
                            (j - 0.5, i - 0.5),
                            1,
                            1,
                            facecolor="black",
                            alpha=0.8,
                            edgecolor="white",
                        )
                    )
                elif env.grid[i, j] == 2:  # Random obstacle
                    axes[1].add_patch(
                        plt.Rectangle(
                            (j - 0.5, i - 0.5), 1, 1, facecolor="gray", alpha=0.8, edgecolor="white"
                        )
                    )

        # Mark start and goal positions
        axes[1].plot(
            start_col,
            start_row,
            "go",
            markersize=15,
            markeredgecolor="white",
            markeredgewidth=2,
            label="Start",
        )
        axes[1].plot(
            goal_col,
            goal_row,
            "bo",
            markersize=15,
            markeredgecolor="white",
            markeredgewidth=2,
            label="Goal",
        )
        axes[1].legend(loc="upper right")

    axes[1].set_title("Q-Values of Optimal Actions")
    axes[1].set_xlabel("Column")
    axes[1].set_ylabel("Row")

    # Add colorbar
    cbar = plt.colorbar(im2, ax=axes[1])
    cbar.set_label("Q-Value of Best Action")

    # Add value annotations for high Q-values
    for i in range(grid_size):
        for j in range(grid_size):
            if not np.isnan(best_q_values_grid[i, j]) and best_q_values_grid[i, j] > np.percentile(
                best_q_values_grid, 80
            ):
                color = (
                    "white" if best_q_values_grid[i, j] > np.median(best_q_values_grid) else "black"
                )
                axes[1].text(
                    j,
                    i,
                    f"{best_q_values_grid[i, j]:.1f}",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=8,
                    fontweight="bold",
                )

    plt.tight_layout()

    # Save or display
    if save_dir:
        save_path = Path(save_dir) / "optimal_policy_heatmap.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Optimal policy heatmap saved to: {save_path}")
    else:
        plt.show()

    plt.close()


def create_all_visualizations(
    experiment_dir, trained_results=None, random_results=None, agent=None, env=None
):
    """
    Generate all visualizations for an experiment

    Args:
        experiment_dir: Directory containing the CSV files
        trained_results: Dictionary with trained agent metrics (optional)
        random_results: Dictionary with random agent metrics (optional)
        agent: QLearningAgent instance for state value visualization (optional)
        env: Environment instance for showing obstacles and goal positions (optional)
    """
    exp_path = Path(experiment_dir)
    train_csv = exp_path / "training_metrics.csv"
    eval_csv = exp_path / "evaluation_metrics.csv"

    # Create plots
    if train_csv.exists():
        plot_training_metrics(train_csv, save_dir=exp_path)
        plot_learning_curve(train_csv, save_dir=exp_path)
        plot_convergence_metrics(train_csv, save_dir=exp_path)
    else:
        print("Warning: Training CSV not found")

    if eval_csv.exists():
        plot_evaluation_metrics(eval_csv, save_dir=exp_path)
    else:
        print("Warning: Evaluation CSV not found")

    # Create agent comparison plot if both results are provided
    if trained_results is not None and random_results is not None:
        plot_agent_comparison(trained_results, random_results, save_dir=exp_path)

    # Create Q-value heatmap visualizations if agent is provided
    if agent is not None:
        print("\nGenerating Q-value heatmap visualizations...")
        plot_state_value_heatmap(agent, save_dir=exp_path, env=env)
        plot_action_value_heatmaps(agent, save_dir=exp_path, env=env)
        plot_optimal_policy_heatmap(agent, save_dir=exp_path, env=env)

    print("-" * 60)
    print(f"All visualizations saved to: {exp_path}\n")
