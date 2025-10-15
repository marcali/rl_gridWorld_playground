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


def create_all_visualizations(experiment_dir, trained_results=None, random_results=None):
    """
    Generate all visualizations for an experiment

    Args:
        experiment_dir: Directory containing the CSV files
        trained_results: Dictionary with trained agent metrics (optional)
        random_results: Dictionary with random agent metrics (optional)
    """
    exp_path = Path(experiment_dir)
    train_csv = exp_path / "training_metrics.csv"
    eval_csv = exp_path / "evaluation_metrics.csv"

    # Create plots
    if train_csv.exists():
        plot_training_metrics(train_csv, save_dir=exp_path)
        plot_learning_curve(train_csv, save_dir=exp_path)
    else:
        print("Warning: Training CSV not found")

    if eval_csv.exists():
        plot_evaluation_metrics(eval_csv, save_dir=exp_path)
    else:
        print("Warning: Evaluation CSV not found")

    # Create agent comparison plot if both results are provided
    if trained_results is not None and random_results is not None:
        plot_agent_comparison(trained_results, random_results, save_dir=exp_path)

    print("-" * 60)
    print(f"All visualizations saved to: {exp_path}\n")
