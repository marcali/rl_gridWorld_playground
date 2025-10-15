"""Metrics logging and tracking for training"""

import csv
import os
from datetime import datetime
from pathlib import Path


class MetricsLogger:
    """Logger for tracking and saving training metrics"""

    def __init__(self, log_dir="results", experiment_name=None):
        """
        Initialize metrics logger

        Args:
            log_dir: Directory to save logs
            experiment_name: Name of the experiment (defaults to timestamp)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Create experiment subdirectory
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = self.log_dir / experiment_name
        self.experiment_dir.mkdir(exist_ok=True)

        # CSV file paths
        self.train_csv = self.experiment_dir / "training_metrics.csv"
        self.eval_csv = self.experiment_dir / "evaluation_metrics.csv"

        # Initialize storage
        self.training_data = []
        self.evaluation_data = []

        # Initialize CSV files
        self._init_csv_files()

    def _init_csv_files(self):
        """Initialize CSV files with headers"""
        # Training CSV
        with open(self.train_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "episode",
                    "total_reward",
                    "steps",
                    "avg_reward_last_100",
                    "success",
                    "goal_rewards",
                    "step_penalties",
                    "collision_penalties",
                    "timestamp",
                ]
            )

        # Evaluation CSV
        with open(self.eval_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "total_reward", "steps", "success", "timestamp"])

    def log_training_episode(
        self,
        episode,
        total_reward,
        steps,
        avg_reward_last_100,
        success=False,
        goal_rewards=0,
        step_penalties=0,
        collision_penalties=0,
    ):
        """
        Log metrics for a training episode

        Args:
            episode: Episode number
            total_reward: Total reward obtained
            steps: Number of steps taken
            avg_reward_last_100: Average reward over last 100 episodes
            success: Whether the agent reached the goal
            goal_rewards: Rewards from reaching goals
            step_penalties: Penalties from steps taken
            collision_penalties: Penalties from collisions
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Store in memory
        self.training_data.append(
            {
                "episode": episode,
                "total_reward": total_reward,
                "steps": steps,
                "avg_reward_last_100": avg_reward_last_100,
                "success": success,
                "goal_rewards": goal_rewards,
                "step_penalties": step_penalties,
                "collision_penalties": collision_penalties,
                "timestamp": timestamp,
            }
        )

        # Append to CSV
        with open(self.train_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    episode,
                    total_reward,
                    steps,
                    avg_reward_last_100,
                    int(success),
                    goal_rewards,
                    step_penalties,
                    collision_penalties,
                    timestamp,
                ]
            )

    def log_evaluation_episode(self, episode, total_reward, steps, success):
        """
        Log metrics for an evaluation episode

        Args:
            episode: Episode number
            total_reward: Total reward obtained
            steps: Number of steps taken
            success: Whether the episode was successful
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Store in memory
        self.evaluation_data.append(
            {
                "episode": episode,
                "total_reward": total_reward,
                "steps": steps,
                "success": success,
                "timestamp": timestamp,
            }
        )

        # Append to CSV
        with open(self.eval_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([episode, total_reward, steps, success, timestamp])

    def get_experiment_dir(self):
        """Get the experiment directory path"""
        return str(self.experiment_dir)
