"""Trajectory class for storing complete episodes in RL"""

import numpy as np
from typing import List, Optional, Dict, Any
from collections import namedtuple
from datetime import datetime
import uuid


# Step tuple for individual trajectory steps
Step = namedtuple("Step", ["state", "action", "reward", "next_state", "done"])


class Trajectory:
    """
    Trajectory class for storing complete episodes in reinforcement learning.

    A trajectory represents one complete episode from start to finish,
    including all states, actions, rewards, and metadata.
    """

    def __init__(
        self,
        episode_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        environment_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize trajectory

        Args:
            episode_id: Unique identifier for this episode
            agent_id: Identifier of the agent that generated this trajectory
            environment_id: Identifier of the environment used
            metadata: Additional metadata about the trajectory
        """
        self.episode_id = episode_id or str(uuid.uuid4())
        self.agent_id = agent_id
        self.environment_id = environment_id
        self.metadata = metadata or {}

        # Core trajectory data
        self.states: List[int] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.next_states: List[int] = []
        self.done_flags: List[bool] = []

        # Computed properties
        self._total_reward: Optional[float] = None
        self._length: Optional[int] = None
        self._success: Optional[bool] = None
        self._timestamp = datetime.now()

        # Performance metrics
        self._efficiency: Optional[float] = None  # Steps to goal ratio
        self._exploration_rate: Optional[float] = None  # Unique states visited

    def add_step(self, state: int, action: int, reward: float, next_state: int, done: bool) -> None:
        """
        Add a step to the trajectory

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state after action
            done: Whether episode ended
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.done_flags.append(done)

        # Invalidate computed properties
        self._total_reward = None
        self._length = None
        self._success = None
        self._efficiency = None
        self._exploration_rate = None

    def get_step(self, index: int) -> Step:
        """Get a specific step from the trajectory"""
        if 0 <= index < len(self.states):
            return Step(
                state=self.states[index],
                action=self.actions[index],
                reward=self.rewards[index],
                next_state=self.next_states[index],
                done=self.done_flags[index],
            )
        raise IndexError(f"Step index {index} out of range")

    def get_all_steps(self) -> List[Step]:
        """Get all steps as a list of Step namedtuples"""
        return [
            Step(state, action, reward, next_state, done)
            for state, action, reward, next_state, done in zip(
                self.states, self.actions, self.rewards, self.next_states, self.done_flags
            )
        ]

    @property
    def length(self) -> int:
        """Get trajectory length (number of steps)"""
        if self._length is None:
            self._length = len(self.states)
        return self._length

    @property
    def total_reward(self) -> float:
        """Get total reward for the trajectory"""
        if self._total_reward is None:
            self._total_reward = sum(self.rewards)
        return self._total_reward

    @property
    def success(self) -> bool:
        """Check if trajectory was successful (reached goal)"""
        if self._success is None:
            self._success = any(self.done_flags) and self.rewards[-1] > 0
        return self._success

    @property
    def efficiency(self) -> float:
        """
        Calculate trajectory efficiency (lower is better)
        Returns ratio of actual steps to optimal steps
        """
        if self._efficiency is None:
            if self.length == 0:
                self._efficiency = 0.0
            else:
                # Simple efficiency metric: reward per step
                # Higher positive reward per step = more efficient
                self._efficiency = self.total_reward / self.length
        return self._efficiency

    @property
    def exploration_rate(self) -> float:
        """
        Calculate exploration rate (unique states visited)
        Returns ratio of unique states to total states
        """
        if self._exploration_rate is None:
            if self.length == 0:
                self._exploration_rate = 0.0
            else:
                unique_states = len(set(self.states))
                self._exploration_rate = unique_states / self.length
        return self._exploration_rate

    @property
    def timestamp(self) -> datetime:
        """Get when trajectory was created"""
        return self._timestamp

    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Get comprehensive performance metrics for this trajectory

        Returns:
            Dictionary of performance metrics
        """
        return {
            "length": self.length,
            "total_reward": self.total_reward,
            "success": self.success,
            "efficiency": self.efficiency,
            "exploration_rate": self.exploration_rate,
            "avg_reward_per_step": self.total_reward / max(1, self.length),
            "unique_states": len(set(self.states)),
            "unique_actions": len(set(self.actions)),
        }

    def get_state_action_pairs(self) -> List[tuple]:
        """Get all (state, action) pairs in the trajectory"""
        return list(zip(self.states, self.actions))

    def get_reward_sequence(self) -> List[float]:
        """Get the sequence of rewards"""
        return self.rewards.copy()

    def get_state_sequence(self) -> List[int]:
        """Get the sequence of states"""
        return self.states.copy()

    def get_action_sequence(self) -> List[int]:
        """Get the sequence of actions"""
        return self.actions.copy()

    def is_complete(self) -> bool:
        """Check if trajectory is complete (has a done=True step)"""
        return any(self.done_flags)

    def get_subtrajectory(self, start: int, end: int) -> "Trajectory":
        """
        Extract a subtrajectory from this trajectory

        Args:
            start: Starting step index (inclusive)
            end: Ending step index (exclusive)

        Returns:
            New Trajectory object with the specified steps
        """
        subtraj = Trajectory(
            episode_id=f"{self.episode_id}_sub_{start}_{end}",
            agent_id=self.agent_id,
            environment_id=self.environment_id,
            metadata=self.metadata.copy(),
        )

        for i in range(start, min(end, len(self.states))):
            subtraj.add_step(
                self.states[i],
                self.actions[i],
                self.rewards[i],
                self.next_states[i],
                self.done_flags[i],
            )

        return subtraj

    def to_dict(self) -> Dict[str, Any]:
        """Convert trajectory to dictionary for serialization"""
        return {
            "episode_id": self.episode_id,
            "agent_id": self.agent_id,
            "environment_id": self.environment_id,
            "metadata": self.metadata,
            "timestamp": self._timestamp.isoformat(),
            "states": self.states,
            "actions": self.actions,
            "rewards": self.rewards,
            "next_states": self.next_states,
            "done_flags": self.done_flags,
            "performance_metrics": self.get_performance_metrics(),
        }

    def __len__(self) -> int:
        """Return trajectory length"""
        return self.length

    def __str__(self) -> str:
        """String representation of trajectory"""
        return (
            f"Trajectory(id={self.episode_id[:8]}..., "
            f"length={self.length}, "
            f"reward={self.total_reward:.2f}, "
            f"success={self.success})"
        )

    def __repr__(self) -> str:
        """Detailed string representation"""
        return (
            f"Trajectory(episode_id='{self.episode_id}', "
            f"agent_id='{self.agent_id}', "
            f"length={self.length}, "
            f"total_reward={self.total_reward:.2f}, "
            f"success={self.success})"
        )


class TrajectoryCollection:
    """
    Collection of trajectories for analysis and storage
    """

    def __init__(self, name: Optional[str] = None):
        """
        Initialize trajectory collection

        Args:
            name: Name for this collection
        """
        self.name = name or "trajectory_collection"
        self.trajectories: List[Trajectory] = []
        self._metrics_cache: Optional[Dict[str, Any]] = None

    def add_trajectory(self, trajectory: Trajectory) -> None:
        """Add a trajectory to the collection"""
        self.trajectories.append(trajectory)
        self._metrics_cache = None  # Invalidate cache

    def get_successful_trajectories(self) -> List[Trajectory]:
        """Get all successful trajectories"""
        return [t for t in self.trajectories if t.success]

    def get_failed_trajectories(self) -> List[Trajectory]:
        """Get all failed trajectories"""
        return [t for t in self.trajectories if not t.success]

    def get_collection_metrics(self) -> Dict[str, Any]:
        """
        Get aggregate metrics for the entire collection

        Returns:
            Dictionary of collection-level metrics
        """
        if self._metrics_cache is not None:
            return self._metrics_cache

        if not self.trajectories:
            return {
                "total_trajectories": 0,
                "success_rate": 0.0,
                "avg_length": 0.0,
                "avg_reward": 0.0,
                "avg_efficiency": 0.0,
            }

        successful = self.get_successful_trajectories()
        lengths = [t.length for t in self.trajectories]
        rewards = [t.total_reward for t in self.trajectories]
        efficiencies = [t.efficiency for t in self.trajectories]

        self._metrics_cache = {
            "total_trajectories": len(self.trajectories),
            "successful_trajectories": len(successful),
            "success_rate": len(successful) / len(self.trajectories),
            "avg_length": np.mean(lengths),
            "std_length": np.std(lengths),
            "avg_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "avg_efficiency": np.mean(efficiencies),
            "std_efficiency": np.std(efficiencies),
            "min_reward": np.min(rewards),
            "max_reward": np.max(rewards),
        }

        return self._metrics_cache

    def __len__(self) -> int:
        """Return number of trajectories in collection"""
        return len(self.trajectories)

    def __str__(self) -> str:
        """String representation of collection"""
        metrics = self.get_collection_metrics()
        return (
            f"TrajectoryCollection(name='{self.name}', "
            f"trajectories={metrics['total_trajectories']}, "
            f"success_rate={metrics['success_rate']:.2f})"
        )
