"""Hindsight Experience Replay (HER) implementation"""

import numpy as np
import random
from typing import List, Tuple, Callable, Optional, Any
from collections import deque
from .replay_buffer import Experience, ReplayBuffer


class HindsightExperienceReplay:
    """
    Hindsight Experience Replay (HER) wrapper for goal-conditioned RL

    HER generates additional training experiences by treating achieved goals
    as if they were the intended goals, improving sample efficiency for
    sparse reward environments.

    Args:
        replay_buffer: The underlying replay buffer to use
        k: Number of hindsight goals to sample per episode
        strategy: Strategy for sampling hindsight goals ('future', 'final', 'random')
        goal_extraction_func: Function to extract achieved goal from state
        desired_goal_extraction_func: Function to extract desired goal from state
        goal_substitution_func: Function to substitute goal in state
    """

    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        k: int = 4,
        strategy: str = "future",
        goal_extraction_func: Optional[Callable] = None,
        desired_goal_extraction_func: Optional[Callable] = None,
        goal_substitution_func: Optional[Callable] = None,
        seed: Optional[int] = None,
    ):
        self.replay_buffer = replay_buffer
        self.k = k
        self.strategy = strategy
        self.seed = seed

        # Episode storage for HER processing
        self.current_episode = []
        self.episode_goals = []

        # Goal extraction functions (to be set by user)
        self._extract_achieved_goal = goal_extraction_func
        self._extract_desired_goal = desired_goal_extraction_func
        self._substitute_goal = goal_substitution_func

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def add_experience(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any,
        done: bool,
        info: Optional[dict] = None,
    ) -> None:
        """
        Add experience to replay buffer and episode storage

        Args:
            state: Current state (flattened index)
            action: Action taken
            reward: Reward received
            next_state: Next state after action (flattened index)
            done: Whether episode is finished
            info: Additional info dict (should contain 'goal_state' for HER)
        """
        # Add original experience to replay buffer
        self.replay_buffer.add(state, action, reward, next_state, done)

        # Store in current episode for HER processing
        experience = {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done,
            "info": info or {},
        }
        self.current_episode.append(experience)

        # If episode is done, process with HER
        if done:
            self._process_episode_with_her()
            self.current_episode = []
            self.episode_goals = []

    def _process_episode_with_her(self) -> None:
        """Process completed episode with HER to generate additional experiences"""
        if len(self.current_episode) == 0:
            return

        # Extract achieved goals from episode (current positions)
        achieved_goals = []
        for exp in self.current_episode:
            achieved_goal = self._extract_achieved_goal(exp["next_state"])
            achieved_goals.append(achieved_goal)

        # Get the original goal from the first experience's info
        original_goal = None
        if self.current_episode and "goal_state" in self.current_episode[0]["info"]:
            original_goal = self.current_episode[0]["info"]["goal_state"]

        if original_goal is None:
            # If no goal information available, skip HER processing
            return

        # Sample hindsight goals
        hindsight_goals = self._sample_hindsight_goals(achieved_goals)

        # Generate HER experiences
        for hindsight_goal in hindsight_goals:
            self._generate_her_experiences(hindsight_goal, original_goal)

    def _sample_hindsight_goals(self, achieved_goals: List[Any]) -> List[Any]:
        """
        Sample hindsight goals based on strategy

        Args:
            achieved_goals: List of achieved goals from the episode

        Returns:
            List of hindsight goals to use for HER
        """
        if self.strategy == "future":
            return self._sample_future_goals(achieved_goals)
        elif self.strategy == "final":
            return self._sample_final_goals(achieved_goals)
        elif self.strategy == "random":
            return self._sample_random_goals(achieved_goals)
        else:
            raise ValueError(f"Unknown HER strategy: {self.strategy}")

    def _sample_future_goals(self, achieved_goals: List[Any]) -> List[Any]:
        """Sample future achieved goals as hindsight goals"""
        hindsight_goals = []

        for i in range(len(achieved_goals)):
            # Sample k goals from future timesteps
            future_goals = achieved_goals[i + 1 :]
            if len(future_goals) > 0:
                k_samples = min(self.k, len(future_goals))
                sampled = random.sample(future_goals, k_samples)
                hindsight_goals.extend(sampled)

        return hindsight_goals

    def _sample_final_goals(self, achieved_goals: List[Any]) -> List[Any]:
        """Sample final achieved goal as hindsight goal"""
        if len(achieved_goals) == 0:
            return []

        final_goal = achieved_goals[-1]
        return [final_goal] * self.k

    def _sample_random_goals(self, achieved_goals: List[Any]) -> List[Any]:
        """Sample random achieved goals as hindsight goals"""
        if len(achieved_goals) == 0:
            return []

        k_samples = min(self.k, len(achieved_goals))
        return random.sample(achieved_goals, k_samples)

    def _generate_her_experiences(self, hindsight_goal: Any, original_goal: Any) -> None:
        """
        Generate HER experiences with substituted hindsight goal

        Args:
            hindsight_goal: The hindsight goal to substitute
            original_goal: The original goal from the episode
        """
        for exp in self.current_episode:
            # Create modified state and next_state with hindsight goal
            modified_state = self._substitute_goal(exp["state"], hindsight_goal)
            modified_next_state = self._substitute_goal(exp["next_state"], hindsight_goal)

            # Calculate new reward based on hindsight goal
            new_reward = self._calculate_her_reward(exp["next_state"], hindsight_goal, exp["done"])

            # Add HER experience to replay buffer
            self.replay_buffer.add(
                modified_state, exp["action"], new_reward, modified_next_state, exp["done"]
            )

    def _calculate_her_reward(self, next_state: Any, goal: Any, done: bool) -> float:
        """
        Calculate reward for HER experience

        Args:
            next_state: The next state in the original experience
            goal: The hindsight goal
            done: Whether the original episode was done

        Returns:
            Reward for the HER experience
        """
        # Extract achieved goal from next_state
        achieved_goal = self._extract_achieved_goal(next_state)

        # Check if hindsight goal was achieved
        if self._goals_equal(achieved_goal, goal):
            return 1.0  # Success reward
        else:
            return 0.0  # No reward

    def _goals_equal(self, goal1: Any, goal2: Any) -> bool:
        """
        Check if two goals are equal

        Args:
            goal1: First goal
            goal2: Second goal

        Returns:
            True if goals are equal, False otherwise
        """
        if isinstance(goal1, np.ndarray) and isinstance(goal2, np.ndarray):
            return np.array_equal(goal1, goal2)
        elif isinstance(goal1, tuple) and isinstance(goal2, tuple):
            return goal1 == goal2
        else:
            return goal1 == goal2

    def sample(self, batch_size: int) -> List[Experience]:
        """Sample batch from underlying replay buffer"""
        return self.replay_buffer.sample(batch_size)

    def __len__(self) -> int:
        """Return size of underlying replay buffer"""
        return len(self.replay_buffer)

    def __getattr__(self, name):
        """Delegate other methods to underlying replay buffer"""
        return getattr(self.replay_buffer, name)
