"""Modular reward system with separate reward terms"""

import sys
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Union

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import base_config


class RewTerm(ABC):
    """Abstract base class for individual reward terms"""

    @abstractmethod
    def calculate(self, state, action, next_state, goal_state, **kwargs) -> float:
        """Calculate reward for this term"""
        pass


class GoalReachedReward(RewTerm):
    """Reward for reaching the goal"""

    def __init__(self, value: float = base_config.REWARD_GOAL):
        self.value = value

    def calculate(self, state, action, next_state, goal_state, **kwargs) -> float:
        if next_state[0] == goal_state[0] and next_state[1] == goal_state[1]:
            return self.value
        return 0.0


class StepPenaltyReward(RewTerm):
    """Penalty for each step taken"""

    def __init__(self, value: float = base_config.REWARD_STEP):
        self.value = value

    def calculate(self, state, action, next_state, goal_state, **kwargs) -> float:
        if state[0] != next_state[0] or state[1] != next_state[1]:
            return self.value
        return 0.0


class CollisionPenaltyReward(RewTerm):
    """Penalty for hitting walls or obstacles"""

    def __init__(self, value: float = base_config.REWARD_COLLISION):
        self.value = value

    def calculate(self, state, action, next_state, goal_state, **kwargs) -> float:
        if state[0] == next_state[0] and state[1] == next_state[1]:
            return self.value
        return 0.0


class RewardManager:
    """Manages multiple reward terms and calculates total reward"""

    def __init__(self, reward_terms: List[RewTerm], curriculum_manager=None):
        if not reward_terms:
            raise ValueError(
                "No reward terms provided. Please specify rewards when creating the environment."
            )
        self.base_reward_terms = reward_terms
        self.curriculum_manager = curriculum_manager
        self.step_count = 0
        self.episode_count = 0

    def update_context(self, step_count: int = None, episode_count: int = None):
        """Update step and episode counts for curriculum learning"""
        if step_count is not None:
            self.step_count = step_count
        if episode_count is not None:
            self.episode_count = episode_count

    def get_current_reward_terms(self) -> List[RewTerm]:
        """Get current reward terms after applying curriculum learning"""
        if self.curriculum_manager:
            return self.curriculum_manager.apply_curriculum(
                self.base_reward_terms, self.step_count, self.episode_count
            )
        return self.base_reward_terms

    def __call__(self, state, action, next_state, goal_state, **kwargs) -> float:
        """Calculate total reward by summing all reward terms"""
        current_terms = self.get_current_reward_terms()
        total_reward = 0.0

        for term in current_terms:
            total_reward += term.calculate(state, action, next_state, goal_state, **kwargs)

        return total_reward
