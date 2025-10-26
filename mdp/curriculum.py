"""Curriculum learning system for modifying rewards over time"""

import sys
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Union, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import base_config


class CurriculumRule(ABC):
    """Abstract base class for curriculum learning rules"""

    @abstractmethod
    def should_apply(self, step_count: int, episode_count: int) -> bool:
        """Check if this rule should be applied at the current step/episode"""
        pass

    @abstractmethod
    def apply(self, reward_terms: List, step_count: int, episode_count: int) -> List:
        """Apply the curriculum rule to modify reward terms"""
        pass


class StepBasedCurriculum(CurriculumRule):
    """Curriculum rule that applies after a certain number of steps"""

    def __init__(self, step_threshold: int, action: str, **kwargs):
        """
        Args:
            step_threshold: Number of steps after which to apply
            action: What action to take ('modify_existing', 'add_new', 'remove_existing')
            **kwargs: Additional parameters for the action
        """
        self.step_threshold = step_threshold
        self.action = action
        self.kwargs = kwargs

    def should_apply(self, step_count: int, episode_count: int) -> bool:
        return step_count >= self.step_threshold

    def apply(self, reward_terms: List, step_count: int, episode_count: int) -> List:
        if self.action == "modify_existing":
            return self._modify_existing_rewards(reward_terms)
        elif self.action == "add_new":
            return self._add_new_rewards(reward_terms)
        elif self.action == "remove_existing":
            return self._remove_existing_rewards(reward_terms)
        else:
            return reward_terms

    def _modify_existing_rewards(self, reward_terms: List) -> List:
        """Modify existing reward values"""
        modified_terms = []
        for term in reward_terms:
            # Create a new instance with modified value
            if hasattr(term, "value"):
                new_value = term.value * self.kwargs.get("multiplier", 1.0)
                new_term = term.__class__(new_value)
                modified_terms.append(new_term)
            else:
                modified_terms.append(term)
        return modified_terms

    def _add_new_rewards(self, reward_terms: List) -> List:
        """Add new reward terms"""
        new_terms = reward_terms.copy()
        for term_class, value in self.kwargs.get("new_terms", []):
            new_terms.append(term_class(value))
        return new_terms

    def _remove_existing_rewards(self, reward_terms: List) -> List:
        """Remove existing reward terms by class name"""
        removed_classes = self.kwargs.get("remove_classes", [])
        return [term for term in reward_terms if term.__class__.__name__ not in removed_classes]


class EpisodeBasedCurriculum(CurriculumRule):
    """Curriculum rule that applies after a certain number of episodes"""

    def __init__(self, episode_threshold: int, action: str, **kwargs):
        self.episode_threshold = episode_threshold
        self.action = action
        self.kwargs = kwargs

    def should_apply(self, step_count: int, episode_count: int) -> bool:
        return episode_count >= self.episode_threshold

    def apply(self, reward_terms: List, step_count: int, episode_count: int) -> List:
        if self.action == "modify_existing":
            return self._modify_existing_rewards(reward_terms)
        elif self.action == "add_new":
            return self._add_new_rewards(reward_terms)
        elif self.action == "remove_existing":
            return self._remove_existing_rewards(reward_terms)
        else:
            return reward_terms

    def _modify_existing_rewards(self, reward_terms: List) -> List:
        modified_terms = []
        for term in reward_terms:
            if hasattr(term, "value"):
                new_value = term.value * self.kwargs.get("multiplier", 1.0)
                new_term = term.__class__(new_value)
                modified_terms.append(new_term)
            else:
                modified_terms.append(term)
        return modified_terms

    def _add_new_rewards(self, reward_terms: List) -> List:
        new_terms = reward_terms.copy()
        for term_class, value in self.kwargs.get("new_terms", []):
            new_terms.append(term_class(value))
        return new_terms

    def _remove_existing_rewards(self, reward_terms: List) -> List:
        removed_classes = self.kwargs.get("remove_classes", [])
        return [term for term in reward_terms if term.__class__.__name__ not in removed_classes]


class CurriculumManager:
    """Manages curriculum learning rules and applies them to rewards"""

    def __init__(self, curriculum_rules: Optional[List[CurriculumRule]] = None):
        self.curriculum_rules = curriculum_rules or []
        self.applied_rules = set()  # Track which rules have been applied

    def add_rule(self, rule: CurriculumRule):
        """Add a curriculum rule"""
        self.curriculum_rules.append(rule)

    def apply_curriculum(self, reward_terms: List, step_count: int, episode_count: int) -> List:
        """Apply curriculum rules to modify reward terms"""
        current_terms = reward_terms.copy()

        for i, rule in enumerate(self.curriculum_rules):
            rule_id = f"{rule.__class__.__name__}_{i}"

            if rule_id not in self.applied_rules and rule.should_apply(step_count, episode_count):
                current_terms = rule.apply(current_terms, step_count, episode_count)
                self.applied_rules.add(rule_id)

        return current_terms

    def get_applied_rules(self) -> List[str]:
        """Get list of applied rule IDs"""
        return list(self.applied_rules)

    def reset(self):
        """Reset applied rules (useful for new training runs)"""
        self.applied_rules.clear()
