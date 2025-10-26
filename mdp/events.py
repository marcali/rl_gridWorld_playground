"""Event system for dynamic environment changes"""

import numpy as np
import random
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import base_config


class EventTerm(ABC):
    """Abstract base class for event terms"""

    @abstractmethod
    def apply(
        self, grid: np.ndarray, state: Tuple[int, int], goal_state: Tuple[int, int]
    ) -> np.ndarray:
        """
        Apply event to the grid

        Args:
            grid: Current grid state
            state: Current agent state
            goal_state: Goal state

        Returns:
            Modified grid
        """
        pass


class RandomObstacleEvent(EventTerm):
    """Event that adds random obstacles to the grid"""

    def __init__(self, n_obstacles: int = base_config.N_RANDOM_OBSTACLES, obstacle_type: int = 2):
        """
        Initialize random obstacle event

        Args:
            n_obstacles: Number of random obstacles to add
            obstacle_type: Type of obstacle (1=static, 2=random)
        """
        self.n_obstacles = n_obstacles
        self.obstacle_type = obstacle_type

    def apply(
        self, grid: np.ndarray, state: Tuple[int, int], goal_state: Tuple[int, int]
    ) -> np.ndarray:
        """Add random obstacles to the grid"""
        grid_copy = grid.copy()
        grid_size = grid_copy.shape[0]

        # Clear existing random obstacles (type 2)
        grid_copy[grid_copy == 2] = 0

        # Add new random obstacles
        obstacles_added = 0
        max_attempts = self.n_obstacles * 20  # Prevent infinite loops
        attempts = 0

        while obstacles_added < self.n_obstacles and attempts < max_attempts:
            attempts += 1

            # Generate random position
            y = np.random.randint(0, grid_size)
            x = np.random.randint(0, grid_size)

            # Check if position is valid (not start, goal, or existing obstacle)
            if (y, x) != state and (y, x) != goal_state and grid_copy[y, x] == 0:
                # Temporarily place obstacle
                grid_copy[y, x] = self.obstacle_type

                # Check if start position is still accessible (path exists to goal)
                if self._does_path_exists(grid_copy, state, goal_state):
                    obstacles_added += 1
                else:
                    # Remove obstacle if it blocks the path
                    grid_copy[y, x] = 0

        return grid_copy

    def _does_path_exists(
        self, grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]
    ) -> bool:
        """Check if a path exists from start to goal using BFS"""
        from collections import deque

        if start == goal:
            return True

        grid_size = grid.shape[0]
        visited = set()
        queue = deque([start])
        visited.add(start)

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right

        while queue:
            current = queue.popleft()

            if current == goal:
                return True

            for dy, dx in directions:
                next_y, next_x = current[0] + dy, current[1] + dx

                if (
                    0 <= next_y < grid_size
                    and 0 <= next_x < grid_size
                    and (next_y, next_x) not in visited
                    and grid[next_y, next_x] == 0
                ):
                    visited.add((next_y, next_x))
                    queue.append((next_y, next_x))

        return False


class EventManager:
    """Manages multiple event terms and applies them to the environment"""

    def __init__(self, event_terms: Optional[List[EventTerm]] = None):
        """
        Initialize event manager

        Args:
            event_terms: List of event terms to manage
        """
        self.event_terms = event_terms or []

    def add_event(self, event_term: EventTerm):
        """Add an event term to the manager"""
        self.event_terms.append(event_term)

    def apply_events(
        self, grid: np.ndarray, state: Tuple[int, int], goal_state: Tuple[int, int]
    ) -> np.ndarray:
        """
        Apply all event terms to the grid

        Args:
            grid: Current grid state
            state: Current agent state
            goal_state: Goal state

        Returns:
            Modified grid after applying all events
        """
        current_grid = grid.copy()

        for event_term in self.event_terms:
            current_grid = event_term.apply(current_grid, state, goal_state)

        return current_grid

    def __call__(
        self, grid: np.ndarray, state: Tuple[int, int], goal_state: Tuple[int, int]
    ) -> np.ndarray:
        """Make EventManager callable"""
        return self.apply_events(grid, state, goal_state)
