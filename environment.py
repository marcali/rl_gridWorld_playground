"""Main environment that integrates GridWorld with MDP components"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mdp import DoneTerm, RewTerm, ObsTerm
import config


class Environment:
    """Sets environment for the agent to interact with
    Args:
        size: Size of the grid
        max_steps: Maximum number of steps per episode
        agent_pos: Starting position of the agent
        goal_pos: Goal position of the agent
        n_static_obstacles: Number of static obstacles
        n_random_obstacles: Number of random obstacles
    """

    def __init__(
        self,
        size=config.GRID_SIZE,
        max_steps=config.MAX_STEPS_PER_EPISODE,
        agent_pos=config.START_STATE,
        goal_pos=config.GOAL_STATE,
        n_static_obstacles=config.N_STATIC_OBSTACLES,
        n_random_obstacles=config.N_RANDOM_OBSTACLES,
    ):
        # Base environment
        self.size = size
        self.max_steps = max_steps
        self.goal_state = goal_pos
        self.start_state = agent_pos

        # Obstacle configuration
        self.n_static_obstacles = n_static_obstacles
        self.n_random_obstacles = n_random_obstacles

        # Initialize grid (0 = free, 1 = static obstacle, 2 = random obstacle)
        self.grid = np.zeros((self.size, self.size), dtype=int)

        # Add static obstacles only (random obstacles added in reset)
        self._add_obstacles(self.n_static_obstacles, obstacle_type=1)

        # MDP components
        self.done_term = DoneTerm()
        self.rew_term = RewTerm()
        self.obs_term = ObsTerm()

        # Episode tracking
        self.step_count = 0
        self.current_state = None

    def _add_obstacles(self, n_obstacles, obstacle_type=1):
        """Add obstacles to the grid while ensuring start position remains accessible

        Args:
            n_obstacles: Number of obstacles to add
            obstacle_type: 1 for static, 2 for random (per-episode)
        """
        obstacles_added = 0
        max_attempts = n_obstacles * 20
        attempts = 0

        while obstacles_added < n_obstacles and attempts < max_attempts:
            y = np.random.randint(0, self.size)
            x = np.random.randint(0, self.size)

            # Don't place obstacles on start, goal, or existing obstacles
            if (y, x) != self.start_state and (y, x) != self.goal_state and self.grid[y, x] == 0:
                # Temporarily place obstacle
                self.grid[y, x] = obstacle_type

                # Check if start position is still accessible (path exists to goal)
                if self._does_path_exists(self.start_state, self.goal_state):
                    obstacles_added += 1
                else:
                    # Remove obstacle if it blocks the path
                    self.grid[y, x] = 0

            attempts += 1

    def reset(self):
        """Reset environment to starting state"""

        # Reset to starting state
        self.current_state = self.start_state

        # Clear random obstacles from previous episode (type 2)
        self.grid[self.grid == 2] = 0

        # Add new random obstacles for this episode
        self._add_obstacles(self.n_random_obstacles, obstacle_type=2)

        self.step_count = 0

        # Return observation
        return self.obs_term(self.current_state, self.size)

    def step(self, action):
        """
        Execute one step in the environment

        Args:
            action: Action to take (0=up, 1=down, 2=left, 3=right)

        Returns:
            observation: Current observation
            reward: Reward for the transition
            done: Whether episode is finished
        """
        # Store previous state
        prev_state = self.current_state

        # Calculate next state based on action
        y, x = self.current_state

        if action == 0:  # up
            next_state = (y - 1, x)
        elif action == 1:  # down
            next_state = (y + 1, x)
        elif action == 2:  # left
            next_state = (y, x - 1)
        elif action == 3:  # right
            next_state = (y, x + 1)
        else:
            next_state = self.current_state

        # Check boundaries (walls) and obstacles
        if (
            next_state[0] < 0
            or next_state[0] >= self.size
            or next_state[1] < 0
            or next_state[1] >= self.size
            or self.grid[next_state[0], next_state[1]] > 0  # Check for any obstacle (1 or 2)
        ):
            # Hit a wall or obstacle, stay in place
            next_state = self.current_state

        # Calculate reward
        reward = self.rew_term(prev_state, action, next_state, self.goal_state)

        # Increment step counter
        self.step_count += 1

        # Check if done
        done = self.done_term(next_state, self.goal_state, self.max_steps, self.step_count)

        # Get observation
        observation = self.obs_term(next_state, self.size)

        # Update state
        self.current_state = next_state

        return observation, reward, done

    def render(self, mode="human"):
        """Visualize the environment using matplotlib"""

        if mode == "human":
            # Create figure and axis
            fig, ax = plt.subplots(1, 1, figsize=config.FIGURE_SIZE)

            # Draw grid
            for i in range(self.size + 1):
                ax.axhline(i, color="gray", linewidth=0.5)
                ax.axvline(i, color="gray", linewidth=0.5)

            # Draw obstacles
            for y in range(self.size):
                for x in range(self.size):
                    if self.grid[y, x] == 1:
                        # Static obstacles (dark red)
                        obstacle_rect = patches.Rectangle(
                            (x, y),
                            1,
                            1,
                            linewidth=1,
                            edgecolor="darkred",
                            facecolor="red",
                            alpha=0.7,
                        )
                        ax.add_patch(obstacle_rect)
                    elif self.grid[y, x] == 2:
                        # Random obstacles (orange)
                        obstacle_rect = patches.Rectangle(
                            (x, y),
                            1,
                            1,
                            linewidth=1,
                            edgecolor="darkorange",
                            facecolor="orange",
                            alpha=0.7,
                        )
                        ax.add_patch(obstacle_rect)

            # Draw goal (green square)
            goal_rect = patches.Rectangle(
                (self.goal_state[1], self.goal_state[0]),
                1,
                1,
                linewidth=2,
                edgecolor="green",
                facecolor="lightgreen",
                alpha=0.7,
            )
            ax.add_patch(goal_rect)
            ax.text(
                self.goal_state[1] + 0.5,
                self.goal_state[0] + 0.5,
                "G",
                ha="center",
                va="center",
                fontsize=20,
                fontweight="bold",
                color="darkgreen",
            )

            # Draw agent (blue circle)
            agent_circle = patches.Circle(
                (self.current_state[1] + 0.5, self.current_state[0] + 0.5),
                0.3,
                linewidth=2,
                edgecolor="blue",
                facecolor="lightblue",
                alpha=0.9,
            )
            ax.add_patch(agent_circle)
            ax.text(
                self.current_state[1] + 0.5,
                self.current_state[0] + 0.5,
                "A",
                ha="center",
                va="center",
                fontsize=16,
                fontweight="bold",
                color="darkblue",
            )

            # Set axis properties
            ax.set_xlim(0, self.size)
            ax.set_ylim(0, self.size)
            ax.set_aspect("equal")
            ax.set_xlabel("X", fontsize=12)
            ax.set_ylabel("Y", fontsize=12)
            ax.set_title(
                f"GridWorld Environment (Step: {self.step_count})", fontsize=14, fontweight="bold"
            )

            ax.invert_yaxis()

            plt.tight_layout()
            plt.show()

        return fig if mode == "human" else None

    def render_path(self, path, title="Agent Path", save_path=None):
        """
        Visualize a complete path taken by the agent

        Args:
            path: List of states (tuples) representing the path
            title: Title for the plot
            save_path: If provided, saves figure to this path instead of showing

        Returns:
            matplotlib figure
        """
        fig, ax = plt.subplots(1, 1, figsize=config.PATH_FIGURE_SIZE)

        # Draw grid
        for i in range(self.size + 1):
            ax.axhline(i, color="gray", linewidth=0.5)
            ax.axvline(i, color="gray", linewidth=0.5)

        # Draw obstacles
        for y in range(self.size):
            for x in range(self.size):
                if self.grid[y, x] == 1:
                    # Static obstacles (dark red)
                    obstacle_rect = patches.Rectangle(
                        (x, y), 1, 1, linewidth=1, edgecolor="darkred", facecolor="red", alpha=0.6
                    )
                    ax.add_patch(obstacle_rect)
                elif self.grid[y, x] == 2:
                    # Random obstacles (orange)
                    obstacle_rect = patches.Rectangle(
                        (x, y),
                        1,
                        1,
                        linewidth=1,
                        edgecolor="darkorange",
                        facecolor="orange",
                        alpha=0.6,
                    )
                    ax.add_patch(obstacle_rect)

        # Draw path as line
        if len(path) > 1:
            path_x = [state[1] + 0.5 for state in path]
            path_y = [state[0] + 0.5 for state in path]
            ax.plot(path_x, path_y, "b-", linewidth=3, alpha=0.6, label=f"Path ({len(path)} steps)")

            # Add arrows to show direction
            for i in range(len(path) - 1):
                dx = path_x[i + 1] - path_x[i]
                dy = path_y[i + 1] - path_y[i]
                if dx != 0 or dy != 0:  # Only draw if moved
                    ax.arrow(
                        path_x[i],
                        path_y[i],
                        dx * 0.7,
                        dy * 0.7,
                        head_width=0.15,
                        head_length=0.1,
                        fc="blue",
                        ec="blue",
                        alpha=0.5,
                    )

            # Mark start with circle
            ax.plot(path_x[0], path_y[0], "go", markersize=15, label="Start", zorder=10)

            # Mark end position
            if path[-1] == self.goal_state:
                ax.plot(
                    path_x[-1], path_y[-1], "g*", markersize=25, label="Goal Reached!", zorder=10
                )
            else:
                ax.plot(path_x[-1], path_y[-1], "rx", markersize=15, label="Stopped", zorder=10)

        # Draw goal (green square)
        goal_rect = patches.Rectangle(
            (self.goal_state[1], self.goal_state[0]),
            1,
            1,
            linewidth=3,
            edgecolor="green",
            facecolor="lightgreen",
            alpha=0.5,
        )
        ax.add_patch(goal_rect)
        ax.text(
            self.goal_state[1] + 0.5,
            self.goal_state[0] + 0.5,
            "G",
            ha="center",
            va="center",
            fontsize=20,
            fontweight="bold",
            color="darkgreen",
        )

        # Set axis properties
        ax.set_xlim(0, self.size)
        ax.set_ylim(0, self.size)
        ax.set_aspect("equal")
        ax.set_xlabel("X", fontsize=12)
        ax.set_ylabel("Y", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(loc="upper left")
        ax.invert_yaxis()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=config.DPI, bbox_inches="tight")
            print(f"✓ Path visualization saved to: {save_path}")
            plt.close(fig)
        else:
            plt.show()

        return fig

    def _does_path_exists(self, start, goal):
        """Check if a path exists from start to goal using BFS

        Args:
            start: Starting position (y, x)
            goal: Goal position (y, x)

        Returns:
            bool: True if path exists, False otherwise
        """
        from collections import deque

        if start == goal:
            return True

        # BFS to find path
        queue = deque([start])
        visited = {start}
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # right, left, down, up

        while queue:
            current = queue.popleft()

            for dy, dx in directions:
                next_y, next_x = current[0] + dy, current[1] + dx
                next_pos = (next_y, next_x)

                # Check bounds
                if next_y < 0 or next_y >= self.size or next_x < 0 or next_x >= self.size:
                    continue

                # Check if it's an obstacle
                if self.grid[next_y, next_x] > 0:
                    continue

                # Check if we've reached the goal
                if next_pos == goal:
                    return True

                # Add to queue if not visited
                if next_pos not in visited:
                    visited.add(next_pos)
                    queue.append(next_pos)

        return False
