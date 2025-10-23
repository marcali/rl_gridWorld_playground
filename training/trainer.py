"""Training module for RL agents"""

import numpy as np
from collections import deque
from agents import QLearningAgent, AgentProtocol
from metrics import MetricsLogger
from config import base_config, experiement_config as exp_config


class Trainer:
    """Handles training of RL agents"""

    def __init__(self, experiment_name=None):
        """
        Initialize trainer

        Args:
            experiment_name: Name for this experiment
        """
        self.experiment_name = experiment_name

    def train(
        self,
        agent: AgentProtocol,
        env,
        n_episodes=exp_config.N_EPISODES,
        epsilon_start=exp_config.EPSILON_START,
        epsilon_end=exp_config.EPSILON_END,
        epsilon_decay=exp_config.EPSILON_DECAY,
        silent=False,
    ):
        """
        Train the agent with metrics logging

        Args:
            agent: Pre-initialized agent to train
            env: Environment to train in
            n_episodes: Number of training episodes
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Epsilon decay rate per episode
            silent: If True, suppress progress output for grid search

        Returns:
            tuple: (agent, logger, success_rate) if silent, else (agent, logger)
        """
        # Validate required parameters
        if agent is None:
            raise ValueError("Agent must be provided. Please initialize an agent first.")

        if env is None:
            raise ValueError(
                "Environment must be provided. Please initialize an environment first."
            )

        # Only create logger if not in silent mode
        logger = MetricsLogger(experiment_name=self.experiment_name) if not silent else None

        epsilon = epsilon_start
        # automatically deletes oldest item when maxlen is reached, only keep last 100 items
        reward_history = deque(maxlen=100)
        success_history = deque(maxlen=100)

        # Training loop
        for episode in range(n_episodes):
            state = env.reset()
            episode_reward = 0
            steps = 0
            done = False

            # Track reward components
            goal_rewards = 0
            step_penalties = 0
            collision_penalties = 0

            while not done:
                # Select action
                action = agent.act(state, epsilon=epsilon)

                # Take step
                next_state, reward, done = env.step(action)

                # Learn from experience
                agent.learn(state, action, reward, next_state, done)

                # Track reward components
                if reward == base_config.REWARD_GOAL:  # Goal reached
                    goal_rewards += reward
                elif reward == base_config.REWARD_COLLISION:  # Collision
                    collision_penalties += reward
                elif reward == base_config.REWARD_STEP:  # Step
                    step_penalties += reward

                # Update tracking
                episode_reward += reward
                steps += 1
                state = next_state

            # Decay epsilon
            epsilon = max(epsilon_end, epsilon * epsilon_decay)

            # Check if agent reached goal
            success = env.current_state == env.goal_state

            # Update histories
            reward_history.append(episode_reward)
            success_history.append(success)
            avg_reward = np.mean(reward_history)
            success_rate = np.mean(success_history) * 100

            # Log Q-value statistics and convergence metrics
            q_stats = agent.log_q_value_stats(episode)
            convergence_metrics = agent.get_convergence_metrics()
            convergence_rate = (
                convergence_metrics["convergence_rate"] if convergence_metrics else 0.0
            )

            # Log metrics
            if logger is not None:
                logger.log_training_episode(
                    episode=episode,
                    total_reward=episode_reward,
                    steps=steps,
                    avg_reward_last_100=avg_reward,
                    success=success,
                    goal_rewards=goal_rewards,
                    step_penalties=step_penalties,
                    collision_penalties=collision_penalties,
                    q_stats=q_stats,
                    convergence_rate=convergence_rate,
                )

            # Print progress (only if not silent)
            if not silent and episode % 100 == 0:
                print(
                    f"Episode {episode:4d} | "
                    f"Reward: {episode_reward:7.2f} | "
                    f"Average reward(100 episodes): {avg_reward:7.2f} | "
                    f"Steps: {steps:4d} | "
                    f"Success: {success_rate:5.1f}% | "
                )
                # Show reward breakdown
                if episode > 0:
                    print(
                        f"              | Goal: {goal_rewards:+6.0f} | Step: {step_penalties:+6.0f} | Collision: {collision_penalties:+6.0f}"
                    )
                # Show convergence metrics
                if convergence_metrics:
                    print(
                        f"              | Mean Q: {q_stats['mean_q']:6.2f} | Convergence: {convergence_rate:5.3f} | "
                        f"Explored States: {q_stats['non_zero_states']:3d}/{q_stats['total_states']}"
                    )

        # Return values based on mode
        if silent:
            return agent, logger, success_rate
        else:
            return agent, logger
