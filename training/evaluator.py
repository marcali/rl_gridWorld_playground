"""Evaluation module for RL agents"""

import numpy as np
from agents import RandomAgent, AgentProtocol
from config import base_config, experiement_config as exp_config


class Evaluator:
    """Handles evaluation of RL agents"""

    def __init__(self):
        """Initialize evaluator"""
        pass

    def evaluate(
        self,
        agent: AgentProtocol,
        logger,
        env,
        n_episodes=base_config.N_EVAL_EPISODES,
        silent=False,
        eval_epsilon=None,
        tolerance=None,
        agent_type=None,
        save_paths=None,
    ):
        """
        Evaluates agent with metrics logging

        Args:
            agent: Agent to evaluate (QLearningAgent, RandomAgent, etc.)
            logger: MetricsLogger instance (can be None for silent mode)
            env: Environment (same as training)
            n_episodes: Number of evaluation episodes
            silent: If True, suppress output and doesn't save visualizations
            eval_epsilon: Custom evaluation epsilon
            tolerance: Custom tolerance for tie-breaking
            agent_type: String to identify agent type
            save_paths: Whether to save path visualizations (auto-determined if None)

        Returns:
            Dictionary with evaluation metrics if silent=True, otherwise returns logger
        """
        # Validate required parameters
        if agent is None:
            raise ValueError("Agent must be provided. Please initialize an agent first.")

        if env is None:
            raise ValueError(
                "Environment must be provided. Please initialize an environment first."
            )
        # Auto-determine save_paths if not specified
        if save_paths is None:
            save_paths = not silent

        total_rewards = []
        total_steps = []
        successes = []

        # Evaluation loop
        for episode in range(n_episodes):
            state = env.reset()
            episode_reward = 0
            steps = 0
            done = False

            # Track path for visualization
            path = [env.current_state]

            while not done and steps < base_config.MAX_EVAL_STEPS:
                # Use custom eval_epsilon if provided, otherwise use config default
                epsilon_to_use = (
                    eval_epsilon if eval_epsilon is not None else base_config.EVAL_EPSILON
                )
                # Use custom tolerance if provided, otherwise use config default
                tolerance_to_use = (
                    tolerance if tolerance is not None else exp_config.Q_VALUE_TOLERANCE
                )

                # Select action
                action = agent.act(state, epsilon=epsilon_to_use, tolerance=tolerance_to_use)

                # Take step
                next_state, reward, done = env.step(action)

                # Update tracking
                episode_reward += reward
                steps += 1
                state = next_state

                # Record path
                path.append(env.current_state)

            # Determine success (did agent reach goal)
            success = env.current_state == env.goal_state

            # Save path visualization and log metrics
            if save_paths and logger is not None:
                # files are named after agent_type
                save_path = (
                    logger.experiment_dir / f"{agent_type.lower()}_path_episode_{episode+1}.png"
                )
                title = f"{agent_type} Agent - Episode {episode+1} - {'SUCCESS' if success else 'FAILED'} ({steps} steps)"
                env.render_path(path, title=title, save_path=save_path)

                # Log evaluation episode
                logger.log_evaluation_episode(
                    episode=episode,
                    total_reward=episode_reward,
                    steps=steps,
                    success=success,
                )

            if not silent:
                status = "goal reached" if success else "Fail"
                print(
                    f"{agent_type} Episode {episode + 1:2d} | "
                    f"Reward: {episode_reward:7.2f} | "
                    f"Steps: {steps:4d} | "
                    f"{status}"
                )

            # Store results
            total_rewards.append(episode_reward)
            total_steps.append(steps)
            successes.append(success)

        # Calculate summary statistics
        success_rate = np.mean(successes) * 100
        avg_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)
        avg_steps = np.mean(total_steps)
        std_steps = np.std(total_steps)

        results = {
            "success_rate": success_rate,
            "avg_reward": avg_reward,
            "std_reward": std_reward,
            "avg_steps": avg_steps,
            "std_steps": std_steps,
        }

        if silent:
            return results
        else:
            # Return both logger and results for non-silent mode
            return logger, results

    def evaluate_with_random_baseline(
        self,
        agent: AgentProtocol,
        logger,
        env,
        n_episodes=base_config.N_EVAL_EPISODES,
        eval_epsilon=None,
        tolerance=None,
    ):
        """
        Evaluate trained agent and compare against random baseline

        Args:
            agent: Previously trained agent
            logger: MetricsLogger instance
            env: Environment (same as training)
            n_episodes: Number of evaluation episodes
            eval_epsilon: Custom evaluation epsilon
            tolerance: Custom tolerance for tie-breaking

        Returns:
            tuple: (logger, trained_results, random_results)
        """
        # Validate required parameters
        if agent is None:
            raise ValueError("Agent must be provided. Please initialize an agent first.")

        if env is None:
            raise ValueError(
                "Environment must be provided. Please initialize an environment first."
            )
        # Evaluate trained agent
        print("\n" + "=" * 60)
        print("TRAINED AGENT EVALUATION")
        print("=" * 60)

        _, trained_results = self.evaluate(
            agent=agent,
            logger=logger,
            env=env,
            n_episodes=n_episodes,
            silent=False,
            eval_epsilon=eval_epsilon,
            tolerance=tolerance,
            agent_type="Trained",
            save_paths=True,
        )

        # Evaluate random agent for comparison
        print("\n" + "=" * 60)
        print("RANDOM AGENT EVALUATION")
        print("=" * 60)

        random_agent = RandomAgent(n_states=base_config.N_STATES, n_actions=base_config.N_ACTIONS)
        _, random_results = self.evaluate(
            agent=random_agent,
            logger=None,
            env=env,
            n_episodes=n_episodes,
            silent=False,
            eval_epsilon=None,
            tolerance=None,
            agent_type="Random",
            save_paths=False,
        )

        return logger, trained_results, random_results
