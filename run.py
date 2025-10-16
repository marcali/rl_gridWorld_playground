"""Training script - Main entry point to put it all together"""

import numpy as np
from collections import deque
from environment import Environment
from agents import QLearningAgent, RandomAgent, AgentProtocol
from metrics import MetricsLogger
from metrics.visualization import create_all_visualizations
import config


def train(
    n_episodes=config.N_EPISODES,
    epsilon_start=config.EPSILON_START,
    epsilon_end=config.EPSILON_END,
    epsilon_decay=config.EPSILON_DECAY,
    alpha=config.ALPHA,
    gamma=config.GAMMA,
    experiment_name=None,
    silent=False,
    env=None,
):
    """
    Train the agent with metrics logging

    Args:
        n_episodes: Number of training episodes
        epsilon_start: Initial exploration rate
        epsilon_end: Minimum exploration rate
        epsilon_decay: Epsilon decay rate per episode
        alpha: Learning rate
        gamma: Discount factor
        experiment_name: Name for this experiment
        silent: If True, suppress progress output for grid search
        env: Environment to use (if None, creates new one)
    """

    # instanciate learning agent
    agent = QLearningAgent(
        n_states=config.N_STATES, n_actions=config.N_ACTIONS, alpha=alpha, gamma=gamma
    )

    # Only create logger if not in silent mode
    logger = MetricsLogger(experiment_name=experiment_name) if not silent else None

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
            if reward == config.REWARD_GOAL:  # Goal reached
                goal_rewards += reward
            elif reward == config.REWARD_COLLISION:  # Collision
                collision_penalties += reward
            elif reward == config.REWARD_STEP:  # Step
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

    # Return values based on mode
    if silent:
        return agent, logger, success_rate
    else:
        return agent, logger


def evaluate(
    agent: AgentProtocol,
    logger,
    env,
    n_episodes=config.N_EVAL_EPISODES,
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

        while not done and steps < config.MAX_EVAL_STEPS:
            # Use custom eval_epsilon if provided, otherwise use config default
            epsilon_to_use = eval_epsilon if eval_epsilon is not None else config.EVAL_EPSILON
            # Use custom tolerance if provided, otherwise use config default
            tolerance_to_use = tolerance if tolerance is not None else config.Q_VALUE_TOLERANCE

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
            save_path = logger.experiment_dir / f"{agent_type.lower()}_path_episode_{episode+1}.png"
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
    agent: AgentProtocol,
    logger,
    env,
    n_episodes=config.N_EVAL_EPISODES,
    eval_epsilon=None,
    tolerance=None,
):
    """
    Evaluate trained agent and compare against random baseline

    Args:
        agent:Previously trained agent
        logger: MetricsLogger instance
        env: Environment (same as training)
        n_episodes: Number of evaluation episodes
        eval_epsilon: Custom evaluation epsilon
        tolerance: Custom tolerance for tie-breaking

    Returns:
        tuple: (logger, trained_results, random_results)
    """
    # Evaluate trained agent
    print("\n" + "=" * 60)
    print("TRAINED AGENT EVALUATION")
    print("=" * 60)

    _, trained_results = evaluate(
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

    random_agent = RandomAgent(n_states=config.N_STATES, n_actions=config.N_ACTIONS)
    _, random_results = evaluate(
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


if __name__ == "__main__":

    env = Environment()
    # Train agent (uses defaults from config.py)
    trained_agent, metrics_logger = train(env=env)

    # Evaluate agent with random baseline comparison (uses defaults from config.py)
    metrics_logger, trained_results, random_results = evaluate_with_random_baseline(
        trained_agent, metrics_logger, env
    )

    # Generate visualizations
    print("\nGenerating visualizations...")
    create_all_visualizations(metrics_logger.get_experiment_dir(), trained_results, random_results)

    print("\n" + "=" * 60)
    print(" TRAINING AND EVALUATION COMPLETE!")
    print("=" * 60)
