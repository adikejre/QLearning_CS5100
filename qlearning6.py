"""
CS5100 FAI Capstone: Empirical Validation of Q-Learning Convergence
Based on: Watkins & Dayan (1992) "Q-Learning"

This file contains:
1. Custom GridWorld environment
2. Value Iteration (computes exact Q*)
3. Tabular Q-learning
4. SARSA
5. Experiments validating the paper's convergence theorem
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import gymnasium as gym

# =============================================================================
# 1. CUSTOM GRIDWORLD ENVIRONMENT
# =============================================================================

class GridWorld:
    """
    Simple NxN grid. Agent starts top-left, goal is bottom-right.
    Actions: 0=up, 1=right, 2=down, 3=left
    Reward: +1 at goal, -0.01 per step (encourages shortest path).
    Deterministic transitions. Walls bounce the agent back.
    """
    def __init__(self, size=4):
        self.size = size
        self.n_states = size * size
        self.n_actions = 4
        self.goal = self.n_states - 1  # bottom-right corner
        self.state = 0

    def reset(self):
        self.state = 0
        return self.state

    def step(self, action):
        if self.state == self.goal:
            return self.state, 0.0, True

        row, col = divmod(self.state, self.size)

        if action == 0:    # up
            row = max(row - 1, 0)
        elif action == 1:  # right
            col = min(col + 1, self.size - 1)
        elif action == 2:  # down
            row = min(row + 1, self.size - 1)
        elif action == 3:  # left
            col = max(col - 1, 0)

        self.state = row * self.size + col

        if self.state == self.goal:
            return self.state, 1.0, True
        else:
            return self.state, -0.1, False

    def get_transition_model(self):
        """
        Returns P[s][a] = list of (prob, next_state, reward, done)
        Needed by value iteration to compute exact Q*.
        """
        P = {}
        for s in range(self.n_states):
            P[s] = {}
            if s == self.goal:
                # Terminal state — no meaningful transitions
                for a in range(self.n_actions):
                    P[s][a] = [(1.0, s, 0.0, True)]
                continue
            for a in range(self.n_actions):
                row, col = divmod(s, self.size)
                if a == 0:   row = max(row - 1, 0)
                elif a == 1: col = min(col + 1, self.size - 1)
                elif a == 2: row = min(row + 1, self.size - 1)
                elif a == 3: col = max(col - 1, 0)

                ns = row * self.size + col
                if ns == self.goal:
                    P[s][a] = [(1.0, ns, 1.0, True)]
                else:
                    P[s][a] = [(1.0, ns, -0.1, False)]
        return P


# =============================================================================
# 2. VALUE ITERATION — Computes exact Q* (ground truth)
# =============================================================================

def value_iteration(P, n_states, n_actions, gamma=0.99, theta=1e-10, terminal_states=None):
    """
    Standard DP value iteration.
    P[s][a] = [(prob, next_state, reward, done), ...]
    terminal_states: set of states where the episode ends (Q = 0 there)

    Returns: Q_star[s, a] — the true optimal Q-values
    """
    if terminal_states is None:
        terminal_states = set()

    Q = np.zeros((n_states, n_actions))

    for iteration in range(10000):
        Q_new = np.zeros_like(Q)
        for s in range(n_states):
            if s in terminal_states:
                continue  # Q = 0 at terminal states
            for a in range(n_actions):
                for prob, ns, reward, done in P[s][a]:
                    if done:
                        Q_new[s, a] += prob * reward
                    else:
                        Q_new[s, a] += prob * (reward + gamma * np.max(Q[ns]))

        if np.max(np.abs(Q_new - Q)) < theta:
            print(f"Value iteration converged in {iteration+1} iterations")
            return Q_new
        Q = Q_new

    print("Value iteration did not converge")
    return Q


# =============================================================================
# 3. TABULAR Q-LEARNING
# =============================================================================

def q_learning(env, n_episodes, gamma=0.99, alpha_mode="decay",
               alpha_fixed=0.1, epsilon_mode="decay", epsilon_fixed=0.1,
               Q_star=None):
    """
    Tabular Q-learning with configurable learning rate and exploration.

    alpha_mode: "decay" uses 1/visit_count(s,a), "fixed" uses alpha_fixed
    epsilon_mode: "decay" uses 1/sqrt(episode), "fixed" uses epsilon_fixed,
                  "greedy" uses 0

    Returns:
        Q: learned Q-table
        metrics: dict with error history, reward history, policy_match history
    """
    n_states = env.n_states if hasattr(env, 'n_states') else env.observation_space.n
    n_actions = env.n_actions if hasattr(env, 'n_actions') else env.action_space.n

    Q = np.zeros((n_states, n_actions))
    visit_count = np.zeros((n_states, n_actions))

    rewards_per_episode = []
    q_errors = []          # ||Q - Q*||_inf at each episode
    mean_abs_errors = []   # mean |Q - Q*| at each episode
    policy_matches = []    # fraction of states where argmax matches pi*

    for ep in range(n_episodes):
        state = env.reset()
        if isinstance(state, tuple):  # gymnasium returns (obs, info)
            state = state[0]

        total_reward = 0
        done = False

        # Epsilon for this episode
        if epsilon_mode == "decay":
            epsilon = max(0.01, 1.0 / (1 + ep * 0.01))  # slower decay, floors at 0.01
        elif epsilon_mode == "fixed":
            epsilon = epsilon_fixed
        else:  # greedy
            epsilon = 0.0

        steps = 0
        while not done and steps < 500:
            # Epsilon-greedy action selection
            if np.random.random() < epsilon:
                action = np.random.randint(n_actions)
            else:
                action = np.argmax(Q[state])

            # Take action
            result = env.step(action)
            if len(result) == 5:  # gymnasium style
                next_state, reward, terminated, truncated, _ = result
                done = terminated or truncated
            else:  # our custom env
                next_state, reward, done = result

            # Update visit count
            visit_count[state, action] += 1

            # Learning rate
            if alpha_mode == "decay":
                # Use learning rate that decays based on visit count.
                # α = c / (c + visit_count) keeps α meaningful for longer.
                # Still satisfies theorem: Σα = ∞, Σα² < ∞ (harmonic-like)
                alpha = 10.0 / (10.0 + visit_count[state, action])
            elif alpha_mode == "decay_strict":
                # Pure 1/n — satisfies theorem but very slow in practice
                alpha = 1.0 / visit_count[state, action]
            else:
                alpha = alpha_fixed

            # Q-learning update: uses MAX over next state (off-policy)
            td_target = reward + gamma * np.max(Q[next_state]) * (1 - done)
            td_error = td_target - Q[state, action]
            Q[state, action] += alpha * td_error

            total_reward += reward
            state = next_state
            steps += 1

        rewards_per_episode.append(total_reward)

        # Track metrics if ground truth is available
        if Q_star is not None:
            # Exclude terminal/goal states from error calculation.
            non_terminal = [s for s in range(n_states)
                           if not (hasattr(env, 'goal') and s == env.goal)]
            if not non_terminal:
                non_terminal = list(range(n_states))

            # Track two error metrics:
            # 1. Max absolute error across all state-action pairs (strict)
            q_errors.append(np.max(np.abs(Q[non_terminal] - Q_star[non_terminal])))

            # 2. Mean absolute error (more representative of overall convergence)
            mean_abs_errors.append(np.mean(np.abs(Q[non_terminal] - Q_star[non_terminal])))

            # Policy match: count a state as matching if the learned greedy
            # action is within a small tolerance of the optimal value.
            # This handles ties where multiple actions are equally optimal.
            matches = 0
            for s in non_terminal:
                learned_action = np.argmax(Q[s])
                optimal_val = np.max(Q_star[s])
                # If the learned action's Q* value is within 0.01 of optimal, it's fine
                if Q_star[s, learned_action] >= optimal_val - 0.01:
                    matches += 1
            policy_matches.append(matches / len(non_terminal))

    metrics = {
        "rewards": rewards_per_episode,
        "q_errors": q_errors,
        "mean_abs_errors": mean_abs_errors,
        "policy_matches": policy_matches
    }
    return Q, metrics


# =============================================================================
# 4. SARSA (On-Policy TD Control)
# =============================================================================

def sarsa(env, n_episodes, gamma=0.99, alpha_mode="decay",
          alpha_fixed=0.1, epsilon_mode="decay", epsilon_fixed=0.1,
          Q_star=None):
    """
    SARSA: same as Q-learning except the update uses the ACTUAL next action
    (on-policy) instead of the max (off-policy).

    Only difference from Q-learning is one line in the update:
      Q-learning:  td_target = r + gamma * max_a' Q(s', a')
      SARSA:       td_target = r + gamma * Q(s', a')  where a' is the
                   action actually chosen by the policy
    """
    n_states = env.n_states if hasattr(env, 'n_states') else env.observation_space.n
    n_actions = env.n_actions if hasattr(env, 'n_actions') else env.action_space.n

    Q = np.zeros((n_states, n_actions))
    visit_count = np.zeros((n_states, n_actions))

    rewards_per_episode = []
    q_errors = []
    policy_matches = []

    for ep in range(n_episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]

        if epsilon_mode == "decay":
            epsilon = max(0.01, 1.0 / (1 + ep * 0.01))
        elif epsilon_mode == "fixed":
            epsilon = epsilon_fixed
        else:
            epsilon = 0.0

        # Choose initial action (epsilon-greedy)
        if np.random.random() < epsilon:
            action = np.random.randint(n_actions)
        else:
            action = np.argmax(Q[state])

        total_reward = 0
        done = False
        steps = 0

        while not done and steps < 500:
            result = env.step(action)
            if len(result) == 5:
                next_state, reward, terminated, truncated, _ = result
                done = terminated or truncated
            else:
                next_state, reward, done = result

            # Choose NEXT action (epsilon-greedy) — this is used in the update
            if np.random.random() < epsilon:
                next_action = np.random.randint(n_actions)
            else:
                next_action = np.argmax(Q[next_state])

            visit_count[state, action] += 1

            if alpha_mode == "decay":
                alpha = 10.0 / (10.0 + visit_count[state, action])
            elif alpha_mode == "decay_strict":
                alpha = 1.0 / visit_count[state, action]
            else:
                alpha = alpha_fixed

            # SARSA update: uses the ACTUAL next action, not the max
            td_target = reward + gamma * Q[next_state, next_action] * (1 - done)
            td_error = td_target - Q[state, action]
            Q[state, action] += alpha * td_error

            total_reward += reward
            state = next_state
            action = next_action  # carry forward the chosen action
            steps += 1

        rewards_per_episode.append(total_reward)

        if Q_star is not None:
            non_terminal = [s for s in range(n_states)
                           if not (hasattr(env, 'goal') and s == env.goal)]
            if not non_terminal:
                non_terminal = list(range(n_states))
            q_errors.append(np.max(np.abs(Q[non_terminal] - Q_star[non_terminal])))
            matches = 0
            for s in non_terminal:
                learned_action = np.argmax(Q[s])
                optimal_val = np.max(Q_star[s])
                if Q_star[s, learned_action] >= optimal_val - 0.01:
                    matches += 1
            policy_matches.append(matches / len(non_terminal))

    metrics = {
        "rewards": rewards_per_episode,
        "q_errors": q_errors,
        "policy_matches": policy_matches
    }
    return Q, metrics


# =============================================================================
# 5. DOUBLE Q-LEARNING
# =============================================================================

def double_q_learning(env, n_episodes, gamma=0.99, alpha_mode="decay",
                      alpha_fixed=0.1, epsilon_mode="decay", epsilon_fixed=0.1,
                      Q_star=None):
    """
    Double Q-learning (van Hasselt, 2010).
    Maintains two Q-tables. On each update, randomly pick one table to
    select the best action and use the OTHER table to evaluate it.
    This reduces overestimation bias caused by the max operator.
    """
    n_states = env.n_states if hasattr(env, 'n_states') else env.observation_space.n
    n_actions = env.n_actions if hasattr(env, 'n_actions') else env.action_space.n

    QA = np.zeros((n_states, n_actions))
    QB = np.zeros((n_states, n_actions))
    visit_count = np.zeros((n_states, n_actions))

    rewards_per_episode = []
    q_errors = []
    overestimation = []  # track mean(Q - Q*) to measure positive bias

    for ep in range(n_episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]

        if epsilon_mode == "decay":
            epsilon = max(0.01, 1.0 / (1 + ep * 0.01))
        elif epsilon_mode == "fixed":
            epsilon = epsilon_fixed
        else:
            epsilon = 0.0

        total_reward = 0
        done = False
        steps = 0

        while not done and steps < 500:
            # Action selection uses the sum of both tables
            if np.random.random() < epsilon:
                action = np.random.randint(n_actions)
            else:
                action = np.argmax(QA[state] + QB[state])

            result = env.step(action)
            if len(result) == 5:
                next_state, reward, terminated, truncated, _ = result
                done = terminated or truncated
            else:
                next_state, reward, done = result

            visit_count[state, action] += 1

            if alpha_mode == "decay":
                alpha = 10.0 / (10.0 + visit_count[state, action])
            elif alpha_mode == "decay_strict":
                alpha = 1.0 / visit_count[state, action]
            else:
                alpha = alpha_fixed

            # Randomly update QA or QB
            if np.random.random() < 0.5:
                # Use QA to select best action, QB to evaluate it
                best_a = np.argmax(QA[next_state])
                td_target = reward + gamma * QB[next_state, best_a] * (1 - done)
                QA[state, action] += alpha * (td_target - QA[state, action])
            else:
                # Use QB to select best action, QA to evaluate it
                best_a = np.argmax(QB[next_state])
                td_target = reward + gamma * QA[next_state, best_a] * (1 - done)
                QB[state, action] += alpha * (td_target - QB[state, action])

            total_reward += reward
            state = next_state
            steps += 1

        rewards_per_episode.append(total_reward)

        if Q_star is not None:
            Q_avg = (QA + QB) / 2  # combined estimate
            non_terminal = [s for s in range(n_states)
                           if not (hasattr(env, 'goal') and s == env.goal)]
            if not non_terminal:
                non_terminal = list(range(n_states))
            q_errors.append(np.max(np.abs(Q_avg[non_terminal] - Q_star[non_terminal])))
            overestimation.append(np.mean(Q_avg[non_terminal] - Q_star[non_terminal]))

    metrics = {
        "rewards": rewards_per_episode,
        "q_errors": q_errors,
        "overestimation": overestimation
    }
    return (QA + QB) / 2, metrics


# =============================================================================
# 6. HELPER FUNCTIONS
# =============================================================================

def smooth(data, window=100):
    """Rolling average for cleaner plots."""
    return np.convolve(data, np.ones(window)/window, mode='valid')

def run_multiple_seeds(algorithm_fn, env_fn, n_seeds=10, **kwargs):
    """Run an algorithm over multiple seeds, collect metrics."""
    all_metrics = []
    for seed in range(n_seeds):
        np.random.seed(seed)
        env = env_fn()
        _, metrics = algorithm_fn(env, **kwargs)
        all_metrics.append(metrics)
    return all_metrics

def aggregate_metric(all_metrics, key):
    """Compute mean and std across seeds for a given metric."""
    arrays = [np.array(m[key]) for m in all_metrics if len(m[key]) > 0]
    if not arrays:
        return None, None
    min_len = min(len(a) for a in arrays)
    arrays = [a[:min_len] for a in arrays]
    stacked = np.stack(arrays)
    return stacked.mean(axis=0), stacked.std(axis=0)


# =============================================================================
# EXPERIMENT 1: Core Convergence Verification
# =============================================================================

def experiment_1_convergence():
    """
    Central claim of the paper: Q_n -> Q* with probability 1.
    We verify this by running Q-learning with proper conditions
    (decaying alpha, decaying epsilon) and tracking ||Q_n - Q*||_inf.
    """
    print("=" * 60)
    print("EXPERIMENT 1: Core Convergence Verification")
    print("=" * 60)

    env = GridWorld(size=4)
    P = env.get_transition_model()
    Q_star = value_iteration(P, env.n_states, env.n_actions, gamma=0.99,
                             terminal_states={env.goal})

    print(f"\nQ* computed. Max Q*: {Q_star.max():.4f}, Min Q*: {Q_star.min():.4f}")

    # Run Q-learning with proper conditions across multiple seeds
    n_episodes = 50000
    n_seeds = 10

    all_errors = []
    all_mean_errors = []
    all_policy_matches = []

    for seed in range(n_seeds):
        np.random.seed(seed)
        env = GridWorld(size=4)
        _, metrics = q_learning(
            env, n_episodes=n_episodes, gamma=0.99,
            alpha_mode="decay", epsilon_mode="decay", Q_star=Q_star
        )
        all_errors.append(metrics["q_errors"])
        all_mean_errors.append(metrics["mean_abs_errors"])
        all_policy_matches.append(metrics["policy_matches"])

    errors = np.array(all_errors)
    mean_errs = np.array(all_mean_errors)
    matches = np.array(all_policy_matches)

    mean_error = errors.mean(axis=0)
    std_error = errors.std(axis=0)
    mean_mean_err = mean_errs.mean(axis=0)
    mean_match = matches.mean(axis=0)

    # Plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    episodes = np.arange(n_episodes)
    ax1.plot(episodes, mean_error, color='blue', linewidth=0.5, alpha=0.7, label='Max error')
    ax1.plot(episodes, mean_mean_err, color='red', linewidth=0.5, label='Mean error')
    ax1.fill_between(episodes, mean_error - std_error, mean_error + std_error,
                     alpha=0.1, color='blue')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Error')
    ax1.set_title('Q-Value Error (||Q - Q*||)')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(episodes, mean_match, color='green', linewidth=0.5)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Fraction matching π*')
    ax2.set_title('Policy Correctness')
    ax2.set_ylim([0, 1.05])
    ax2.grid(True, alpha=0.3)

    # Reward per episode
    all_rewards = [metrics["rewards"] for metrics in [metrics]]  # use last seed
    ax3.plot(smooth(np.array(all_errors[0]), 500), color='purple', linewidth=1)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Max Q-Error (smoothed)')
    ax3.set_title('Convergence Trend (smoothed)')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)

    plt.suptitle('Experiment 1: Validating Convergence Theorem', fontsize=14)
    plt.tight_layout()
    plt.savefig('exp1_convergence.png', dpi=150)
    plt.show()

    print(f"\nFinal max error (mean ± std): {mean_error[-1]:.6f} ± {std_error[-1]:.6f}")
    print(f"Final mean error: {mean_mean_err[-1]:.6f}")
    print(f"Final policy match: {mean_match[-1]:.4f}")

    # Debug: print Q* and learned Q for one seed to diagnose
    np.random.seed(0)
    env_dbg = GridWorld(size=4)
    Q_dbg, _ = q_learning(env_dbg, n_episodes=50000, gamma=0.99,
                          alpha_mode="decay", epsilon_mode="decay", Q_star=Q_star)

    print("\n--- DEBUG: Q* (non-terminal, max per state) ---")
    for s in range(env.n_states):
        if s != env.goal:
            print(f"  State {s:2d}: Q*={Q_star[s].max():.4f}  Q_learned={Q_dbg[s].max():.4f}  "
                  f"err={np.abs(Q_dbg[s] - Q_star[s]).max():.4f}  "
                  f"pi*={np.argmax(Q_star[s])} pi_learned={np.argmax(Q_dbg[s])}")


# =============================================================================
# EXPERIMENT 2: Learning Rate Condition Testing
# =============================================================================

def experiment_2_learning_rates():
    """
    The theorem requires: Σα = ∞ and Σα² < ∞.
    - α = 1/n^0.6 (decaying): satisfies both → should converge to Q*
    - α = 0.1 fixed → Σα² = ∞, violates condition → should oscillate
    - α = 0.01 fixed → same violation, but smaller oscillation
    """
    print("=" * 60)
    print("EXPERIMENT 2: Learning Rate Condition Testing")
    print("=" * 60)

    env = GridWorld(size=4)
    P = env.get_transition_model()
    Q_star = value_iteration(P, env.n_states, env.n_actions, gamma=0.99)

    n_episodes = 20000
    n_seeds = 10

    configs = [
        ("α = 1/n^0.6 (decaying)", "decay", 0.0),
        ("α = 0.1 (fixed)", "fixed", 0.1),
        ("α = 0.01 (fixed)", "fixed", 0.01),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    for label, mode, fixed_val in configs:
        all_errors = []
        for seed in range(n_seeds):
            np.random.seed(seed)
            env = GridWorld(size=4)
            _, metrics = q_learning(
                env, n_episodes=n_episodes, gamma=0.99,
                alpha_mode=mode, alpha_fixed=fixed_val,
                epsilon_mode="decay", Q_star=Q_star
            )
            all_errors.append(metrics["q_errors"])

        errors = np.array(all_errors)
        mean_err = errors.mean(axis=0)

        # Smooth for readability
        smoothed = smooth(mean_err, window=200)
        ax.plot(smoothed, label=label, linewidth=1.5)

    ax.set_xlabel('Episode')
    ax.set_ylabel('||Q_n - Q*||∞ (smoothed)')
    ax.set_title('Experiment 2: Effect of Learning Rate on Convergence')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('exp2_learning_rates.png', dpi=150)
    plt.show()


# =============================================================================
# EXPERIMENT 3: Exploration Condition Testing
# =============================================================================

def experiment_3_exploration():
    """
    The theorem requires all state-action pairs visited infinitely often.
    - Decaying ε-greedy: satisfies this (all actions keep getting tried)
    - Pure greedy (ε=0): violates this → suboptimal policy
    """
    print("=" * 60)
    print("EXPERIMENT 3: Exploration Condition Testing")
    print("=" * 60)

    env = GridWorld(size=4)
    P = env.get_transition_model()
    Q_star = value_iteration(P, env.n_states, env.n_actions, gamma=0.99)

    n_episodes = 20000
    n_seeds = 10

    configs = [
        ("ε-greedy (decaying)", "decay", 0.0),
        ("ε = 0.1 (constant)", "fixed", 0.1),
        ("ε = 0 (pure greedy)", "greedy", 0.0),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for label, mode, fixed_val in configs:
        all_errors = []
        all_matches = []
        for seed in range(n_seeds):
            np.random.seed(seed)
            env = GridWorld(size=4)
            _, metrics = q_learning(
                env, n_episodes=n_episodes, gamma=0.99,
                alpha_mode="decay", epsilon_mode=mode,
                epsilon_fixed=fixed_val, Q_star=Q_star
            )
            all_errors.append(metrics["q_errors"])
            all_matches.append(metrics["policy_matches"])

        errors = np.array(all_errors)
        matches = np.array(all_matches)

        smoothed_err = smooth(errors.mean(axis=0), 200)
        smoothed_match = smooth(matches.mean(axis=0), 200)

        ax1.plot(smoothed_err, label=label, linewidth=1.5)
        ax2.plot(smoothed_match, label=label, linewidth=1.5)

    ax1.set_xlabel('Episode')
    ax1.set_ylabel('||Q_n - Q*||∞ (smoothed)')
    ax1.set_title('Q-Value Error')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Fraction matching π*')
    ax2.set_title('Policy Stability')
    ax2.set_ylim([0, 1.05])
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Experiment 3: Effect of Exploration Strategy', fontsize=14)
    plt.tight_layout()
    plt.savefig('exp3_exploration.png', dpi=150)
    plt.show()


# =============================================================================
# EXPERIMENT 4: Discount Factor Analysis
# =============================================================================

def experiment_4_discount():
    """
    The theorem requires γ < 1. We test how different values of γ
    affect convergence speed and policy quality.
    """
    print("=" * 60)
    print("EXPERIMENT 4: Discount Factor Analysis")
    print("=" * 60)

    n_episodes = 50000
    n_seeds = 10
    gammas = [0.1, 0.5, 0.9, 0.99]

    fig, ax = plt.subplots(figsize=(10, 6))

    for gamma in gammas:
        env = GridWorld(size=4)
        P = env.get_transition_model()
        Q_star = value_iteration(P, env.n_states, env.n_actions, gamma=gamma,
                                 terminal_states={env.goal})

        all_errors = []
        for seed in range(n_seeds):
            np.random.seed(seed)
            env = GridWorld(size=4)
            _, metrics = q_learning(
                env, n_episodes=n_episodes, gamma=gamma,
                alpha_mode="decay", epsilon_mode="decay", Q_star=Q_star
            )
            all_errors.append(metrics["q_errors"])

        errors = np.array(all_errors)
        smoothed = smooth(errors.mean(axis=0), 200)
        ax.plot(smoothed, label=f'γ = {gamma}', linewidth=1.5)

    ax.set_xlabel('Episode')
    ax.set_ylabel('||Q_n - Q*||∞ (smoothed)')
    ax.set_title('Experiment 4: Effect of Discount Factor on Convergence')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('exp4_discount.png', dpi=150)
    plt.show()


# =============================================================================
# EXPERIMENT 5: Q-Learning vs SARSA on CliffWalking
# =============================================================================

def experiment_5_qlearning_vs_sarsa():
    """
    Q-learning is off-policy: it learns about the optimal policy even while
    exploring. SARSA is on-policy: it learns about the policy it's actually
    following (including exploratory moves).

    On CliffWalking, this produces different behavior:
    - Q-learning: learns the optimal (cliff-edge) path
    - SARSA: learns the safer (away from cliff) path because it accounts
      for the fact that epsilon-greedy exploration might step off the cliff
    """
    print("=" * 60)
    print("EXPERIMENT 5: Q-Learning vs SARSA (CliffWalking)")
    print("=" * 60)

    n_episodes = 5000
    n_seeds = 10

    ql_rewards_all = []
    sarsa_rewards_all = []

    for seed in range(n_seeds):
        np.random.seed(seed)
        env_ql = gym.make("CliffWalking-v1")
        Q_ql, metrics_ql = q_learning(
            env_ql, n_episodes=n_episodes, gamma=0.99,
            alpha_mode="fixed", alpha_fixed=0.1,
            epsilon_mode="fixed", epsilon_fixed=0.1
        )
        ql_rewards_all.append(metrics_ql["rewards"])

        np.random.seed(seed)
        env_sarsa = gym.make("CliffWalking-v1")
        Q_sarsa, metrics_sarsa = sarsa(
            env_sarsa, n_episodes=n_episodes, gamma=0.99,
            alpha_mode="fixed", alpha_fixed=0.1,
            epsilon_mode="fixed", epsilon_fixed=0.1
        )
        sarsa_rewards_all.append(metrics_sarsa["rewards"])

    ql_rewards = np.array(ql_rewards_all)
    sarsa_rewards = np.array(sarsa_rewards_all)

    # Plot reward comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    ql_smoothed = smooth(ql_rewards.mean(axis=0), 100)
    sarsa_smoothed = smooth(sarsa_rewards.mean(axis=0), 100)

    ax.plot(ql_smoothed, label='Q-learning', linewidth=1.5)
    ax.plot(sarsa_smoothed, label='SARSA', linewidth=1.5)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward per Episode (smoothed)')
    ax.set_title('Experiment 5: Q-Learning vs SARSA on CliffWalking')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('exp5_ql_vs_sarsa.png', dpi=150)
    plt.show()

    # Print learned policies
    print("\nQ-learning learned policy (CliffWalking 4x12 grid):")
    print("Actions: 0=up, 1=right, 2=down, 3=left")
    action_symbols = ['↑', '→', '↓', '←']
    policy_ql = np.argmax(Q_ql, axis=1)
    policy_sarsa = np.argmax(Q_sarsa, axis=1)

    print("\nQ-learning policy:")
    for row in range(4):
        line = ""
        for col in range(12):
            s = row * 12 + col
            if s == 36:  # start
                line += " S "
            elif s == 47:  # goal
                line += " G "
            elif 37 <= s <= 46:  # cliff
                line += " C "
            else:
                line += f" {action_symbols[policy_ql[s]]} "
        print(line)

    print("\nSARSA policy:")
    for row in range(4):
        line = ""
        for col in range(12):
            s = row * 12 + col
            if s == 36:
                line += " S "
            elif s == 47:
                line += " G "
            elif 37 <= s <= 46:
                line += " C "
            else:
                line += f" {action_symbols[policy_sarsa[s]]} "
        print(line)


# =============================================================================
# EXPERIMENT 6: Overestimation Bias (Q-learning vs Double Q-learning)
# =============================================================================

def experiment_6_overestimation():
    """
    Standard Q-learning uses max_a Q(s', a) in its update. In stochastic
    environments, this max introduces a positive bias because:
      E[max(X1, X2, ...)] >= max(E[X1], E[X2], ...)

    Double Q-learning fixes this by decoupling action selection from
    evaluation using two Q-tables.
    """
    print("=" * 60)
    print("EXPERIMENT 6: Overestimation Bias")
    print("=" * 60)

    # Use FrozenLake (stochastic) where overestimation is more visible
    n_episodes = 50000
    n_seeds = 10

    # We need Q* for FrozenLake
    env_temp = gym.make("FrozenLake-v1", is_slippery=True)
    P_frozen = env_temp.unwrapped.P
    n_states_fl = env_temp.observation_space.n
    n_actions_fl = env_temp.action_space.n

    # Convert FrozenLake's P format to our format
    Q_star = np.zeros((n_states_fl, n_actions_fl))
    gamma = 0.99
    for _ in range(10000):
        Q_new = np.zeros_like(Q_star)
        for s in range(n_states_fl):
            for a in range(n_actions_fl):
                for prob, ns, reward, done in P_frozen[s][a]:
                    if done:
                        Q_new[s, a] += prob * reward
                    else:
                        Q_new[s, a] += prob * (reward + gamma * np.max(Q_star[ns]))
        if np.max(np.abs(Q_new - Q_star)) < 1e-10:
            break
        Q_star = Q_new

    print(f"FrozenLake Q* computed. Max Q*: {Q_star.max():.4f}")

    ql_overest_all = []
    dql_overest_all = []
    ql_errors_all = []
    dql_errors_all = []

    for seed in range(n_seeds):
        np.random.seed(seed)
        env = gym.make("FrozenLake-v1", is_slippery=True)
        _, m_ql = q_learning(
            env, n_episodes=n_episodes, gamma=gamma,
            alpha_mode="decay", epsilon_mode="decay", Q_star=Q_star
        )
        ql_errors_all.append(m_ql["q_errors"])

        # Compute overestimation for Q-learning manually
        # (re-run to get Q table at intervals)
        np.random.seed(seed)
        env = gym.make("FrozenLake-v1", is_slippery=True)
        Q_ql, _ = q_learning(
            env, n_episodes=n_episodes, gamma=gamma,
            alpha_mode="decay", epsilon_mode="decay", Q_star=Q_star
        )
        ql_overest_all.append(np.mean(Q_ql - Q_star))

        np.random.seed(seed)
        env = gym.make("FrozenLake-v1", is_slippery=True)
        Q_dql, m_dql = double_q_learning(
            env, n_episodes=n_episodes, gamma=gamma,
            alpha_mode="decay", epsilon_mode="decay", Q_star=Q_star
        )
        dql_errors_all.append(m_dql["q_errors"])
        dql_overest_all.append(np.mean(Q_dql - Q_star))

    ql_errors = np.array(ql_errors_all)
    dql_errors = np.array(dql_errors_all)

    fig, ax = plt.subplots(figsize=(10, 6))
    ql_smoothed = smooth(ql_errors.mean(axis=0), 200)
    dql_smoothed = smooth(dql_errors.mean(axis=0), 200)

    ax.plot(ql_smoothed, label='Q-learning', linewidth=1.5)
    ax.plot(dql_smoothed, label='Double Q-learning', linewidth=1.5)
    ax.set_xlabel('Episode')
    ax.set_ylabel('||Q_n - Q*||∞ (smoothed)')
    ax.set_title('Experiment 6: Q-learning vs Double Q-learning (FrozenLake)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('exp6_overestimation.png', dpi=150)
    plt.show()

    print(f"\nFinal mean overestimation (Q - Q*):")
    print(f"  Q-learning:        {np.mean(ql_overest_all):.6f}")
    print(f"  Double Q-learning: {np.mean(dql_overest_all):.6f}")


# =============================================================================
# EXPERIMENT 7: Scalability (Taxi-v3)
# =============================================================================

def experiment_7_scalability():
    """
    Test Q-learning on a larger state space (Taxi-v3: 500 states, 6 actions)
    to see if convergence still occurs within reasonable episodes.
    """
    print("=" * 60)
    print("EXPERIMENT 7: Scalability on Taxi-v3")
    print("=" * 60)

    n_episodes = 50000
    n_seeds = 5

    all_rewards = []
    for seed in range(n_seeds):
        np.random.seed(seed)
        env = gym.make("Taxi-v3")
        _, metrics = q_learning(
            env, n_episodes=n_episodes, gamma=0.99,
            alpha_mode="decay", epsilon_mode="decay"
        )
        all_rewards.append(metrics["rewards"])

    rewards = np.array(all_rewards)
    mean_rewards = rewards.mean(axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    smoothed = smooth(mean_rewards, 500)
    ax.plot(smoothed, linewidth=1.5, color='purple')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward per Episode (smoothed)')
    ax.set_title('Experiment 7: Q-Learning on Taxi-v3 (500 states)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('exp7_taxi.png', dpi=150)
    plt.show()

    print(f"\nFinal avg reward (last 1000 episodes): {mean_rewards[-1000:].mean():.2f}")


# =============================================================================
# Q-VALUE HEATMAP VISUALIZATION
# =============================================================================

def visualize_q_values(Q, Q_star, size=4):
    """Show learned Q-values vs optimal Q-values as heatmaps."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    V_learned = np.max(Q, axis=1).reshape(size, size)
    V_star = np.max(Q_star, axis=1).reshape(size, size)
    V_diff = np.abs(V_learned - V_star).reshape(size, size)

    im1 = axes[0].imshow(V_learned, cmap='YlOrRd')
    axes[0].set_title('Learned V(s) = max_a Q(s,a)')
    plt.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(V_star, cmap='YlOrRd')
    axes[1].set_title('Optimal V*(s)')
    plt.colorbar(im2, ax=axes[1])

    im3 = axes[2].imshow(V_diff, cmap='Blues')
    axes[2].set_title('|V(s) - V*(s)|')
    plt.colorbar(im3, ax=axes[2])

    # Add grid labels
    for ax in axes:
        ax.set_xticks(range(size))
        ax.set_yticks(range(size))
        for i in range(size):
            for j in range(size):
                state = i * size + j
                if state == 0:
                    ax.text(j, i, 'S', ha='center', va='center', fontsize=10)
                elif state == size*size - 1:
                    ax.text(j, i, 'G', ha='center', va='center', fontsize=10)

    plt.suptitle('Q-Value Heatmap Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig('q_value_heatmap.png', dpi=150)
    plt.show()


def visualize_policy(Q, size=4):
    """Show learned policy as arrows on the grid."""
    action_symbols = ['↑', '→', '↓', '←']
    policy = np.argmax(Q, axis=1)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(-0.5, size - 0.5)
    ax.set_ylim(-0.5, size - 0.5)
    ax.invert_yaxis()

    for s in range(size * size):
        row, col = divmod(s, size)
        if s == 0:
            ax.text(col, row, 'S', ha='center', va='center', fontsize=16, color='blue')
        elif s == size * size - 1:
            ax.text(col, row, 'G', ha='center', va='center', fontsize=16, color='green')
        else:
            ax.text(col, row, action_symbols[policy[s]],
                    ha='center', va='center', fontsize=18)

    ax.set_xticks(range(size))
    ax.set_yticks(range(size))
    ax.grid(True)
    ax.set_title('Learned Policy')
    plt.tight_layout()
    plt.savefig('learned_policy.png', dpi=150)
    plt.show()


# =============================================================================
# MAIN — Run all experiments
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("CS5100 CAPSTONE: Q-Learning Convergence Validation")
    print("Based on Watkins & Dayan (1992)")
    print("=" * 60 + "\n")

    # Run core experiments
    experiment_1_convergence()
    experiment_2_learning_rates()
    experiment_3_exploration()
    experiment_4_discount()
    experiment_5_qlearning_vs_sarsa()
    experiment_6_overestimation()
    experiment_7_scalability()

    # Generate visualizations for the presentation
    print("\n" + "=" * 60)
    print("Generating Q-value visualizations...")
    print("=" * 60)

    env = GridWorld(size=4)
    P = env.get_transition_model()
    Q_star = value_iteration(P, env.n_states, env.n_actions, gamma=0.99,
                             terminal_states={env.goal})

    np.random.seed(42)
    env = GridWorld(size=4)
    Q_learned, _ = q_learning(
        env, n_episodes=50000, gamma=0.99,
        alpha_mode="decay", epsilon_mode="decay", Q_star=Q_star
    )

    visualize_q_values(Q_learned, Q_star, size=4)
    visualize_policy(Q_learned, size=4)

    print("\nAll experiments complete. Plots saved.")