# Empirical Validation of Q-Learning Convergence

**CS5100 — Foundations of Artificial Intelligence, Spring 2026**
<br>
**Author:** Aditya Kejrewal

## Video Link:
[Sharepoint Link](https://northeastern-my.sharepoint.com/:v:/g/personal/kejrewal_a_northeastern_edu/IQALVcbPgaeUR4tSL4KswSFFAa8_zACzB0oNQDVbDWOGYos?e=fPVTRU&nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifX0%3D)

## Repository Link:
[Github Link](https://github.com/adikejre/QLearning_CS5100/tree/main)

---

## 1. Introduction

Reinforcement learning addresses the problem of how an agent can learn to make decisions in an unknown environment by interacting with it and observing rewards. A central question in the field is whether a learning algorithm can be guaranteed to converge to the correct solution. This project focuses on one of the foundational answers to that question: the convergence proof for Q-learning presented by Watkins and Dayan (1992).

Q-learning is a model-free reinforcement learning algorithm that enables an agent to learn optimal behavior without knowing the environment's transition probabilities or reward structure. The 1992 paper is significant because it provides the first rigorous proof that Q-learning converges to the optimal action-value function Q* with probability 1 under specified conditions. However, the paper is entirely theoretical and contains no empirical experiments, benchmarks, or demonstrations.

This project was selected because it presents an opportunity to bridge that gap: to treat the theorem's claims as testable hypotheses and validate them empirically through controlled experiments. The project also extends beyond basic convergence verification by comparing Q-learning against SARSA (an on-policy variant), testing convergence on stochastic environments, and evaluating scalability.

## 2. Paper Overview

### Core Idea

Q-learning (Watkins, 1989) is a model-free, off-policy reinforcement learning algorithm. An agent interacts with a Markov Decision Process (MDP) by observing states, taking actions, and receiving rewards. It maintains a Q-table where each entry Q(s, a) estimates the expected discounted return for taking action a in state s and acting optimally thereafter. The Q-values are updated incrementally using the rule:

Q(s, a) ← Q(s, a) + α · [r + γ · max_{a'} Q(s', a') - Q(s, a)]

Here, α is the learning rate, r is the observed reward, γ is the discount factor, and s' is the observed next state. The term r + γ · max_{a'} Q(s', a') is the TD target (what the agent thinks Q(s, a) should be based on the latest experience), and the difference between this target and the current estimate is the TD error.

The key property of Q-learning is that it is off-policy: the update uses max_{a'} Q(s', a'), the value of the best action at the next state, regardless of which action the agent actually takes. This allows the agent to learn about the optimal policy while following an exploratory one.

### Main Contribution

The paper's primary contribution is a formal convergence proof. It proves that Q_n(s, a) → Q*(s, a) with probability 1 as the number of episodes n → ∞, under the following conditions:

1. The environment is a finite MDP (finite states and actions)
2. Q-values are stored in a lookup table (one entry per state-action pair)
3. Every state-action pair is visited infinitely often
4. Learning rates satisfy: Σα = ∞ and Σα² < ∞
5. Rewards are bounded (|r| < R for some finite R)
6. Discount factor γ < 1

The proof constructs an artificial Markov process called the Action Replay Process (ARP), which replays past episodes probabilistically. Two key lemmas show that (a) Q-learning's values are exactly the optimal values of the ARP, and (b) the ARP converges to the real environment as episodes accumulate. Combined, these imply that Q-learning's values converge to Q*.

### Experimental Setup in the Paper

The paper contains no experiments. No benchmarks, no learning curves, no environment specifications, no hyperparameter settings. The contribution is purely theoretical.

## 3. Reproduction Objective

Since the paper has no empirical results to reproduce directly, the reproduction takes the form of empirically validating the theoretical claims. Specifically:

1. **Core convergence:** Demonstrate that Q-learning's values converge to Q* (computed via value iteration) under proper conditions, on environments where Q* can be computed exactly.

2. **Condition testing:** Systematically violate each theorem condition individually (learning rate decay, exploration, discount factor) and show that convergence degrades.

3. **Algorithm comparison:** Compare Q-learning (off-policy) with SARSA (on-policy) to demonstrate the behavioral consequences of the off-policy update rule described in the paper.

4. **Stochastic environments:** Test convergence on FrozenLake, where transitions are stochastic, to verify the theorem applies beyond deterministic settings.

5. **Scalability:** Test on Taxi-v3 (500 states) to evaluate performance on a larger state space within the tabular setting.

The following were intentionally excluded: function approximation experiments (outside the paper's scope), the full ARP proof reproduction (the project focuses on empirical validation, not re-derivation), and the TD(λ) extension (which the paper notes does not trivially extend).

## 4. Methodology

### Implementation

All algorithms were implemented from scratch in Python using only NumPy and Matplotlib. No RL libraries (such as Stable-Baselines) were used. The implementations include:

**Value Iteration:** Standard dynamic programming algorithm. Takes the full transition model P(s'|s,a) and reward function as input. Iteratively applies the Bellman optimality equation to every state-action pair until convergence (max change < 10⁻¹⁰). Produces the exact Q* used as ground truth.

**Tabular Q-Learning:** Implements the update rule from the paper (equation 2). Configurable learning rate (decaying or fixed), exploration strategy (decaying ε-greedy, constant ε, or pure greedy), and discount factor. Tracks visit counts per state-action pair for the decaying learning rate.

**SARSA:** Identical to Q-learning except the update uses Q(s', a'_actual) instead of max_{a'} Q(s', a'), where a'_actual is the action chosen by the ε-greedy policy at the next state. This is the on-policy counterpart to Q-learning.

### Environments

**GridWorld (custom, 4x4):** A 4×4 grid where the agent starts at the top-left (state 0) and must reach the bottom-right (state 15). Actions are up, right, down, left. Walls bounce the agent back. Reward: +1.0 at goal, -0.1 per step. Deterministic transitions. 16 states, 4 actions. Small enough for exact Q* computation and visual inspection.

**FrozenLake 4x4 (Gymnasium):** A 4×4 grid with stochastic ("slippery") transitions. Choosing an action results in movement in the intended direction only 1/3 of the time; the other 2/3 of the time the agent slips to a perpendicular direction. Tests convergence under stochastic dynamics.

**CliffWalking (Gymnasium):** A 4×12 grid where the bottom row (except start and goal) is a cliff with -100 reward. Standard benchmark for comparing Q-learning and SARSA.

**Taxi-v3 (Gymnasium):** A taxi navigation task with 500 discrete states and 6 actions. Tests scalability within the tabular setting.

### Hyperparameters

**Learning rate:** For "decaying" mode, α = c/(c + n) where c = 10 and n is the visit count for the state-action pair. This satisfies the theorem's conditions: Σ c/(c+n) behaves like the harmonic series (diverges) and Σ [c/(c+n)]² converges. For "fixed" mode, α is constant (0.1 or 0.01).

**Exploration:** For "decaying" mode, ε = 1/(1 + 0.01 × episode) with a floor at 0.01. This starts with near-complete exploration and gradually shifts to exploitation while maintaining a minimum exploration rate.

**Discount factor:** γ = 0.99 unless otherwise specified.

**Episodes:** 50,000 for GridWorld experiments, 5,000 for CliffWalking, 50,000 for FrozenLake, 50,000 for Taxi.

**Seeds:** 10 random seeds for each experiment to assess consistency.

### Metrics

**Max absolute error (||Q - Q*||∞):** The largest absolute difference between learned and true Q-values across all non-terminal state-action pairs. Directly measures the paper's convergence claim.

**Mean absolute error:** The average absolute difference across all non-terminal state-action pairs. More representative of overall convergence than the max, which can be dominated by a single outlier.

**Policy correctness:** The fraction of non-terminal states where the greedy policy from Q matches the optimal policy from Q*. Accounts for ties (multiple actions with near-equal Q* values).

**Cumulative reward:** Total reward per episode, smoothed for readability.

## 5. Experiments

### Experiment 1: Core Convergence Verification

Q-learning was run on 4x4 GridWorld with decaying α, decaying ε, and γ = 0.99. Q* was computed via value iteration. The max error, mean error, and policy correctness were tracked over 50,000 episodes across 10 seeds.

### Experiment 2: Learning Rate Condition Testing

Three learning rate schedules were compared on 4x4 GridWorld: decaying α = c/(c+n) (satisfies theorem), fixed α = 0.1 (violates Σα² < ∞), and fixed α = 0.01 (same violation, smaller step).

### Experiment 3: Exploration Condition Testing

Three exploration strategies were compared: decaying ε-greedy (satisfies infinite visitation), constant ε = 0.1, and pure greedy ε = 0 (violates infinite visitation).

### Experiment 4: Discount Factor Analysis

Q-learning was run with γ ∈ {0.1, 0.5, 0.9, 0.99} on 4x4 GridWorld to compare convergence rates.

### Experiment 5: Q-Learning vs. SARSA

Both algorithms were run on CliffWalking with fixed α = 0.1 and constant ε = 0.1 for 5,000 episodes. Learned policies and reward curves were compared.

### Experiment 6: Stochastic Environment (FrozenLake)

Q-learning was run on FrozenLake (slippery transitions) to test convergence in a stochastic environment where the theorem still applies but transitions are noisy.

### Experiment 7: Scalability (Taxi-v3)

Q-learning was run on Taxi-v3 (500 states, 6 actions) for 50,000 episodes to evaluate scalability within the tabular setting.

## 6. Results

### Experiment 1: Core Convergence

The mean absolute error dropped to 0.003 within the first few thousand episodes, confirming that Q-learning converges to Q* under proper conditions. The policy became 100% optimal (every non-terminal state's greedy action matched π*) early in training.

The max error remained around 0.1. Debugging revealed this was driven entirely by a single state-action pair: the "go left" action at the bottom-left corner (state 12), which bounces off the wall. The agent learned not to take this action, so it was visited rarely and its Q-value converged slowly. The per-state analysis confirmed that Q-values for all useful actions were accurate to within 0.006 of Q*.

The Q-value heatmap comparison showed the learned V(s) = max_a Q(s,a) was visually indistinguishable from V*(s), with maximum per-state error of 0.00016.

<figure>
  <img src="plots\exp1_convergence.png" alt="Description">
</figure>

<figure>
  <img src="plots\q_value_heatmap.png" alt="Description">
</figure>


### Experiment 2: Learning Rate Conditions

The decaying learning rate achieved the lowest final error. Fixed α = 0.1 converged initially but plateaued at a higher error because the constant step size prevents Q-values from settling exactly. Fixed α = 0.01 converged slowest because the small step size limits how quickly the agent can learn.

This validates the theorem's learning rate conditions. The decaying schedule satisfies both Σα = ∞ and Σα² < ∞, while fixed rates violate Σα² < ∞ and produce permanent oscillation.

<figure>
  <img src="plots\exp2_learning_rates.png" alt="Description">
</figure>


### Experiment 3: Exploration Conditions

Decaying ε-greedy achieved the lowest error. Pure greedy (ε = 0) performed worst, with error remaining near 1.0 because the agent locked into its initial path and many state-action pairs were never visited. Constant ε = 0.1 performed better than greedy but worse than decaying, with error slowly decreasing.

This validates the infinite visitation condition: without sufficient exploration, convergence fails.

<figure>
  <img src="plots\exp3_exploration.png" alt="Description">
</figure>


### Experiment 4: Discount Factor

Lower γ values converged faster and to lower error. γ = 0.1 reached error ~0.00004 quickly, while γ = 0.99 plateaued around 0.1. This is expected: lower γ means Q-values are smaller (less to learn) and the agent focuses on near-term rewards (simpler problem). All curves decreased, confirming the theorem holds for any γ < 1.


<figure>
  <img src="plots\exp4_discount.png" alt="Description">
</figure>


### Experiment 5: Q-Learning vs. SARSA

On CliffWalking, SARSA achieved higher average reward during training (~-20 per episode) compared to Q-learning (~-50). The learned policies differed meaningfully: Q-learning's policy followed the cliff edge (row 2), while SARSA's policy took the safer top-row path.

This occurs because Q-learning is off-policy: its update uses max (assuming optimal play), so it values the cliff-edge path highest. But during execution with ε = 0.1, random actions occasionally step off the cliff. SARSA is on-policy: it uses the actual next action in its update, so random cliff-falls directly lower the Q-values near the cliff, causing SARSA to learn the safer path.

<figure>
  <img src="plots\exp5_ql_vs_sarsa.png" alt="Description">
</figure>


### Experiment 6: FrozenLake (Stochastic)

Q-learning's error decreased on FrozenLake but more slowly and with more noise than on deterministic GridWorld. This is expected: stochastic transitions mean each sample provides noisier information about the true transition probabilities, requiring more episodes for the Q-values to stabilize.


<figure>
  <img src="plots\exp6_frozenlake.png" alt="Description">
</figure>


### Experiment 7: Scalability

On Taxi-v3, reward climbed from approximately -150 (random behavior) to +7 (successful taxi trips) over 50,000 episodes. This demonstrates Q-learning works on a state space more than 30x larger than GridWorld, though convergence takes proportionally more episodes.

<figure>
  <img src="plots\exp7_taxi.png" alt="Description">
</figure>


## 7. Analysis and Discussion

The experimental results strongly support the paper's theoretical claims. Under the conditions specified in the theorem (decaying learning rate, sufficient exploration, tabular representation, bounded rewards, γ < 1), Q-learning's values converge to Q* and the learned policy becomes optimal.

The condition-violation experiments (experiments 2 and 3) demonstrate that the theorem's conditions are not just theoretical formalities. Each violated condition produces measurably different behavior: fixed learning rates cause persistent oscillation; insufficient exploration causes convergence failure. This is a useful practical finding that the purely theoretical paper cannot provide.

A notable observation is that **policy convergence is much faster than value convergence**. The agent finds the correct actions (100% policy match) long before every Q-value is accurate. This makes sense: the policy only depends on the relative ordering of Q-values at each state, not their exact magnitudes. The max error can remain elevated due to rarely-visited state-action pairs (like wall-bounce actions) without affecting the policy.

The Q-learning vs. SARSA comparison (experiment 5) provides a clear demonstration of the off-policy property described in the paper. The theoretical distinction (max vs. actual next action) produces a concrete behavioral difference in practice.

### Discrepancies

Since the paper contains no empirical results, there are no direct discrepancies to report. However, a practical gap between theory and practice was observed: the learning rate schedule 1/n, which satisfies the theorem's conditions, decayed too aggressively in small environments where visit counts accumulate quickly. The alternative schedule c/(c+n) with c=10 also satisfies the conditions but converges more effectively in practice. The theorem guarantees asymptotic convergence but provides no guidance on finite-time convergence speed, which depends heavily on these practical choices.

## 8. Challenges in Reproduction

**Terminal state handling:** The initial implementation computed Q* with nonzero values at the goal state (since value iteration considers the reward for arriving there), while Q-learning's goal-state values remained at zero (since the agent never takes actions from a terminal state). This discrepancy inflated the error metric until terminal states were excluded from evaluation.

**Learning rate tuning:** The theoretically motivated 1/n schedule converged too slowly for practical experiments. Visit counts on a 16-state grid grow quickly, driving α to near-zero before value propagation from the goal reached distant states. The c/(c+n) schedule resolved this while still satisfying the convergence conditions.

**Error metric interpretation:** The max absolute error (||Q - Q*||∞) was dominated by a single rarely-visited state-action pair (wall-bounce at a corner). The mean error was orders of magnitude smaller (0.003 vs. 0.1). Both metrics are useful but tell different stories about convergence. Reporting both was necessary for an accurate picture.

**Stochastic environments:** FrozenLake required substantially more episodes than deterministic GridWorld because each transition provides a noisy sample. The theorem guarantees eventual convergence but says nothing about how many episodes are needed.

## 9. Limitations

This study has several limitations. All experiments use small, tabular environments (16 to 500 states). The theorem's most practically important limitation, the inapplicability to function approximation, was not empirically tested. Testing Q-learning with neural network approximation on larger environments would strengthen the analysis.

The number of random seeds (10) is relatively small. While the results are consistent, more seeds would provide tighter confidence intervals. Additionally, only one decaying learning rate schedule was tested against fixed rates; a more comprehensive comparison of valid schedules (1/n, 1/√n, c/(c+n) with different c values) would clarify the practical impact of schedule choice.

The Double Q-learning comparison, originally planned, was not completed. This extension would have tested whether overestimation bias is empirically observable in these environments.

## 10. Conclusion and Future Work

This project empirically validates the convergence theorem from Watkins and Dayan (1992) through seven controlled experiments. The core finding is that Q-learning's values converge to Q* when all theorem conditions are satisfied, and that violating individual conditions produces measurable degradation in convergence.

Key insights from the reproduction include: policy convergence is much faster than value convergence; practical learning rate schedules matter more than the theorem suggests; and the off-policy property produces concrete behavioral differences (Q-learning vs. SARSA on CliffWalking).

Future work could extend this study by implementing Double Q-learning for overestimation bias analysis, testing Q-learning with linear function approximation to empirically demonstrate the convergence failure the paper warns about, and applying the framework to larger environments where convergence takes significantly longer.

### Reproducibility Verdict: Partially Reproducible

The paper is classified as **partially reproducible**. The algorithm itself is precisely specified and trivially implementable. The convergence theorem is stated with explicit conditions. However, the paper contains no experiments, so there are no specific empirical results to reproduce. The theoretical proof, while complete, relies on the ARP construction which is complex enough that verifying every detail requires significant mathematical background. All conditions for convergence are clearly stated, but practical choices (learning rate schedule, exploration strategy, episode count) are left entirely unspecified, requiring the reproducer to make design decisions that significantly affect finite-time behavior. The theorem guarantees asymptotic convergence but provides no guidance on how many episodes are needed in practice, which is the question practitioners care most about.

---

## References

Watkins, C. J. C. H., & Dayan, P. (1992). Q-learning. *Machine Learning*, 8, 279–292.

Watkins, C. J. C. H. (1989). *Learning from Delayed Rewards*. PhD Thesis, University of Cambridge.

van Hasselt, H. (2010). Double Q-learning. *Advances in Neural Information Processing Systems*, 23.
