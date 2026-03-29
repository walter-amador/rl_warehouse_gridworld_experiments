# -*- coding: utf-8 -*-
"""
COSC-4117EL: Assignment 2
Q-Learning with Exploration Bonus (Visit-Count Based)

Group 8: Walter Amador and Babatunde Omoyele
Note: This file was developed with the assistance of Claude Code (claude.ai/code).
All AI-generated code is acknowledged per assignment guidelines.

CHECKPOINT 7: Exploration Bonus Agent
--------------------------------------
Instead of randomly picking actions (epsilon-greedy), this agent always picks
the action that maximises an exploration function:

    f(Q(s,a), N(s,a)) = Q(s,a) + k / sqrt(N(s,a) + 1)

Where:
  Q(s,a)  = current learned Q-value  (exploit: go where reward is expected)
  N(s,a)  = visit count for (s,a)    (explore: go where we've been least)
  k       = exploration constant     (how much to weight curiosity vs reward)

Key difference from epsilon-greedy:
  - Epsilon-greedy: random chance determines exploration (stochastic)
  - Exploration bonus: exploration is DELIBERATE — least-visited pairs get
    a mathematical bonus that shrinks as we visit them more (deterministic)

As N(s,a) grows, k/sqrt(N+1) → 0, so the agent naturally shifts from
exploring to exploiting WITHOUT a decay schedule — it emerges from the math.
"""

import numpy as np
import matplotlib
from warehouse_gridworld_domain_random import (
    WarehouseGridWorld,
    ACTIONS,
)

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42

# ── Shared hyperparameters (same as best config from epsilon-greedy) ──────────
ALPHA        = 0.5
GAMMA        = 0.9
NUM_EPISODES = 500
SMOOTHING_WINDOW = 20

# ── k values to experiment with ───────────────────────────────────────────────
#
# k=0.5 → small bonus, Q-values dominate quickly, less exploration pressure
# k=2.0 → moderate bonus, balanced exploration and exploitation
# k=5.0 → large bonus, agent is very curious, explores aggressively
#
K_VALUES = [0.5, 2.0, 5.0]


# ── Core: the exploration function ───────────────────────────────────────────

def exploration_function(q_value: float, visit_count: int, k: float) -> float:
    """
    f(Q(s,a), N(s,a)) = Q(s,a) + k / sqrt(N(s,a) + 1)

    The bonus k/sqrt(N+1) is large when N is small (unvisited → explore)
    and approaches 0 as N grows (well-visited → trust Q-value).
    The +1 prevents division by zero on the very first visit.
    """
    return q_value + k / np.sqrt(visit_count + 1)


def select_action_bonus(
    q_table: np.ndarray,
    visit_counts: np.ndarray,
    state_idx: int,
    k: float,
) -> int:
    """
    Pick the action with the highest exploration function value.
    No randomness — the bonus itself drives exploration.
    """
    scores = [
        exploration_function(q_table[state_idx, a], visit_counts[state_idx, a], k)
        for a in range(len(ACTIONS))
    ]
    return int(np.argmax(scores))


# ── Training loop ─────────────────────────────────────────────────────────────

def train_exploration_bonus(
    world: WarehouseGridWorld,
    alpha: float,
    gamma: float,
    num_episodes: int,
    k: float,
) -> tuple[np.ndarray, np.ndarray, list[float], list[bool]]:
    """
    Train a Q-learning agent using the exploration bonus for action selection.

    Returns
    -------
    q_table       : learned Q-values, shape (n_states, n_actions)
    visit_counts  : how many times each (state, action) was visited
    episode_rewards
    episode_success
    """
    n_states  = world.get_state_space_size()
    n_actions = world.get_action_space_size()

    q_table      = np.zeros((n_states, n_actions))
    visit_counts = np.zeros((n_states, n_actions), dtype=np.int32)

    episode_rewards = []
    episode_success = []

    for episode in range(num_episodes):
        state     = world.reset()
        state_idx = world.state_to_index(state)
        total_reward = 0.0
        done = False

        while not done:
            # 1. Select action using exploration function (no randomness)
            action_idx = select_action_bonus(q_table, visit_counts, state_idx, k)
            action     = ACTIONS[action_idx]

            # 2. Increment visit count BEFORE the update
            visit_counts[state_idx, action_idx] += 1

            # 3. Take step, observe outcome
            result        = world.step(action)
            new_state_idx = world.state_to_index(result.state)

            # 4. Standard Q-learning update (same as epsilon-greedy)
            td_target = result.reward + gamma * np.max(q_table[new_state_idx])
            td_error  = td_target - q_table[state_idx, action_idx]
            q_table[state_idx, action_idx] += alpha * td_error

            total_reward += result.reward
            state_idx     = new_state_idx
            done          = result.done

        episode_rewards.append(total_reward)
        episode_success.append(world.delivered and (world.robot_pos == world.dock_pos))

    return q_table, visit_counts, episode_rewards, episode_success


# ── Checkpoint 7 ─────────────────────────────────────────────────────────────

def checkpoint_7_exploration_bonus():
    """
    Train with 3 values of k and compare behavior.
    Key questions:
      - Does the agent explore MORE or LESS than epsilon-greedy?
      - How does k change when the agent commits to exploitation?
      - Which k finds the task fastest?
    """
    print("=" * 60)
    print("CHECKPOINT 7 — Exploration Bonus Agent")
    print(f"  k values={K_VALUES}  alpha={ALPHA}  gamma={GAMMA}  episodes={NUM_EPISODES}")
    print("=" * 60)

    # ── How the bonus changes with visit count ────────────────────────────────
    print("\n[1] How the bonus k/sqrt(N+1) shrinks as a state-action is visited more")
    print(f"  {'N (visits)':>12} | " + " | ".join(f"k={k:<5}" for k in K_VALUES))
    print("  " + "-" * 45)
    for n in [0, 1, 5, 10, 25, 50, 100, 500]:
        row = f"  {n:>12} | "
        row += " | ".join(f"{k / np.sqrt(n + 1):>7.3f}" for k in K_VALUES)
        print(row)
    print("    → At N=0 (never visited), bonus is k — pure curiosity.")
    print("    → At N=500, bonus ≈ 0 — agent trusts Q-values entirely.")

    # ── Train all k values ────────────────────────────────────────────────────
    results = {}
    for k in K_VALUES:
        world = WarehouseGridWorld(seed=SEED)
        print(f"\n  Training k={k} ...")
        q_table, visit_counts, rewards, successes = train_exploration_bonus(
            world, ALPHA, GAMMA, NUM_EPISODES, k
        )
        results[k] = {
            "q_table":      q_table,
            "visit_counts": visit_counts,
            "rewards":      rewards,
            "successes":    successes,
        }

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n[2] Results summary")
    print(f"  {'k':>5} | {'Avg reward (all)':>18} | {'Avg reward (last 100)':>22} | "
          f"{'Success (last 100)':>19} | {'States visited':>15}")
    print("  " + "-" * 90)
    for k in K_VALUES:
        r  = results[k]["rewards"]
        s  = results[k]["successes"]
        vc = results[k]["visit_counts"]
        states_visited = int(np.sum(np.any(vc > 0, axis=1)))
        print(
            f"  {k:>5.1f} | {np.mean(r):>18.1f} | {np.mean(r[-100:]):>22.1f} | "
            f"{np.mean(s[-100:])*100:>18.1f}% | {states_visited:>15}"
        )

    # ── How many states did each k explore? ──────────────────────────────────
    print("\n[3] State-space coverage: unique (state, action) pairs visited")
    n_total = 400 * 4
    for k in K_VALUES:
        vc = results[k]["visit_counts"]
        n_visited = int(np.sum(vc > 0))
        print(f"  k={k}: {n_visited} / {n_total} pairs visited  "
              f"({n_visited/n_total*100:.1f}%)  "
              f"avg visits per pair: {vc[vc > 0].mean():.1f}")

    # ── Progression per k ─────────────────────────────────────────────────────
    print("\n[4] Reward progression across training windows")
    windows = [(0, 50), (100, 150), (250, 300), (450, 500)]
    header = f"  {'Window':>12} | " + " | ".join(f"k={k:<6}" for k in K_VALUES)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for start, end in windows:
        row = f"  ep {start+1:>3}–{end:>3}  | "
        row += " | ".join(
            f"{np.mean(results[k]['rewards'][start:end]):>8.1f}" for k in K_VALUES
        )
        print(row)

    # ── Visit count heatmap insight ───────────────────────────────────────────
    print("\n[5] Most-visited actions at the start state")
    world_tmp = WarehouseGridWorld(seed=SEED)
    start_idx = world_tmp.state_to_index(world_tmp.get_state())
    print(f"  Start state: {world_tmp.get_state()}")
    for k in K_VALUES:
        vc_row = results[k]["visit_counts"][start_idx]
        print(f"  k={k}: " + "  ".join(f"{a}={vc_row[i]}" for i, a in enumerate(ACTIONS)))

    # ── Plot ──────────────────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt

        matplotlib.rcParams.update({"font.size": 11})
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(
            f"Checkpoint 7 — Exploration Bonus (k experiments)  "
            f"(seed={SEED}, α={ALPHA}, γ={GAMMA})",
            fontsize=13,
        )

        colors = {0.5: "#e74c3c", 2.0: "#2980b9", 5.0: "#27ae60"}

        # Left: cumulative reward
        for k in K_VALUES:
            r = results[k]["rewards"]
            smoothed = np.convolve(
                r, np.ones(SMOOTHING_WINDOW) / SMOOTHING_WINDOW, mode="valid"
            )
            axes[0].plot(range(SMOOTHING_WINDOW, NUM_EPISODES + 1), smoothed,
                         label=f"k={k}", color=colors[k], linewidth=1.8)
        axes[0].set_xlabel("Episode")
        axes[0].set_ylabel(f"Cumulative reward (rolling avg {SMOOTHING_WINDOW}ep)")
        axes[0].set_title("Cumulative Reward vs Episode")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].axhline(0, color="black", linewidth=0.7, linestyle="--")

        # Middle: success rate
        for k in K_VALUES:
            s = np.array(results[k]["successes"], dtype=float)
            smoothed_s = np.convolve(
                s, np.ones(SMOOTHING_WINDOW) / SMOOTHING_WINDOW, mode="valid"
            )
            axes[1].plot(range(SMOOTHING_WINDOW, NUM_EPISODES + 1), smoothed_s * 100,
                         label=f"k={k}", color=colors[k], linewidth=1.8)
        axes[1].set_xlabel("Episode")
        axes[1].set_ylabel(f"Success rate % (rolling avg {SMOOTHING_WINDOW}ep)")
        axes[1].set_title("Task Success Rate vs Episode")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(-5, 105)

        # Right: visit count distribution (how evenly was the state space explored?)
        # Use step histograms so all three k lines are clearly visible
        max_visits = max(
            results[k]["visit_counts"].max() for k in K_VALUES
        )
        bins = np.linspace(1, max_visits + 1, 40)
        for k in K_VALUES:
            vc = results[k]["visit_counts"].flatten()
            visited = vc[vc > 0]
            axes[2].hist(visited, bins=bins, histtype="step",
                         label=f"k={k}", color=colors[k], linewidth=2.0)
        axes[2].set_xlabel("Times visited")
        axes[2].set_ylabel("Number of (state, action) pairs")
        axes[2].set_title("Visit Count Distribution\n(step = each k is a separate line)")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("checkpoint7_exploration_bonus.png", dpi=120, bbox_inches="tight")
        print("\n[6] Plot saved → checkpoint7_exploration_bonus.png")
        plt.show()

    except ImportError:
        print("\n[6] matplotlib not found — skipping plot")

    print("\n" + "=" * 60)
    print("CHECKPOINT 7 COMPLETE")
    print("Look at the plots and think about:")
    print("  • Section [1]: at N=0, the bonus equals k exactly.")
    print("    Which k would push the agent hardest toward unvisited pairs?")
    print("  • Section [3]: does higher k visit MORE unique states?")
    print("    What does that tell you about how k controls curiosity?")
    print("  • Right plot (visit distribution): is one k more 'uniform'?")
    print("    A uniform distribution means the agent explored everywhere evenly.")
    print("  • Does k=0.5 converge faster because it exploits earlier?")
    print("    Is that the same tradeoff as fast epsilon decay?")
    print("=" * 60)

    return results


if __name__ == "__main__":
    checkpoint_7_exploration_bonus()
