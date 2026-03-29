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



# ── CHECKPOINT 8: Head-to-Head Comparison (Multiple Seeds) ───────────────────
#
# Best configs carried forward:
#   Epsilon-greedy : alpha=0.5, gamma=0.9, medium decay (ep ~300)
#   Exploration bonus: alpha=0.5, gamma=0.9, k=2.0 (balanced from Checkpoint 7)
#
# We run both methods on 3 seeds:
#   42  — the seed we trained on all along
#   99  — the assignment default seed (different layout)
#   123 — a third layout to test generalization
#
# Robustness question: does one method perform consistently across layouts,
# or does it only work well on the seed it was designed around?

SEEDS_TO_TEST = [42, 99, 123]
BEST_K        = 2.0   # balanced from Checkpoint 7

# Epsilon-greedy training reimplemented here to keep this file self-contained
# (Python cannot import files with hyphens in their names directly)

def _epsilon_greedy_action(q_table: np.ndarray, state_idx: int, epsilon: float) -> int:
    if np.random.random() < epsilon:
        return np.random.randint(len(ACTIONS))
    return int(np.argmax(q_table[state_idx]))


def _train_epsilon_greedy(
    world: WarehouseGridWorld,
    alpha: float,
    gamma: float,
    num_episodes: int,
    decay_episode: int = 300,
) -> tuple[list[float], list[bool]]:
    """Epsilon-greedy Q-learning with linear decay schedule."""
    n_states  = world.get_state_space_size()
    n_actions = world.get_action_space_size()
    q_table   = np.zeros((n_states, n_actions))
    eps_start, eps_min = 1.0, 0.05

    rewards, successes = [], []

    for episode in range(num_episodes):
        frac    = min(episode / decay_episode, 1.0)
        epsilon = max(eps_min, eps_start - frac * (eps_start - eps_min))

        state     = world.reset()
        state_idx = world.state_to_index(state)
        total_reward = 0.0
        done = False

        while not done:
            action_idx    = _epsilon_greedy_action(q_table, state_idx, epsilon)
            result        = world.step(ACTIONS[action_idx])
            new_state_idx = world.state_to_index(result.state)

            td_target = result.reward + gamma * np.max(q_table[new_state_idx])
            q_table[state_idx, action_idx] += alpha * (td_target - q_table[state_idx, action_idx])

            total_reward += result.reward
            state_idx     = new_state_idx
            done          = result.done

        rewards.append(total_reward)
        successes.append(world.delivered and world.robot_pos == world.dock_pos)

    return rewards, successes


def _find_convergence(rewards: list[float], threshold: float = -20.0, window: int = 30) -> int:
    for i in range(window, len(rewards)):
        if np.mean(rewards[i - window:i]) >= threshold:
            return i
    return -1


def checkpoint_8_head_to_head():
    """
    Compare both methods across 3 seeds.
    Metrics: avg cumulative reward, success rate, convergence episode.
    """
    print("\n" + "=" * 60)
    print("CHECKPOINT 8 — Head-to-Head Comparison (Multiple Seeds)")
    print(f"  Seeds={SEEDS_TO_TEST}")
    print(f"  Epsilon-greedy: α={ALPHA}, γ={GAMMA}, medium decay")
    print(f"  Exploration bonus: α={ALPHA}, γ={GAMMA}, k={BEST_K}")
    print("=" * 60)

    eg_results  = {}   # epsilon-greedy per seed
    bon_results = {}   # exploration bonus per seed

    for seed in SEEDS_TO_TEST:
        print(f"\n  Seed {seed} ...")
        world = WarehouseGridWorld(seed=seed)
        print(f"    Layout: Pickup={world.pickup_pos}  "
              f"Packing={world.packing_pos}  Dock={world.dock_pos}")

        # Epsilon-greedy
        np.random.seed(seed)
        eg_r, eg_s = _train_epsilon_greedy(world, ALPHA, GAMMA, NUM_EPISODES)
        eg_results[seed] = {"rewards": eg_r, "successes": eg_s}

        # Exploration bonus
        world2 = WarehouseGridWorld(seed=seed)
        _, _, bon_r, bon_s = train_exploration_bonus(world2, ALPHA, GAMMA, NUM_EPISODES, BEST_K)
        bon_results[seed] = {"rewards": bon_r, "successes": bon_s}

    # ── Per-seed comparison table ─────────────────────────────────────────────
    print("\n[A] Per-seed results")
    print(f"\n  {'Method':<22} | {'Seed':>5} | {'Avg reward':>11} | "
          f"{'Last-100 reward':>16} | {'Success rate':>13} | {'Converges ~ep':>14}")
    print("  " + "-" * 95)

    for seed in SEEDS_TO_TEST:
        for label, res in [("Epsilon-greedy", eg_results), ("Exploration bonus", bon_results)]:
            r  = res[seed]["rewards"]
            s  = res[seed]["successes"]
            cv = _find_convergence(r)
            print(
                f"  {label:<22} | {seed:>5} | {np.mean(r):>11.1f} | "
                f"{np.mean(r[-100:]):>16.1f} | {np.mean(s[-100:])*100:>12.1f}% | "
                f"{'~ep ' + str(cv) if cv != -1 else 'never':>14}"
            )
        print("  " + "-" * 95)

    # ── Aggregate across seeds ────────────────────────────────────────────────
    print("\n[B] Aggregated across all seeds (robustness summary)")
    print(f"  {'Metric':<35} {'Epsilon-greedy':>16} {'Exploration bonus':>18}")
    print("  " + "-" * 70)

    eg_all_r  = [eg_results[s]["rewards"]   for s in SEEDS_TO_TEST]
    eg_all_s  = [eg_results[s]["successes"] for s in SEEDS_TO_TEST]
    bon_all_r = [bon_results[s]["rewards"]  for s in SEEDS_TO_TEST]
    bon_all_s = [bon_results[s]["successes"] for s in SEEDS_TO_TEST]

    eg_last100_r  = [np.mean(r[-100:]) for r in eg_all_r]
    bon_last100_r = [np.mean(r[-100:]) for r in bon_all_r]
    eg_last100_s  = [np.mean(s[-100:]) * 100 for s in eg_all_s]
    bon_last100_s = [np.mean(s[-100:]) * 100 for s in bon_all_s]

    eg_convs  = [_find_convergence(r) for r in eg_all_r]
    bon_convs = [_find_convergence(r) for r in bon_all_r]

    rows = [
        ("Mean last-100 reward",
         f"{np.mean(eg_last100_r):>14.1f}",   f"{np.mean(bon_last100_r):>16.1f}"),
        ("Std last-100 reward (↓ better)",
         f"{np.std(eg_last100_r):>14.1f}",    f"{np.std(bon_last100_r):>16.1f}"),
        ("Mean success rate %",
         f"{np.mean(eg_last100_s):>13.1f}%",  f"{np.mean(bon_last100_s):>15.1f}%"),
        ("Std success rate (↓ better)",
         f"{np.std(eg_last100_s):>13.1f}%",   f"{np.std(bon_last100_s):>15.1f}%"),
        ("Mean convergence episode",
         f"{np.mean([c for c in eg_convs if c!=-1]):>14.0f}",
         f"{np.mean([c for c in bon_convs if c!=-1]):>16.0f}"),
    ]
    for label, eg_val, bon_val in rows:
        print(f"  {label:<35} {eg_val}  {bon_val}")

    print("\n  Std = standard deviation across seeds.")
    print("  Lower std = more robust (performs consistently regardless of layout).")

    # ── Plot ──────────────────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt

        matplotlib.rcParams.update({"font.size": 10})
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(
            f"Checkpoint 8 — Head-to-Head: Epsilon-Greedy vs Exploration Bonus\n"
            f"α={ALPHA}, γ={GAMMA} | EG: medium decay | Bonus: k={BEST_K}",
            fontsize=13,
        )

        eg_color  = "#2980b9"
        bon_color = "#e74c3c"

        for col, seed in enumerate(SEEDS_TO_TEST):
            ax_r = axes[0, col]   # reward row
            ax_s = axes[1, col]   # success row

            for rewards, label, color in [
                (eg_results[seed]["rewards"],  "Epsilon-greedy",  eg_color),
                (bon_results[seed]["rewards"], "Bonus k=2.0",     bon_color),
            ]:
                smoothed = np.convolve(
                    rewards, np.ones(SMOOTHING_WINDOW) / SMOOTHING_WINDOW, mode="valid"
                )
                x = range(SMOOTHING_WINDOW, NUM_EPISODES + 1)
                ax_r.plot(x, smoothed, label=label, color=color, linewidth=1.8)

            ax_r.set_title(f"Seed {seed} — Cumulative Reward")
            ax_r.set_xlabel("Episode")
            ax_r.set_ylabel("Reward")
            ax_r.legend(fontsize=9)
            ax_r.grid(True, alpha=0.3)
            ax_r.axhline(0, color="black", linewidth=0.6, linestyle="--")

            for successes, label, color in [
                (eg_results[seed]["successes"],  "Epsilon-greedy",  eg_color),
                (bon_results[seed]["successes"], "Bonus k=2.0",     bon_color),
            ]:
                s = np.array(successes, dtype=float)
                smoothed_s = np.convolve(
                    s, np.ones(SMOOTHING_WINDOW) / SMOOTHING_WINDOW, mode="valid"
                )
                ax_s.plot(range(SMOOTHING_WINDOW, NUM_EPISODES + 1), smoothed_s * 100,
                          label=label, color=color, linewidth=1.8)

            ax_s.set_title(f"Seed {seed} — Success Rate")
            ax_s.set_xlabel("Episode")
            ax_s.set_ylabel("Success %")
            ax_s.legend(fontsize=9)
            ax_s.grid(True, alpha=0.3)
            ax_s.set_ylim(-5, 105)

        plt.tight_layout()
        plt.savefig("checkpoint8_head_to_head.png", dpi=120, bbox_inches="tight")
        print("\n[C] Plot saved → checkpoint8_head_to_head.png")
        plt.show()

    except ImportError:
        print("\n[C] matplotlib not found — skipping plot")

    print("\n" + "=" * 60)
    print("CHECKPOINT 8 COMPLETE")
    print("Look at section [B] and the plots — think about:")
    print("  • Which method has LOWER std across seeds?")
    print("    Lower std = more robust = works regardless of layout.")
    print("  • Do both methods fail on any seed, or do both always converge?")
    print("  • Does the gap between them grow or shrink on harder layouts?")
    print("    (Seeds 99 and 123 have different pickup/packing/dock positions.)")
    print("  • Which would you trust on a layout you have never seen before?")
    print("=" * 60)



# ── CHECKPOINT 9: Visualization ───────────────────────────────────────────────
#
# Part 1 — Combined reward comparison plot (both methods, seed=42)
# Part 2 — Pygame real-time demo: watch each trained agent navigate live
#
# The pygame window shows the robot moving step by step.
# Press any key or wait for the episode to finish, then the window closes.

def plot_combined_comparison(
    eg_rewards: list[float],
    bon_rewards: list[float],
    eg_successes: list[bool],
    bon_successes: list[bool],
    seed: int,
) -> None:
    """Clean side-by-side comparison plot for the report."""
    try:
        import matplotlib.pyplot as plt

        matplotlib.rcParams.update({"font.size": 11})
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(
            f"Epsilon-Greedy vs Exploration Bonus — seed={seed}\n"
            f"α={ALPHA}, γ={GAMMA} | EG: medium decay | Bonus: k={BEST_K}",
            fontsize=12,
        )

        eg_color  = "#2980b9"
        bon_color = "#e74c3c"

        for rewards, label, color in [
            (eg_rewards,  f"Epsilon-greedy (α={ALPHA}, γ={GAMMA})", eg_color),
            (bon_rewards, f"Exploration bonus (k={BEST_K})",         bon_color),
        ]:
            smoothed = np.convolve(
                rewards, np.ones(SMOOTHING_WINDOW) / SMOOTHING_WINDOW, mode="valid"
            )
            ax1.plot(range(SMOOTHING_WINDOW, NUM_EPISODES + 1), smoothed,
                     label=label, color=color, linewidth=2.0)

        ax1.set_xlabel("Episode")
        ax1.set_ylabel(f"Cumulative reward (rolling avg {SMOOTHING_WINDOW}ep)")
        ax1.set_title("Cumulative Reward vs Episode")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axhline(0, color="black", linewidth=0.7, linestyle="--")

        for successes, label, color in [
            (eg_successes,  f"Epsilon-greedy", eg_color),
            (bon_successes, f"Exploration bonus (k={BEST_K})", bon_color),
        ]:
            s = np.array(successes, dtype=float)
            smoothed_s = np.convolve(
                s, np.ones(SMOOTHING_WINDOW) / SMOOTHING_WINDOW, mode="valid"
            )
            ax2.plot(range(SMOOTHING_WINDOW, NUM_EPISODES + 1), smoothed_s * 100,
                     label=label, color=color, linewidth=2.0)

        ax2.set_xlabel("Episode")
        ax2.set_ylabel(f"Success rate % (rolling avg {SMOOTHING_WINDOW}ep)")
        ax2.set_title("Task Success Rate vs Episode")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(-5, 105)

        plt.tight_layout()
        plt.savefig("checkpoint9_combined_comparison.png", dpi=120, bbox_inches="tight")
        print("  Plot saved → checkpoint9_combined_comparison.png")
        plt.show()

    except ImportError:
        print("  matplotlib not found — skipping plot")


def run_pygame_demo(
    world: WarehouseGridWorld,
    q_table: np.ndarray,
    title: str,
    step_delay_ms: int = 250,
) -> None:
    """
    Animate one evaluation episode in a pygame window.
    The agent follows its learned policy (epsilon=0).
    Close the window or wait for the episode to end.
    """
    try:
        import pygame
        from warehouse_gridworld_domain_random import (
            setup_pygame, draw_grid,
            CELL_SIZE, GRID_SIZE,
        )
    except ImportError:
        print("  pygame not available — skipping live demo")
        return

    screen, clock, font, small_font = setup_pygame()
    pygame.display.set_caption(title)

    state     = world.reset()
    state_idx = world.state_to_index(state)
    done      = False
    running   = True

    draw_grid(world, screen, font, small_font)
    pygame.display.flip()
    pygame.time.wait(600)   # brief pause so you can see the start position

    while running and not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                running = False

        if running:
            action_idx    = int(np.argmax(q_table[state_idx]))
            result        = world.step(ACTIONS[action_idx])
            state_idx     = world.state_to_index(result.state)
            done          = result.done

            draw_grid(world, screen, font, small_font)
            pygame.display.flip()
            clock.tick(1000 // step_delay_ms)

    # Hold the final frame for 2 seconds before closing
    if running:
        pygame.time.wait(2000)
    pygame.quit()

    status = "SUCCESS" if (world.delivered and world.robot_pos == world.dock_pos) else "TIMEOUT"
    print(f"  Demo finished — {status} in {world.steps} steps, score={world.score:.0f}")


def checkpoint_9_visualization():
    """
    Train both methods on seed=42, then:
      1. Show combined reward/success comparison plot
      2. Run live pygame demo for epsilon-greedy agent
      3. Run live pygame demo for exploration bonus agent
    """
    print("\n" + "=" * 60)
    print("CHECKPOINT 9 — Visualization")
    print("=" * 60)

    # ── Train both methods (seed=42, best configs) ────────────────────────────
    print(f"\nTraining Epsilon-Greedy (α={ALPHA}, γ={GAMMA}, medium decay) ...")
    world_eg = WarehouseGridWorld(seed=SEED)
    np.random.seed(SEED)
    eg_r, eg_s = _train_epsilon_greedy(world_eg, ALPHA, GAMMA, NUM_EPISODES)

    # Rebuild Q-table for evaluation (train_epsilon_greedy returns rewards/successes only)
    # Re-train to get the Q-table as well
    from warehouse_gridworld_domain_random import WarehouseGridWorld as WGW

    def _train_eg_with_qtable(seed):
        w = WGW(seed=seed)
        n_s = w.get_state_space_size()
        n_a = w.get_action_space_size()
        q   = np.zeros((n_s, n_a))
        eps_start, eps_min, decay_ep = 1.0, 0.05, 300
        for ep in range(NUM_EPISODES):
            frac    = min(ep / decay_ep, 1.0)
            epsilon = max(eps_min, eps_start - frac * (eps_start - eps_min))
            state   = w.reset()
            s_idx   = w.state_to_index(state)
            done    = False
            while not done:
                a_idx  = _epsilon_greedy_action(q, s_idx, epsilon)
                res    = w.step(ACTIONS[a_idx])
                ns_idx = w.state_to_index(res.state)
                q[s_idx, a_idx] += ALPHA * (
                    res.reward + GAMMA * np.max(q[ns_idx]) - q[s_idx, a_idx]
                )
                s_idx = ns_idx
                done  = res.done
        return w, q

    print("  Re-training to capture Q-table ...")
    np.random.seed(SEED)
    world_eg, q_eg = _train_eg_with_qtable(SEED)

    print(f"\nTraining Exploration Bonus (α={ALPHA}, γ={GAMMA}, k={BEST_K}) ...")
    world_bon = WarehouseGridWorld(seed=SEED)
    q_bon, _, bon_r, bon_s = train_exploration_bonus(
        world_bon, ALPHA, GAMMA, NUM_EPISODES, BEST_K
    )

    # ── Combined comparison plot ──────────────────────────────────────────────
    print("\n[1] Generating combined comparison plot ...")
    plot_combined_comparison(eg_r, bon_r, eg_s, bon_s, SEED)

    # ── Pygame demos ─────────────────────────────────────────────────────────
    print("\n[2] Launching Epsilon-Greedy demo (watch the robot navigate) ...")
    print("    Close the window or press any key to continue.")
    run_pygame_demo(
        WarehouseGridWorld(seed=SEED), q_eg,
        title=f"Epsilon-Greedy Agent (α={ALPHA}, γ={GAMMA})",
    )

    print("\n[3] Launching Exploration Bonus demo ...")
    print("    Close the window or press any key to continue.")
    run_pygame_demo(
        WarehouseGridWorld(seed=SEED), q_bon,
        title=f"Exploration Bonus Agent (k={BEST_K}, α={ALPHA}, γ={GAMMA})",
    )

    print("\n" + "=" * 60)
    print("CHECKPOINT 9 COMPLETE — All checkpoints finished.")
    print("\nWhile watching the two demos, notice:")
    print("  • Do both agents take the same path or different routes?")
    print("  • Does either agent detour around congestion zones (red cells)?")
    print("  • Which agent moves more 'confidently' — fewer hesitations?")
    print("    (Hesitation = revisiting the same cell, which means the Q-table")
    print("     has conflicting values for neighbouring actions.)")
    print("=" * 60)


if __name__ == "__main__":
    checkpoint_7_exploration_bonus()
    checkpoint_8_head_to_head()
    checkpoint_9_visualization()
