# -*- coding: utf-8 -*-
"""
COSC-4117EL: Assignment 2
Q-Learning with Epsilon-Greedy Exploration

Group 8: Walter Amador and Babatunde Omoyele
Note: This file was developed with the assistance of Claude Code (claude.ai/code).
All AI-generated code is acknowledged per assignment guidelines.

CHECKPOINT 1: State Space & Q-Table Initialization
---------------------------------------------------
Goal: Understand what the agent "knows" before any learning happens.
- What is a state?
- How many states exist and why?
- What does Q(s,a) = 0 mean?
"""

import numpy as np
from warehouse_gridworld_domain_random import (
    WarehouseGridWorld,
    ACTIONS,
    EMPTY, SHELF, CONGESTION, PICKUP, PACKING, DOCK,
)

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42


# ── CHECKPOINT 1: Explore the state space before any training ────────────────

def checkpoint_1_state_space():
    """
    Inspect the environment and Q-table before a single training step.
    Run this first to build intuition about the problem representation.
    """
    world = WarehouseGridWorld(seed=SEED)

    # ── 1. Grid layout ────────────────────────────────────────────────────────
    print("=" * 60)
    print("CHECKPOINT 1 — State Space & Q-Table Initialization")
    print("=" * 60)

    print("\n[1] Grid layout (seed=42)")
    print("    Legend: . empty  # shelf  ! congestion  P pickup  D packing  C dock")
    world.display()

    # ── 2. State space dimensions ────────────────────────────────────────────
    n_states  = world.get_state_space_size()   # 10*10*2*2 = 400
    n_actions = world.get_action_space_size()  # 4

    print("\n[2] State space breakdown")
    print(f"    Grid positions  : {world.size} rows × {world.size} cols = {world.size**2} cells")
    print(f"    has_package     : 2 values  (0 = no package, 1 = carrying)")
    print(f"    delivered       : 2 values  (0 = not yet,    1 = delivered)")
    print(f"    Total states    : {world.size}² × 2 × 2 = {n_states}")
    print(f"    Actions         : {n_actions}  ({ACTIONS})")
    print(f"    Q-table shape   : ({n_states}, {n_actions})  →  {n_states * n_actions} entries")

    # ── 3. Why task-progress flags matter ────────────────────────────────────
    print("\n[3] Why (row, col) alone is not enough as a state")
    print("    Same position (row, col) — three different situations:")
    print("    (r, c, has_package=0, delivered=0) → agent must go find package")
    print("    (r, c, has_package=1, delivered=0) → agent must go deliver package")
    print("    (r, c, has_package=0, delivered=1) → agent must return to dock")
    print("    Without these flags the agent cannot distinguish what to do next.")

    # ── 4. State indexing ────────────────────────────────────────────────────
    print("\n[4] State → linear index examples")
    sample_states = [
        world.get_state(),                    # initial state at dock
        (world.pickup_pos[0],  world.pickup_pos[1],  0, 0),  # at pickup, no package
        (world.pickup_pos[0],  world.pickup_pos[1],  1, 0),  # at pickup, with package
        (world.packing_pos[0], world.packing_pos[1], 1, 0),  # at packing, carrying
        (world.packing_pos[0], world.packing_pos[1], 0, 1),  # at packing, delivered
        (world.dock_pos[0],    world.dock_pos[1],    0, 1),  # back at dock ✓
    ]
    labels = [
        "start (at dock, no package)",
        "at pickup — before pickup",
        "at pickup — after pickup",
        "at packing — carrying package",
        "at packing — after delivery",
        "back at dock — DONE",
    ]
    for state, label in zip(sample_states, labels):
        idx = world.state_to_index(state)
        print(f"    {str(state):<30}  index={idx:>3}   ({label})")

    # ── 5. Q-table initialized to zeros ──────────────────────────────────────
    print("\n[5] Q-table initialized to zeros")
    q_table = np.zeros((n_states, n_actions))
    print(f"    Shape : {q_table.shape}")
    print(f"    Sum   : {q_table.sum():.1f}   ← agent has no preference yet")
    print(f"    What Q(s,a)=0 means:")
    print(f"      The agent has NO knowledge of which action is best in any state.")
    print(f"      Every action looks equally good (or equally bad).")
    print(f"      Training will fill these values with experience.")

    # Show the initial row for the starting state
    start_idx = world.state_to_index(world.get_state())
    print(f"\n    Q-table row for start state {world.get_state()} (index {start_idx}):")
    for i, action in enumerate(ACTIONS):
        print(f"      Q({world.get_state()}, '{action}') = {q_table[start_idx, i]:.2f}")

    # ── 6. Valid actions from start ──────────────────────────────────────────
    print("\n[6] Valid actions from start position (walls limit choices)")
    valid = world.valid_actions()
    invalid = [a for a in ACTIONS if a not in valid]
    print(f"    Position      : {world.robot_pos}")
    print(f"    Valid actions : {valid}")
    print(f"    Blocked       : {invalid}  ← hitting these costs -3 and wastes a step")

    print("\n" + "=" * 60)
    print("CHECKPOINT 1 COMPLETE")
    print("Before moving on, think about:")
    print("  • Why would Q(s,a)=0 everywhere cause the agent to wander randomly?")
    print("  • Why do we need delivered=1 as a separate flag from has_package=0?")
    print("  • What would happen if we forgot the task-progress flags?")
    print("=" * 60)



# ── CHECKPOINT 2: Bare-Bones Training Loop ───────────────────────────────────
#
# Goal: Get the agent training with ONE fixed set of hyperparameters.
# No tuning yet — just understand the mechanics.
#
# Fixed hyperparameters for this checkpoint:
#   alpha   = 0.3   (learning rate — how much each new experience overwrites old)
#   gamma   = 0.95  (discount factor — how much the agent values future rewards)
#   epsilon = 1.0   (start fully random, decay toward 0.05 over training)
#
# Q-learning update rule:
#   Q[s, a] += alpha * (reward + gamma * max(Q[s', :]) - Q[s, a])
#              ─────   ────────────────────────────────────────────
#           step size        TD error (how wrong was our old estimate?)

ALPHA   = 0.3
GAMMA   = 0.95
EPSILON_START = 1.0
EPSILON_MIN   = 0.05
NUM_EPISODES  = 500


def epsilon_greedy(q_table: np.ndarray, state_idx: int, epsilon: float) -> int:
    """
    With probability epsilon  → pick a random action (explore).
    With probability 1-epsilon → pick the action with highest Q-value (exploit).

    Returns an action index (0=up, 1=down, 2=left, 3=right).
    """
    if np.random.random() < epsilon:
        return np.random.randint(len(ACTIONS))   # random exploration
    return int(np.argmax(q_table[state_idx]))    # greedy exploitation


def decay_epsilon(episode: int, total_episodes: int) -> float:
    """
    Linear decay: epsilon drops from EPSILON_START to EPSILON_MIN
    evenly across all training episodes.
    """
    fraction = episode / total_episodes
    return max(EPSILON_MIN, EPSILON_START - fraction * (EPSILON_START - EPSILON_MIN))


def train_epsilon_greedy(
    world: WarehouseGridWorld,
    alpha: float,
    gamma: float,
    num_episodes: int,
    verbose: bool = False,
) -> tuple[np.ndarray, list[float], list[bool]]:
    """
    Train a Q-learning agent with epsilon-greedy exploration.

    Returns
    -------
    q_table          : learned Q-values, shape (n_states, n_actions)
    episode_rewards  : cumulative reward per episode
    episode_success  : whether the agent completed the full task per episode
    """
    n_states  = world.get_state_space_size()
    n_actions = world.get_action_space_size()
    q_table   = np.zeros((n_states, n_actions))

    episode_rewards = []
    episode_success = []
    epsilon = EPSILON_START

    for episode in range(num_episodes):
        state      = world.reset()
        state_idx  = world.state_to_index(state)
        total_reward = 0.0
        done         = False

        while not done:
            # 1. Choose action via epsilon-greedy
            action_idx = epsilon_greedy(q_table, state_idx, epsilon)
            action     = ACTIONS[action_idx]

            # 2. Take the step, observe outcome
            result     = world.step(action)
            new_state  = result.state
            reward     = result.reward
            done       = result.done

            new_state_idx = world.state_to_index(new_state)

            # 3. Q-learning update
            #    TD target  = reward + gamma * max future Q  (what we now think is true)
            #    TD error   = TD target - Q[s,a]             (how wrong we were)
            #    New Q[s,a] = old Q[s,a] + alpha * TD error
            td_target     = reward + gamma * np.max(q_table[new_state_idx])
            td_error      = td_target - q_table[state_idx, action_idx]
            q_table[state_idx, action_idx] += alpha * td_error

            total_reward += reward
            state_idx     = new_state_idx

        # Decay epsilon after each episode
        epsilon = decay_epsilon(episode + 1, num_episodes)

        episode_rewards.append(total_reward)
        # Success = agent completed full task (returned to dock after delivery)
        episode_success.append(world.delivered and (world.robot_pos == world.dock_pos))

        if verbose and (episode + 1) % 50 == 0:
            recent_rewards  = episode_rewards[-50:]
            recent_successes = episode_success[-50:]
            print(
                f"  Episode {episode+1:>4}/{num_episodes} | "
                f"ε={epsilon:.3f} | "
                f"Avg reward (last 50): {np.mean(recent_rewards):>7.1f} | "
                f"Success rate: {np.mean(recent_successes)*100:>5.1f}%"
            )

    return q_table, episode_rewards, episode_success


def checkpoint_2_training_loop():
    """
    Train with fixed hyperparameters and examine what the agent learns.
    Focus on understanding the Q-update, not on getting good numbers yet.
    """
    print("\n" + "=" * 60)
    print("CHECKPOINT 2 — Bare-Bones Training Loop")
    print(f"  alpha={ALPHA}  gamma={GAMMA}  episodes={NUM_EPISODES}")
    print("=" * 60)

    world = WarehouseGridWorld(seed=SEED)

    print(f"\nLayout: Pickup={world.pickup_pos}  Packing={world.packing_pos}  Dock={world.dock_pos}")
    print(f"\nTraining ({NUM_EPISODES} episodes) ...")

    q_table, rewards, successes = train_epsilon_greedy(
        world, ALPHA, GAMMA, NUM_EPISODES, verbose=True
    )

    # ── What did the agent learn? ─────────────────────────────────────────────
    print("\n[A] Overall training results")
    print(f"    Total successes      : {sum(successes)} / {NUM_EPISODES}")
    print(f"    Success rate         : {np.mean(successes)*100:.1f}%")
    print(f"    Avg reward (all)     : {np.mean(rewards):.1f}")
    print(f"    Avg reward (last 100): {np.mean(rewards[-100:]):.1f}")
    print(f"    Best episode reward  : {max(rewards):.1f}")

    # ── Q-table snapshot at start state ──────────────────────────────────────
    world.reset()
    start_idx = world.state_to_index(world.get_state())
    print(f"\n[B] Q-table at START state {world.get_state()} after training")
    print(f"    (before training every value was 0.0)")
    for i, action in enumerate(ACTIONS):
        bar = "█" * max(0, int(q_table[start_idx, i] / 2))
        print(f"      Q(start, '{action}') = {q_table[start_idx, i]:>8.2f}  {bar}")
    best_action = ACTIONS[int(np.argmax(q_table[start_idx]))]
    print(f"    → Learned best first move: '{best_action}'")

    # ── Q-table at pickup state ───────────────────────────────────────────────
    pickup_state = (world.pickup_pos[0], world.pickup_pos[1], 0, 0)
    pickup_idx   = world.state_to_index(pickup_state)
    print(f"\n[C] Q-table at PICKUP state {pickup_state}")
    for i, action in enumerate(ACTIONS):
        print(f"      Q(pickup, '{action}') = {q_table[pickup_idx, i]:>8.2f}")

    # ── Early vs late episodes ────────────────────────────────────────────────
    print("\n[D] Learning progression")
    windows = [(0, 50), (100, 150), (250, 300), (450, 500)]
    for start, end in windows:
        window_r = rewards[start:end]
        window_s = successes[start:end]
        print(
            f"    Episodes {start+1:>3}–{end:>3} | "
            f"Avg reward: {np.mean(window_r):>7.1f} | "
            f"Success: {np.mean(window_s)*100:>5.1f}%"
        )

    print("\n" + "=" * 60)
    print("CHECKPOINT 2 COMPLETE")
    print("Before moving on, think about:")
    print("  • Why are early episode rewards low/negative?")
    print("  • Why does the reward increase over time?")
    print("  • What does it mean if the Q-value for 'up' at the start is high?")
    print("  • Is the success rate still climbing, or has it leveled off?")
    print("=" * 60)

    return q_table, rewards, successes



# ── CHECKPOINT 3: Alpha Experiments (Learning Rate) ──────────────────────────
#
# Question: How does the learning rate change what the agent learns?
#
# Alpha controls how aggressively the agent updates its Q-values each step:
#   Q[s,a] += alpha * TD_error
#
#   Low alpha  (0.1) → small steps, learns slowly, more stable but slower convergence
#   Mid alpha  (0.3) → baseline from Checkpoint 2
#   High alpha (0.5) → larger steps, learns faster but can overshoot/oscillate
#   Very high  (0.7) → very aggressive updates, can become unstable
#
# Everything else is held FIXED so alpha is the only variable.
# Fixed: gamma=0.95, epsilon decay=standard, episodes=500, seed=42

ALPHAS_TO_TEST = [0.1, 0.3, 0.5, 0.7]
SMOOTHING_WINDOW = 20   # episodes to average for smoother plots


def find_convergence_episode(rewards: list[float], threshold: float = -20.0, window: int = 30) -> int:
    """
    Returns the first episode where the rolling average reward stays above
    a threshold, as a rough convergence point.
    Returns -1 if never converged.
    """
    for i in range(window, len(rewards)):
        if np.mean(rewards[i - window:i]) >= threshold:
            return i
    return -1


def checkpoint_3_alpha_experiments():
    """
    Train with 4 different alphas, everything else fixed.
    Compare: convergence speed, final reward, stability of learning curve.
    """
    print("\n" + "=" * 60)
    print("CHECKPOINT 3 — Alpha (Learning Rate) Experiments")
    print(f"  alphas={ALPHAS_TO_TEST}  gamma={GAMMA}  episodes={NUM_EPISODES}")
    print("=" * 60)

    results = {}

    for alpha in ALPHAS_TO_TEST:
        world = WarehouseGridWorld(seed=SEED)
        print(f"\n  Training alpha={alpha} ...")
        q_table, rewards, successes = train_epsilon_greedy(
            world, alpha=alpha, gamma=GAMMA,
            num_episodes=NUM_EPISODES, verbose=False,
        )
        results[alpha] = {
            "q_table":   q_table,
            "rewards":   rewards,
            "successes": successes,
        }

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n[A] Results summary")
    print(f"  {'Alpha':>6} | {'Avg reward (all)':>18} | {'Avg reward (last 100)':>22} | "
          f"{'Success rate (last 100)':>24} | {'Converges ~ep':>14}")
    print("  " + "-" * 95)
    for alpha in ALPHAS_TO_TEST:
        r  = results[alpha]["rewards"]
        s  = results[alpha]["successes"]
        cv = find_convergence_episode(r)
        print(
            f"  {alpha:>6.1f} | {np.mean(r):>18.1f} | {np.mean(r[-100:]):>22.1f} | "
            f"{np.mean(s[-100:])*100:>23.1f}% | "
            f"{'~ep ' + str(cv) if cv != -1 else 'never':>14}"
        )

    # ── Progression per alpha ─────────────────────────────────────────────────
    print("\n[B] Reward progression across training windows")
    windows = [(0, 50), (100, 150), (250, 300), (450, 500)]
    header = f"  {'Window':>12} | " + " | ".join(f"α={a:<4}" for a in ALPHAS_TO_TEST)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for start, end in windows:
        row = f"  ep {start+1:>3}–{end:>3}  | "
        row += " | ".join(
            f"{np.mean(results[a]['rewards'][start:end]):>7.1f}" for a in ALPHAS_TO_TEST
        )
        print(row)

    # ── Plot ──────────────────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt
        import matplotlib

        matplotlib.rcParams.update({"font.size": 11})
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"Checkpoint 3 — Alpha Experiments  (seed={SEED}, γ={GAMMA})", fontsize=13)

        colors = ["#e74c3c", "#2980b9", "#27ae60", "#f39c12"]

        for alpha, color in zip(ALPHAS_TO_TEST, colors):
            r = results[alpha]["rewards"]
            # Rolling average for smoother curves
            smoothed = np.convolve(r, np.ones(SMOOTHING_WINDOW) / SMOOTHING_WINDOW, mode="valid")
            episodes = range(SMOOTHING_WINDOW, NUM_EPISODES + 1)

            ax1.plot(episodes, smoothed, label=f"α={alpha}", color=color, linewidth=1.8)

        ax1.set_xlabel("Episode")
        ax1.set_ylabel(f"Cumulative reward (rolling avg {SMOOTHING_WINDOW}ep)")
        ax1.set_title("Cumulative Reward vs Episode")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axhline(0, color="black", linewidth=0.7, linestyle="--")

        # Success rate over training (rolling window)
        for alpha, color in zip(ALPHAS_TO_TEST, colors):
            s = np.array(results[alpha]["successes"], dtype=float)
            smoothed_s = np.convolve(s, np.ones(SMOOTHING_WINDOW) / SMOOTHING_WINDOW, mode="valid")
            episodes = range(SMOOTHING_WINDOW, NUM_EPISODES + 1)
            ax2.plot(episodes, smoothed_s * 100, label=f"α={alpha}", color=color, linewidth=1.8)

        ax2.set_xlabel("Episode")
        ax2.set_ylabel(f"Success rate % (rolling avg {SMOOTHING_WINDOW}ep)")
        ax2.set_title("Task Success Rate vs Episode")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(-5, 105)

        plt.tight_layout()
        plt.savefig("checkpoint3_alpha_experiments.png", dpi=120, bbox_inches="tight")
        print("\n[C] Plot saved → checkpoint3_alpha_experiments.png")
        plt.show()

    except ImportError:
        print("\n[C] matplotlib not found — skipping plot (pip install matplotlib)")

    print("\n" + "=" * 60)
    print("CHECKPOINT 3 COMPLETE")
    print("Look at the plots and think about:")
    print("  • Which alpha converges fastest (reaches high success rate earliest)?")
    print("  • Which alpha has the smoothest reward curve vs the noisiest?")
    print("  • Does high alpha (0.7) ever hurt the agent — does the curve dip?")
    print("  • At α=0.1, is the agent still improving at episode 500,")
    print("    or has it leveled off?  What does that tell you about step size?")
    print("=" * 60)

    return results



# ── CHECKPOINT 4: Gamma Experiments (Discount Factor) ────────────────────────
#
# Question: How much should the agent care about future rewards?
#
# Gamma is the discount factor applied to future Q-values in the update:
#   TD target = reward + gamma * max(Q[s', :])
#
# Intuitively:
#   gamma=0.7  → future rewards are worth 70% of what immediate ones are
#               after 10 steps, a reward is worth only 0.7^10 ≈ 2.8% of its value
#   gamma=0.9  → moderate foresight, a common default
#   gamma=0.99 → the agent nearly fully values rewards 20+ steps away
#
# WHY THIS MATTERS FOR THIS TASK:
#   The biggest rewards are DELAYED:
#     pickup  (+25) happens after navigating to the pickup station
#     delivery(+40) happens after navigating to the packing station
#     dock   (+100) happens only after BOTH are done
#   A short-sighted agent (low gamma) may not "see" the dock reward at all
#   from states that are many steps away — so it won't learn to return.
#
# Fixed: alpha=0.5 (smooth + fast from Checkpoint 3), same epsilon decay, seed=42

BEST_ALPHA    = 0.5       # best from Checkpoint 3
GAMMAS_TO_TEST = [0.7, 0.9, 0.99]


def checkpoint_4_gamma_experiments():
    """
    Train with 3 different gammas, everything else fixed.
    Compare: whether the agent learns the full task, how it values the dock return.
    """
    print("\n" + "=" * 60)
    print("CHECKPOINT 4 — Gamma (Discount Factor) Experiments")
    print(f"  gammas={GAMMAS_TO_TEST}  alpha={BEST_ALPHA}  episodes={NUM_EPISODES}")
    print("=" * 60)

    results = {}

    for gamma in GAMMAS_TO_TEST:
        world = WarehouseGridWorld(seed=SEED)
        print(f"\n  Training gamma={gamma} ...")
        q_table, rewards, successes = train_epsilon_greedy(
            world, alpha=BEST_ALPHA, gamma=gamma,
            num_episodes=NUM_EPISODES, verbose=False,
        )
        results[gamma] = {
            "q_table":   q_table,
            "rewards":   rewards,
            "successes": successes,
            "world":     world,
        }

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n[A] Results summary")
    print(f"  {'Gamma':>6} | {'Avg reward (all)':>18} | {'Avg reward (last 100)':>22} | "
          f"{'Success rate (last 100)':>24} | {'Converges ~ep':>14}")
    print("  " + "-" * 95)
    for gamma in GAMMAS_TO_TEST:
        r  = results[gamma]["rewards"]
        s  = results[gamma]["successes"]
        cv = find_convergence_episode(r)
        print(
            f"  {gamma:>6.2f} | {np.mean(r):>18.1f} | {np.mean(r[-100:]):>22.1f} | "
            f"{np.mean(s[-100:])*100:>23.1f}% | "
            f"{'~ep ' + str(cv) if cv != -1 else 'never':>14}"
        )

    # ── Key insight: how far does the agent "see" the dock reward? ────────────
    print("\n[B] How much is the dock reward (+100) worth from N steps away?")
    print(f"    (discounted value = gamma^N * 100)")
    print(f"  {'Steps away':>12} | " + " | ".join(f"γ={g}" for g in GAMMAS_TO_TEST))
    print("  " + "-" * 50)
    for steps in [5, 10, 15, 20, 30]:
        row = f"  {steps:>10} s | "
        row += " | ".join(f"{(g ** steps) * 100:>7.2f}" for g in GAMMAS_TO_TEST)
        print(row)
    print("    → A short-sighted agent can't 'see' the dock reward from far away.")

    # ── Q-value at dock state for each gamma ─────────────────────────────────
    print("\n[C] Q-values at the dock state after delivery")
    print("    State = (dock_row, dock_col, has_package=0, delivered=1)")
    print("    This state immediately precedes the terminal reward (+100).")
    print("    Higher Q-values here = agent more strongly attracted to dock.\n")
    for gamma in GAMMAS_TO_TEST:
        world = results[gamma]["world"]
        dock_r, dock_c = world.dock_pos
        dock_state = (dock_r, dock_c, 0, 1)
        dock_idx   = world.state_to_index(dock_state)
        q_row      = results[gamma]["q_table"][dock_idx]
        best_a     = ACTIONS[int(np.argmax(q_row))]
        print(f"    γ={gamma}: Q-values={np.round(q_row, 1)}  best='{best_a}'")

    # ── Progression per gamma ─────────────────────────────────────────────────
    print("\n[D] Reward progression across training windows")
    windows = [(0, 50), (100, 150), (250, 300), (450, 500)]
    header = f"  {'Window':>12} | " + " | ".join(f"γ={g:<5}" for g in GAMMAS_TO_TEST)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for start, end in windows:
        row = f"  ep {start+1:>3}–{end:>3}  | "
        row += " | ".join(
            f"{np.mean(results[g]['rewards'][start:end]):>8.1f}" for g in GAMMAS_TO_TEST
        )
        print(row)

    # ── Plot ──────────────────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt
        import matplotlib

        matplotlib.rcParams.update({"font.size": 11})
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(
            f"Checkpoint 4 — Gamma Experiments  (seed={SEED}, α={BEST_ALPHA})",
            fontsize=13,
        )

        colors = ["#e74c3c", "#2980b9", "#27ae60"]

        for gamma, color in zip(GAMMAS_TO_TEST, colors):
            r = results[gamma]["rewards"]
            smoothed = np.convolve(r, np.ones(SMOOTHING_WINDOW) / SMOOTHING_WINDOW, mode="valid")
            episodes = range(SMOOTHING_WINDOW, NUM_EPISODES + 1)
            ax1.plot(episodes, smoothed, label=f"γ={gamma}", color=color, linewidth=1.8)

        ax1.set_xlabel("Episode")
        ax1.set_ylabel(f"Cumulative reward (rolling avg {SMOOTHING_WINDOW}ep)")
        ax1.set_title("Cumulative Reward vs Episode")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axhline(0, color="black", linewidth=0.7, linestyle="--")

        for gamma, color in zip(GAMMAS_TO_TEST, colors):
            s = np.array(results[gamma]["successes"], dtype=float)
            smoothed_s = np.convolve(s, np.ones(SMOOTHING_WINDOW) / SMOOTHING_WINDOW, mode="valid")
            episodes = range(SMOOTHING_WINDOW, NUM_EPISODES + 1)
            ax2.plot(episodes, smoothed_s * 100, label=f"γ={gamma}", color=color, linewidth=1.8)

        ax2.set_xlabel("Episode")
        ax2.set_ylabel(f"Success rate % (rolling avg {SMOOTHING_WINDOW}ep)")
        ax2.set_title("Task Success Rate vs Episode")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(-5, 105)

        plt.tight_layout()
        plt.savefig("checkpoint4_gamma_experiments.png", dpi=120, bbox_inches="tight")
        print("\n[E] Plot saved → checkpoint4_gamma_experiments.png")
        plt.show()

    except ImportError:
        print("\n[E] matplotlib not found — skipping plot")

    print("\n" + "=" * 60)
    print("CHECKPOINT 4 COMPLETE")
    print("Look at the plots and think about:")
    print("  • Does γ=0.7 ever achieve consistent success? Why or why not?")
    print("  • Look at section [B]: from 20 steps away, how much of the +100")
    print("    dock reward does each agent 'see'? Does that match the success rate?")
    print("  • γ=0.99 values far-future rewards almost fully — does it converge")
    print("    faster or slower than γ=0.9? What does that tell you?")
    print("=" * 60)

    return results



# ── CHECKPOINT 5: Epsilon Decay Experiments (Exploration Schedule) ────────────
#
# Question: When should the agent stop exploring and start exploiting?
#
# Epsilon controls the explore/exploit tradeoff:
#   High epsilon → agent picks random actions often (explores unknown states)
#   Low epsilon  → agent follows its learned Q-values (exploits knowledge)
#
# We vary HOW FAST epsilon drops from 1.0 to 0.05 across 500 episodes.
# Three schedules, each defined by the episode where epsilon hits its minimum:
#
#   FAST   → converges to ε_min by episode ~100  (exploits early, misses states)
#   MEDIUM → converges to ε_min by episode ~300  (our baseline so far)
#   SLOW   → converges to ε_min by episode ~480  (explores almost the whole run)
#
# Fixed: alpha=0.5, gamma=0.9 (best from Checkpoints 3 & 4)

BEST_GAMMA = 0.9

DECAY_SCHEDULES = {
    "fast":   100,   # epsilon hits minimum at episode 100
    "medium": 300,   # epsilon hits minimum at episode 300
    "slow":   480,   # epsilon hits minimum at episode 480
}


def decay_epsilon_scheduled(episode: int, decay_episode: int) -> float:
    """
    Linear decay from EPSILON_START to EPSILON_MIN, reaching minimum at decay_episode.
    After that, stays at EPSILON_MIN.
    """
    if episode >= decay_episode:
        return EPSILON_MIN
    fraction = episode / decay_episode
    return max(EPSILON_MIN, EPSILON_START - fraction * (EPSILON_START - EPSILON_MIN))


def train_epsilon_greedy_scheduled(
    world: WarehouseGridWorld,
    alpha: float,
    gamma: float,
    num_episodes: int,
    decay_episode: int,
) -> tuple[np.ndarray, list[float], list[bool], list[float]]:
    """
    Same as train_epsilon_greedy but uses a custom epsilon decay schedule.
    Also records epsilon value per episode for plotting.
    """
    n_states  = world.get_state_space_size()
    n_actions = world.get_action_space_size()
    q_table   = np.zeros((n_states, n_actions))

    episode_rewards  = []
    episode_success  = []
    epsilon_trace    = []

    for episode in range(num_episodes):
        epsilon   = decay_epsilon_scheduled(episode, decay_episode)
        state     = world.reset()
        state_idx = world.state_to_index(state)
        total_reward = 0.0
        done = False

        while not done:
            action_idx    = epsilon_greedy(q_table, state_idx, epsilon)
            action        = ACTIONS[action_idx]
            result        = world.step(action)
            new_state_idx = world.state_to_index(result.state)

            td_target = result.reward + gamma * np.max(q_table[new_state_idx])
            td_error  = td_target - q_table[state_idx, action_idx]
            q_table[state_idx, action_idx] += alpha * td_error

            total_reward += result.reward
            state_idx     = new_state_idx
            done          = result.done

        episode_rewards.append(total_reward)
        episode_success.append(world.delivered and (world.robot_pos == world.dock_pos))
        epsilon_trace.append(epsilon)

    return q_table, episode_rewards, episode_success, epsilon_trace


def checkpoint_5_epsilon_decay():
    """
    Train with 3 epsilon decay schedules, everything else fixed.
    Compare: does the agent explore enough before exploiting?
    """
    print("\n" + "=" * 60)
    print("CHECKPOINT 5 — Epsilon Decay (Exploration Schedule) Experiments")
    print(f"  schedules={list(DECAY_SCHEDULES.keys())}  alpha={BEST_ALPHA}  gamma={BEST_GAMMA}")
    print("=" * 60)

    results = {}
    for name, decay_ep in DECAY_SCHEDULES.items():
        world = WarehouseGridWorld(seed=SEED)
        print(f"\n  Training '{name}' decay (ε hits min at ep {decay_ep}) ...")
        q_table, rewards, successes, eps_trace = train_epsilon_greedy_scheduled(
            world, BEST_ALPHA, BEST_GAMMA, NUM_EPISODES, decay_ep
        )
        results[name] = {
            "q_table":   q_table,
            "rewards":   rewards,
            "successes": successes,
            "eps_trace": eps_trace,
            "decay_ep":  decay_ep,
        }

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n[A] Results summary")
    print(f"  {'Schedule':>8} | {'Decay ep':>9} | {'Avg reward (all)':>17} | "
          f"{'Avg reward (last 100)':>22} | {'Success (last 100)':>19} | {'Converges ~ep':>14}")
    print("  " + "-" * 100)
    for name, decay_ep in DECAY_SCHEDULES.items():
        r  = results[name]["rewards"]
        s  = results[name]["successes"]
        cv = find_convergence_episode(r)
        print(
            f"  {name:>8} | {decay_ep:>9} | {np.mean(r):>17.1f} | "
            f"{np.mean(r[-100:]):>22.1f} | {np.mean(s[-100:])*100:>18.1f}% | "
            f"{'~ep ' + str(cv) if cv != -1 else 'never':>14}"
        )

    # ── The explore/exploit tradeoff spelled out ──────────────────────────────
    print("\n[B] Epsilon value at key training milestones")
    milestones = [50, 100, 200, 300, 400, 500]
    header = f"  {'Episode':>9} | " + " | ".join(f"{n:>8}" for n in DECAY_SCHEDULES)
    print(header)
    print("  " + "-" * 45)
    for ep in milestones:
        idx = min(ep - 1, NUM_EPISODES - 1)
        row = f"  {ep:>9} | "
        row += " | ".join(f"{results[n]['eps_trace'][idx]:>8.3f}" for n in DECAY_SCHEDULES)
        print(row)
    print("    → When ε is high the agent is still exploring; low = exploiting.")

    # ── What 'fast' decay risks: locking in before seeing enough ─────────────
    print("\n[C] Success rate in early training (eps 1–150)")
    print("    'Fast' exploits early — but was the Q-table ready?")
    for name in DECAY_SCHEDULES:
        early = results[name]["successes"][:150]
        print(f"    {name:>8}: success rate ep 1–150 = {np.mean(early)*100:.1f}%")

    # ── Progression per schedule ──────────────────────────────────────────────
    print("\n[D] Reward progression across training windows")
    windows = [(0, 50), (100, 150), (250, 300), (450, 500)]
    header2 = f"  {'Window':>12} | " + " | ".join(f"{n:>8}" for n in DECAY_SCHEDULES)
    print(header2)
    print("  " + "-" * (len(header2) - 2))
    for start, end in windows:
        row = f"  ep {start+1:>3}–{end:>3}  | "
        row += " | ".join(
            f"{np.mean(results[n]['rewards'][start:end]):>8.1f}" for n in DECAY_SCHEDULES
        )
        print(row)

    # ── Plot ──────────────────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt
        import matplotlib

        matplotlib.rcParams.update({"font.size": 11})
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(
            f"Checkpoint 5 — Epsilon Decay Schedules  (seed={SEED}, α={BEST_ALPHA}, γ={BEST_GAMMA})",
            fontsize=13,
        )

        colors = {"fast": "#e74c3c", "medium": "#2980b9", "slow": "#27ae60"}

        # Left: epsilon over time
        for name in DECAY_SCHEDULES:
            axes[0].plot(results[name]["eps_trace"], label=name,
                         color=colors[name], linewidth=1.8)
        axes[0].set_xlabel("Episode")
        axes[0].set_ylabel("Epsilon (ε)")
        axes[0].set_title("Exploration Rate Over Time")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Middle: cumulative reward
        for name in DECAY_SCHEDULES:
            r = results[name]["rewards"]
            smoothed = np.convolve(r, np.ones(SMOOTHING_WINDOW) / SMOOTHING_WINDOW, mode="valid")
            axes[1].plot(range(SMOOTHING_WINDOW, NUM_EPISODES + 1), smoothed,
                         label=name, color=colors[name], linewidth=1.8)
        axes[1].set_xlabel("Episode")
        axes[1].set_ylabel(f"Cumulative reward (rolling avg {SMOOTHING_WINDOW}ep)")
        axes[1].set_title("Cumulative Reward vs Episode")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(0, color="black", linewidth=0.7, linestyle="--")

        # Right: success rate
        for name in DECAY_SCHEDULES:
            s = np.array(results[name]["successes"], dtype=float)
            smoothed_s = np.convolve(s, np.ones(SMOOTHING_WINDOW) / SMOOTHING_WINDOW, mode="valid")
            axes[2].plot(range(SMOOTHING_WINDOW, NUM_EPISODES + 1), smoothed_s * 100,
                         label=name, color=colors[name], linewidth=1.8)
        axes[2].set_xlabel("Episode")
        axes[2].set_ylabel(f"Success rate % (rolling avg {SMOOTHING_WINDOW}ep)")
        axes[2].set_title("Task Success Rate vs Episode")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylim(-5, 105)

        plt.tight_layout()
        plt.savefig("checkpoint5_epsilon_decay.png", dpi=120, bbox_inches="tight")
        print("\n[E] Plot saved → checkpoint5_epsilon_decay.png")
        plt.show()

    except ImportError:
        print("\n[E] matplotlib not found — skipping plot")

    print("\n" + "=" * 60)
    print("CHECKPOINT 5 COMPLETE")
    print("Look at the plots and think about:")
    print("  • The LEFT plot shows ε over time — when does each schedule")
    print("    commit to exploitation? Overlay this mentally with the reward curve.")
    print("  • Does 'fast' ever catch up to 'medium' by episode 500?")
    print("    If not — what did it miss by stopping exploration at ep 100?")
    print("  • Does 'slow' suffer in the middle of training even if it recovers?")
    print("  • Which schedule would you pick for a REAL robot with limited")
    print("    training time? What if training time were unlimited?")
    print("=" * 60)

    return results


if __name__ == "__main__":
    checkpoint_1_state_space()
    checkpoint_2_training_loop()
    checkpoint_3_alpha_experiments()
    checkpoint_4_gamma_experiments()
    checkpoint_5_epsilon_decay()
