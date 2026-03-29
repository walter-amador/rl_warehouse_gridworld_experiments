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


if __name__ == "__main__":
    checkpoint_1_state_space()
    checkpoint_2_training_loop()
