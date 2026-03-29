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


if __name__ == "__main__":
    checkpoint_1_state_space()
