# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is **COSC-4117EL Assignment 2** — a reinforcement learning assignment focused on Q-learning implementations in a warehouse grid world domain. The robot must pick up a package, deliver it, and return to a charging dock while navigating obstacles and managing state.

## Key Concepts

### State Representation
States are represented as 4-tuples: `(row, col, has_package, delivered)`
- `row`, `col`: robot position (0-9 on a 10x10 grid)
- `has_package`: boolean flag (0 or 1)
- `delivered`: boolean flag (0 or 1)

Total state space: 10 × 10 × 2 × 2 = 400 states

### Grid Cell Types
- `0` (EMPTY): traversable floor
- `1` (SHELF): impassable obstacle
- `2` (CONGESTION): traversable but incurs -10 penalty
- `3` (PICKUP): pickup station
- `4` (PACKING): delivery station
- `5` (DOCK): charging dock (starting position)

### Actions
Four movement actions: `"up"`, `"down"`, `"left"`, `"right"`

### Rewards & Penalties
- Pickup action at pickup station: +25
- Delivery at packing station: +40
- Return to dock after delivery: +100
- Congestion zone: -10 (in addition to step penalty)
- Each step: -1
- Invalid move (hitting wall or boundary): -3

### Episode Termination
Episode ends when:
- Robot returns to dock after delivering (victory condition)
- Max steps (120) is reached

## Environment API

The main environment class is `WarehouseGridWorld` in `warehouse_gridworld_domain_random.py`.

### Core Methods

**Initialization & Control**
```python
world = WarehouseGridWorld(seed=42, max_steps=120)
state = world.reset()  # Returns initial state as 4-tuple
world.reseed(seed, regenerate_layout=False)  # Change seed
state = world.regenerate_layout(seed)  # New random layout with new seed
```

**Stepping & State**
```python
result = world.step(action)  # Returns StepResult dataclass
# StepResult has: .state, .reward, .done, .info (dict with event info)

state = world.get_state()  # Get current state
valid_actions = world.valid_actions()  # Get valid actions from current position
valid_actions = world.valid_actions(state)  # Get valid actions from any state
```

**State Management**
```python
state_index = world.state_to_index(state)  # Convert 4-tuple to linear index (0-399)
world.is_terminal()  # Check if episode is done
```

**Utilities**
```python
world.display()  # Text visualization with seed, state, score, positions
world.random_action()  # Sample random valid action
```

## Development Commands

### Run the interactive visualization
```bash
python warehouse_gridworld_domain_random.py
```
Use arrow keys to move, `R` to reset, `N` for new layout, `ESC` to quit.

### Run Q-learning implementations
```bash
python COSC_4117EL_A2_G1-q_learning_epsilon.py
python COSC_4117EL_A2_G1-q_learning_exploration.py
```

## Assignment File Structure

- **warehouse_gridworld_domain_random.py**: Environment & visualization (do not modify)
- **COSC_4117EL_A2_G1-q_learning_epsilon.py**: Implement epsilon-greedy Q-learning
- **COSC_4117EL_A2_G1-q_learning_exploration.py**: Implement alternative exploration strategy

## Q-Learning Architecture Notes

### Q-Table Representation
For discrete state-action spaces, use a 2D array or dictionary:
- Shape: `(num_states, num_actions)` where num_states=400, num_actions=4
- Index using: `state_to_index()` for row, action index for column
- Initialize with zeros

### Key Implementation Patterns

**Training Loop Pattern**
```python
for episode in range(num_episodes):
    state = world.reset()
    done = False
    while not done:
        # Select action (eps-greedy or other strategy)
        action = select_action(state, ...)
        result = world.step(action)
        # Update Q-table
        new_state = result.state
        reward = result.reward
        done = result.done
        # Q-learning update: Q[s,a] += alpha * (r + gamma * max(Q[s',a']) - Q[s,a])
        state = new_state
```

**Exploration vs Exploitation**
- Epsilon-greedy: `rand() < epsilon` → random action, else best action
- Other strategies: UCB, Boltzmann, optimistic initialization, etc.

**Validity Constraint**
The environment allows invalid moves, but they incur -3 penalty and don't change position. Consider using `valid_actions()` to restrict the action set during learning.

## Key Grid Layout Facts

- The grid has a fixed base layout with shelves/walls (hardcoded)
- Pickup, packing, and dock positions are randomly placed in empty cells via seeded RNG
- Use same seed for reproducibility; different seeds generate different valid layouts
- Layout persists across episode resets but changes with `regenerate_layout()`
