# Sokoban with Reinforcement Learning

A reinforcement learning project for solving **multi-box Sokoban** using tabular reinforcement learning algorithms.

The project extends a basic Sokoban environment to support multiple boxes and targets, allowing different RL agents to learn how to coordinate box movements and solve increasingly complex puzzles.

## Algorithms

The project currently supports:

- **Q-Learning**
- **SARSA**
- **Monte Carlo (MC)**
- Value Iteration is included in the agent registry but is not recommended for the multi-box environment because its exact state-space enumeration is designed for smaller, single-box problems.

## Environment

The default environment is an **8 × 8 Sokoban board** containing:

- Multiple boxes
- Multiple target locations
- Walls
- A controllable player
- A maximum episode length of 500 steps

The environment uses a fixed puzzle configuration so that the different algorithms can be compared on the same problem.

## Project Structure

```text
.
├── visualize_policy.py                 # Train and evaluate agents
├── sokoban_env.py          # Sokoban environment
├── algorithms.py           # RL agent implementations
├── sprite_renderer.py      # Game rendering and GIF generation
└── outputs/                # Generated solutions and visualizations
└── assets/                 # env assets
```

## Installation

Clone the repository and install the required Python packages:

```bash
git clone <repository-url>
cd <repository-name>

pip install -r requirements.txt
```

## Running the Project

Run the default experiment:

```bash
python visualize_policy.py
```

By default, the program trains:

```text
Q-Learning
SARSA
Monte Carlo
```

for **12,000 episodes**.

### Run a specific algorithm

```bash
python visualize_policy.py --algo qlearning
```

Run multiple algorithms:

```bash
python visualize_policy.py --algo qlearning sarsa mc
```

### Change the number of training episodes

```bash
python visualize_policy.py --episodes 20000
```

### Change the maximum number of steps

```bash
python visualize_policy.py --max_steps 1000
```

### Set a random seed

```bash
python visualize_policy.py --seed 42
```

## Output

After training, each agent is evaluated using its **greedy policy**.

The project generates:

```text
outputs/
├── qlearning/
│   ├── solution.gif
│   └── solved_final.png
├── sarsa/
│   ├── solution.gif
│   └── solved_final.png
└── mc/
    ├── solution.gif
    └── solved_final.png
```

The GIF shows the agent's learned solution step-by-step, while `solved_final.png` shows the final state reached by the agent.

The terminal also reports whether the puzzle was solved, the number of steps taken, and training time.

Example:

```text
=== RESULTS ===
qlearning    | SOLVED     | steps=  19 | train_time=...
sarsa        | SOLVED     | steps=  21 | train_time=...
mc           | NOT SOLVED | steps= 500 | train_time=...
```

## Objective

The goal is for the agent to learn a policy that moves **all boxes onto their corresponding targets** while navigating around walls.

The project can be used to compare how different reinforcement learning algorithms perform when the state space becomes more complex due to the presence of multiple boxes.

## Future Improvements

Possible extensions include:

- Larger Sokoban boards
- More boxes and targets
- Deep Q-Learning (DQN)
- Actor-Critic methods
- Self-play or multi-agent training
- Curriculum learning
- Comparing learning speed and success rates
- Training across randomly generated Sokoban levels

## Author

**Zukisa Mkhize**

Built as an exploration of reinforcement learning and multi-agent/multi-object Sokoban environments.
