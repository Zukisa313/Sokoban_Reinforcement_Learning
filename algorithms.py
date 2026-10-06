"""
algorithms.py

Multiple reinforcement learning algorithms, all trained on SokobanEnv and
all exposing the SAME interface:

    agent.train(env, ...)
    agent.greedy_action(state) -> int

`state` is always the compact flat tuple `(agent_r, agent_c, block_r, block_c)`
from `env.flat_state()`.

Because every agent shares this interface, anything downstream --
especially the visualizer -- can treat them interchangeably. Add a new
algorithm by subclassing `BaseAgent` and implementing `train()` +
`greedy_action()`; nothing else in the project needs to change.

Implemented here:
    - QLearningAgent      (off-policy TD control)
    - SarsaAgent          (on-policy TD control)
    - MonteCarloAgent     (every-visit Monte Carlo control)
    - ValueIterationAgent (model-based dynamic programming -- uses
                           env.transition()/env.all_states() directly,
                           no episodes needed, converges to the *true*
                           optimal policy since the env is small & known)
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Dict, Tuple

import numpy as np

from sokoban_env import SokobanEnv

Position = Tuple[int, int]
Boxes = Tuple[Position, ...]
State = Tuple[int, int, Boxes]


class BaseAgent(ABC):
    name: str = "base"

    def __init__(self, n_actions: int = 4, seed: int = 0):
        self.n_actions = n_actions
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

    @abstractmethod
    def train(self, env: SokobanEnv, **kwargs) -> None:
        """Train the agent on `env`. Should be safe to call once."""
        ...

    @abstractmethod
    def greedy_action(self, state: State) -> int:
        """Return the best action for `state` under the learned policy."""
        ...


# -------------------------------------------------------------------- #
# Shared helper: tabular agents all keep a dict[state] -> Q-values array
# -------------------------------------------------------------------- #
class _TabularAgent(BaseAgent):
    def __init__(self, n_actions: int = 4, seed: int = 0):
        super().__init__(n_actions, seed)
        self.Q: Dict[State, np.ndarray] = {}

    def _q(self, state: State) -> np.ndarray:
        if state not in self.Q:
            self.Q[state] = np.zeros(self.n_actions)
        return self.Q[state]

    def greedy_action(self, state: State) -> int:
        return int(np.argmax(self._q(state)))

    def _epsilon_greedy(self, state: State, epsilon: float) -> int:
        if self.rng.random() < epsilon:
            return self.rng.randrange(self.n_actions)
        return self.greedy_action(state)

    @staticmethod
    def _epsilon_schedule(ep, episodes, eps_start, eps_end, eps_decay_episodes):
        return max(eps_end, eps_start - (eps_start - eps_end) * ep / eps_decay_episodes)


# -------------------------------------------------------------------- #
# Q-learning (off-policy TD control)
# -------------------------------------------------------------------- #
class QLearningAgent(_TabularAgent):
    name = "qlearning"

    def train(
        self,
        env: SokobanEnv,
        episodes: int = 4000,
        alpha: float = 0.2,
        gamma: float = 0.95,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay_episodes: int = 3000,
        verbose: bool = True,
    ) -> None:
        solved_flags = []
        for ep in range(episodes):
            env.reset()
            state = env.flat_state()
            eps = self._epsilon_schedule(ep, episodes, eps_start, eps_end, eps_decay_episodes)

            for _ in range(env.max_steps):
                action = self._epsilon_greedy(state, eps)
                _, reward, done, info = env.step(action)
                next_state = env.flat_state()

                best_next = np.max(self._q(next_state))
                td_target = reward + (0.0 if done else gamma * best_next)
                self._q(state)[action] += alpha * (td_target - self._q(state)[action])

                state = next_state
                if done:
                    solved_flags.append(1 if info.get("solved") else 0)
                    break
            else:
                solved_flags.append(0)

            if verbose and (ep + 1) % 500 == 0:
                recent = solved_flags[-200:]
                print(f"  [{self.name}] episode {ep+1:5d} | eps={eps:.2f} | solve rate: {sum(recent)/len(recent):.2f}")


# -------------------------------------------------------------------- #
# SARSA (on-policy TD control)
# -------------------------------------------------------------------- #
class SarsaAgent(_TabularAgent):
    name = "sarsa"

    def train(
        self,
        env: SokobanEnv,
        episodes: int = 4000,
        alpha: float = 0.2,
        gamma: float = 0.95,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay_episodes: int = 3000,
        verbose: bool = True,
    ) -> None:
        solved_flags = []
        for ep in range(episodes):
            env.reset()
            state = env.flat_state()
            eps = self._epsilon_schedule(ep, episodes, eps_start, eps_end, eps_decay_episodes)
            action = self._epsilon_greedy(state, eps)

            for _ in range(env.max_steps):
                _, reward, done, info = env.step(action)
                next_state = env.flat_state()
                # On-policy: the "next value" comes from the action SARSA
                # actually takes next (epsilon-greedy), not the max.
                next_action = self._epsilon_greedy(next_state, eps)

                td_target = reward + (0.0 if done else gamma * self._q(next_state)[next_action])
                self._q(state)[action] += alpha * (td_target - self._q(state)[action])

                state, action = next_state, next_action
                if done:
                    solved_flags.append(1 if info.get("solved") else 0)
                    break
            else:
                solved_flags.append(0)

            if verbose and (ep + 1) % 500 == 0:
                recent = solved_flags[-200:]
                print(f"  [{self.name}] episode {ep+1:5d} | eps={eps:.2f} | solve rate: {sum(recent)/len(recent):.2f}")


# -------------------------------------------------------------------- #
# Every-visit Monte Carlo control
# -------------------------------------------------------------------- #
class MonteCarloAgent(_TabularAgent):
    name = "monte_carlo"

    def train(
        self,
        env: SokobanEnv,
        episodes: int = 6000,
        gamma: float = 0.95,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay_episodes: int = 4500,
        verbose: bool = True,
    ) -> None:
        # incremental-mean return tracking per (state, action)
        counts: Dict[Tuple[State, int], int] = {}
        solved_flags = []

        for ep in range(episodes):
            env.reset()
            state = env.flat_state()
            eps = self._epsilon_schedule(ep, episodes, eps_start, eps_end, eps_decay_episodes)

            episode_trace = []  # (state, action, reward)
            solved = False
            for _ in range(env.max_steps):
                action = self._epsilon_greedy(state, eps)
                _, reward, done, info = env.step(action)
                episode_trace.append((state, action, reward))
                state = env.flat_state()
                if done:
                    solved = info.get("solved", False)
                    break
            solved_flags.append(1 if solved else 0)

            # Compute discounted returns backward through the episode and
            # do an every-visit incremental-mean Q update.
            G = 0.0
            for state_t, action_t, reward_t in reversed(episode_trace):
                G = reward_t + gamma * G
                key = (state_t, action_t)
                counts[key] = counts.get(key, 0) + 1
                q = self._q(state_t)
                q[action_t] += (G - q[action_t]) / counts[key]

            if verbose and (ep + 1) % 500 == 0:
                recent = solved_flags[-200:]
                print(f"  [{self.name}] episode {ep+1:5d} | eps={eps:.2f} | solve rate: {sum(recent)/len(recent):.2f}")


# -------------------------------------------------------------------- #
# Value Iteration (model-based dynamic programming)
# -------------------------------------------------------------------- #
class ValueIterationAgent(BaseAgent):
    name = "value_iteration"

    def __init__(self, n_actions: int = 4, seed: int = 0):
        super().__init__(n_actions, seed)
        self.V: Dict[State, float] = {}
        self._policy: Dict[State, int] = {}

    def train(
        self,
        env: SokobanEnv,
        gamma: float = 0.95,
        theta: float = 1e-4,
        max_sweeps: int = 500,
        verbose: bool = True,
    ) -> None:
        if len(env.block_positions) != 1:
            raise RuntimeError(
                "ValueIterationAgent is intentionally limited to one-box "
                "Sokoban. For multi-box puzzles use Q-learning, SARSA or MC."
            )

        states = list(env.all_states())
        V = {s: 0.0 for s in states}

        for sweep in range(max_sweeps):
            delta = 0.0
            for s in states:
                ar, ac, boxes = s
                agent_pos = (ar, ac)
                if env.is_solved_positions(boxes):
                    continue  # absorbing terminal state

                best_value = -float("inf")
                for action in range(self.n_actions):
                    new_agent, new_boxes, reward, done, _ = env.transition(
                        agent_pos, boxes, action
                    )
                    next_state = (new_agent[0], new_agent[1], new_boxes)
                    next_value = 0.0 if done else V.get(next_state, 0.0)
                    value = reward + gamma * next_value
                    best_value = max(best_value, value)

                delta = max(delta, abs(best_value - V[s]))
                V[s] = best_value

            if verbose and (sweep + 1) % 25 == 0:
                print(f"  [{self.name}] sweep {sweep+1:4d} | max delta: {delta:.5f}")
            if delta < theta:
                if verbose:
                    print(f"  [{self.name}] converged after {sweep+1} sweeps (delta={delta:.2e})")
                break

        self.V = V

        # Extract greedy policy
        for s in states:
            ar, ac, boxes = s
            agent_pos = (ar, ac)
            if env.is_solved_positions(boxes):
                continue

            best_action, best_value = 0, -float("inf")
            for action in range(self.n_actions):
                new_agent, new_boxes, reward, done, _ = env.transition(
                    agent_pos, boxes, action
                )
                next_state = (new_agent[0], new_agent[1], new_boxes)
                next_value = 0.0 if done else V.get(next_state, 0.0)
                value = reward + gamma * next_value
                if value > best_value:
                    best_value, best_action = value, action
            self._policy[s] = best_action

    def greedy_action(self, state: State) -> int:
        return self._policy.get(state, 0)

AGENT_REGISTRY = {
    "qlearning": QLearningAgent,
    "sarsa": SarsaAgent,
    "mc": MonteCarloAgent,
    "vi": ValueIterationAgent,
}
