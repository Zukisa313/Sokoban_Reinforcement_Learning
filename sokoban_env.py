"""
A dependency-light multi-box Sokoban environment for reinforcement learning.

The environment supports:
- Multiple boxes and multiple targets.
- An episode terminates successfully when EVERY box occupies a target.
- Boxes are interchangeable: their positions are stored in sorted order, which
  avoids treating identical boxes as different objects in the tabular state.
- Configurable grid size and walls.

For computational feasibility, a good default for tabular RL is an 8x8 grid
with 3 boxes and 3 targets. Larger grids/box counts can make the state space
grow very quickly.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, Dict, Any, List

EMPTY = 0
WALL = 1
AGENT = 2
BLOCK = 3
TARGET = 4
BLOCK_ON_TARGET = 5
AGENT_ON_TARGET = 6

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
ACTION_DELTAS = {
    UP: (-1, 0),
    DOWN: (1, 0),
    LEFT: (0, -1),
    RIGHT: (0, 1),
}
ACTION_NAMES = {UP: "UP", DOWN: "DOWN", LEFT: "LEFT", RIGHT: "RIGHT"}

GLYPHS = {
    EMPTY: ".",
    WALL: "#",
    AGENT: "A",
    BLOCK: "B",
    TARGET: "T",
    BLOCK_ON_TARGET: "*",
    AGENT_ON_TARGET: "@",
}

Position = Tuple[int, int]
Boxes = Tuple[Position, ...]
State = Tuple[int, int, Boxes]


class SokobanEnv:
    """
    Multi-box Sokoban-lite environment.

    Parameters
    ----------
    grid_size : int or (int, int)
        Grid dimensions. Default 8x8.
    agent_start : (int, int), optional
        Agent starting cell. Default (0, 0).
    block_starts : list[(int, int)], optional
        Starting positions of all boxes. Default is three boxes placed near
        the centre on an 8x8 grid.
    target_positions : list[(int, int)], optional
        Target cells. Number must equal number of boxes.
    block_start / target_pos :
        Backwards-compatible single-box arguments. If supplied, they are
        converted to one-element lists.
    max_steps : int
        Episode truncation limit.
    walls : list[(int, int)], optional
        Interior wall cells.
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        grid_size=8,
        agent_start: Optional[Position] = None,
        block_starts: Optional[List[Position]] = None,
        target_positions: Optional[List[Position]] = None,
        block_start: Optional[Position] = None,
        target_pos: Optional[Position] = None,
        max_steps: int = 500,
        step_penalty: float = -0.1,
        invalid_penalty: float = -1.0,
        push_reward: float = 1.0,
        success_reward: float = 100.0,
        walls: Optional[list] = None,
        seed: Optional[int] = None,
    ):
        if isinstance(grid_size, int):
            self.rows, self.cols = grid_size, grid_size
        else:
            self.rows, self.cols = grid_size

        self.agent_start = agent_start if agent_start is not None else (0, 0)

        # Backwards compatibility with the old single-box API.
        if block_starts is None:
            block_starts = [block_start] if block_start is not None else [
                (self.rows // 2, self.cols // 2 - 1),
                (self.rows // 2, self.cols // 2),
                (self.rows // 2 + 1, self.cols // 2),
            ]

        if target_positions is None:
            target_positions = [target_pos] if target_pos is not None else [
                (self.rows - 1, self.cols - 1),
                (self.rows - 2, self.cols - 1),
                (self.rows - 1, self.cols - 2),
            ]

        self.block_starts: Boxes = tuple(sorted(tuple(p) for p in block_starts))
        self.target_positions: Tuple[Position, ...] = tuple(
            sorted(tuple(p) for p in target_positions)
        )

        if len(self.block_starts) != len(self.target_positions):
            raise ValueError(
                f"Need the same number of boxes and targets; got "
                f"{len(self.block_starts)} boxes and {len(self.target_positions)} targets."
            )
        if len(self.block_starts) == 0:
            raise ValueError("At least one box and one target are required.")

        # Compatibility aliases for code that used the old API.
        self.block_start = self.block_starts[0]
        self.target_pos = self.target_positions[0]

        self.walls = set(walls) if walls else set()
        self.max_steps = max_steps
        self.step_penalty = step_penalty
        self.invalid_penalty = invalid_penalty
        self.push_reward = push_reward
        self.success_reward = success_reward

        self.action_space_n = 4
        self.rng = np.random.default_rng(seed)

        self._validate_positions()

        self.agent_pos = self.agent_start
        self.block_positions: Boxes = self.block_starts
        self._step_count = 0
        self._done = False

        # One BFS map per target. Used by the reward-shaping heuristic.
        self._dist_to_targets = {
            target: self._compute_distance_map(target)
            for target in self.target_positions
        }

    # ------------------------------------------------------------------ #
    # Setup / validation
    # ------------------------------------------------------------------ #
    def _validate_positions(self):
        all_named = [("agent_start", self.agent_start)]
        all_named += [(f"block_starts[{i}]", p) for i, p in enumerate(self.block_starts)]
        all_named += [(f"target_positions[{i}]", p)
                      for i, p in enumerate(self.target_positions)]

        for name, pos in all_named:
            r, c = pos
            if not (0 <= r < self.rows and 0 <= c < self.cols):
                raise ValueError(
                    f"{name}={pos} is outside the {self.rows}x{self.cols} grid"
                )
            if pos in self.walls:
                raise ValueError(f"{name}={pos} overlaps a wall cell")

        if len(set(self.block_starts)) != len(self.block_starts):
            raise ValueError("Two boxes cannot start in the same cell.")
        if len(set(self.target_positions)) != len(self.target_positions):
            raise ValueError("Two targets cannot occupy the same cell.")
        if self.agent_start in self.block_starts:
            raise ValueError("agent_start cannot overlap a starting box.")

    def _in_bounds(self, pos: Position) -> bool:
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols

    def _is_free(self, pos: Position) -> bool:
        return self._in_bounds(pos) and pos not in self.walls

    def _compute_distance_map(self, target: Position) -> Dict[Position, int]:
        from collections import deque

        dist = {target: 0}
        queue = deque([target])

        while queue:
            r, c = queue.popleft()
            for dr, dc in ACTION_DELTAS.values():
                nxt = (r + dr, c + dc)
                if self._is_free(nxt) and nxt not in dist:
                    dist[nxt] = dist[(r, c)] + 1
                    queue.append(nxt)
        return dist

    def _best_box_target_cost(self, boxes: Boxes) -> float:
        """
        Minimum total wall-aware distance matching boxes to targets.

        With the intended 3-box setting, checking all target permutations is
        tiny compared with the RL state space and gives useful shaping.
        """
        import itertools

        total = float("inf")
        for target_perm in itertools.permutations(self.target_positions):
            cost = 0
            reachable = True
            for box, target in zip(boxes, target_perm):
                d = self._dist_to_targets[target].get(box)
                if d is None:
                    reachable = False
                    break
                cost += d
            if reachable:
                total = min(total, cost)

        if total == float("inf"):
            return float(self.rows * self.cols * len(boxes))
        return float(total)

    # ------------------------------------------------------------------ #
    # Core environment API
    # ------------------------------------------------------------------ #
    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.agent_pos = self.agent_start
        self.block_positions = self.block_starts
        self._step_count = 0
        self._done = False
        return self._get_obs()

    def transition(
        self,
        agent_pos: Position,
        block_positions: Boxes,
        action: int,
    ):
        """
        Pure transition:
        (agent_pos, boxes, action) ->
        (new_agent_pos, new_boxes, reward, done, info)
        """
        if action not in ACTION_DELTAS:
            raise ValueError(f"Invalid action {action}; must be one of {list(ACTION_DELTAS)}")

        block_positions = tuple(sorted(block_positions))
        dr, dc = ACTION_DELTAS[action]
        new_agent = (agent_pos[0] + dr, agent_pos[1] + dc)

        reward = self.step_penalty
        info: Dict[str, Any] = {"pushed": False, "invalid": False}
        new_blocks = block_positions
        done = False

        if not self._is_free(new_agent):
            reward += self.invalid_penalty
            info["invalid"] = True
            new_agent = agent_pos

        elif new_agent in block_positions:
            # Identify the box being pushed.
            box_index = block_positions.index(new_agent)
            candidate_box = (new_agent[0] + dr, new_agent[1] + dc)

            occupied_by_other_box = (
                candidate_box in block_positions and candidate_box != new_agent
            )

            if self._is_free(candidate_box) and not occupied_by_other_box:
                old_cost = self._best_box_target_cost(block_positions)

                updated = list(block_positions)
                updated[box_index] = candidate_box
                new_blocks = tuple(sorted(updated))

                new_cost = self._best_box_target_cost(new_blocks)
                reward += self.push_reward * np.sign(old_cost - new_cost)

                info["pushed"] = True
                info["box_index"] = box_index

                if self.is_solved_positions(new_blocks):
                    reward += self.success_reward
                    done = True
                    info["solved"] = True
            else:
                reward += self.invalid_penalty
                info["invalid"] = True
                new_agent = agent_pos

        return new_agent, new_blocks, reward, done, info

    def step(self, action: int):
        if self._done:
            raise RuntimeError("Episode has ended. Call reset() before step().")

        new_agent, new_blocks, reward, terminated, info = self.transition(
            self.agent_pos, self.block_positions, action
        )

        self.agent_pos = new_agent
        self.block_positions = new_blocks
        self._step_count += 1
        self._done = terminated

        truncated = self._step_count >= self.max_steps and not terminated
        if truncated:
            info["truncated"] = True

        obs = self._get_obs()
        done = terminated or truncated
        return obs, reward, done, info

    # ------------------------------------------------------------------ #
    # State helpers
    # ------------------------------------------------------------------ #
    def flat_state(self) -> State:
        # Sorting makes identical boxes interchangeable, reducing the state
        # space by roughly a factor of num_boxes!.
        return (self.agent_pos[0], self.agent_pos[1], self.block_positions)

    def is_solved_positions(self, boxes: Boxes) -> bool:
        return set(boxes) == set(self.target_positions)

    def is_solved(self) -> bool:
        return self.is_solved_positions(self.block_positions)

    def all_states(self):
        """
        Enumerate all valid states.

        This is intentionally restricted to one-box Sokoban because exact
        tabular Value Iteration becomes combinatorially expensive with several
        boxes. Q-learning/SARSA/MC do not need this method.
        """
        if len(self.block_positions) != 1:
            raise RuntimeError(
                "all_states() is disabled for multi-box Sokoban because exact "
                "Value Iteration becomes too large. Use Q-learning, SARSA or MC."
            )

        cells = [
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if (r, c) not in self.walls
        ]
        for ar, ac in cells:
            for br , bc in cells:
                if (ar, ac) != (br, bc):
                    yield (ar, ac, ((br, bc),))

    def is_solvable(self, max_states: int = 250_000) -> bool:
        """
        Bounded BFS solvability check over the joint state space.

        Returns False if no solution is found before max_states are explored.
        For multi-box puzzles this is deliberately bounded so a difficult
        instance cannot consume unbounded memory.
        """
        from collections import deque

        start = self.flat_state()
        if self.is_solved():
            return True

        visited = {start}
        queue = deque([start])

        while queue and len(visited) < max_states:
            agent_pos_r, agent_pos_c, boxes = queue.popleft()
            agent_pos = (agent_pos_r, agent_pos_c)

            for action in range(self.action_space_n):
                new_agent, new_boxes, _, done, _ = self.transition(
                    agent_pos, boxes, action
                )
                if done and self.is_solved_positions(new_boxes):
                    return True

                state = (new_agent[0], new_agent[1], new_boxes)
                if state not in visited:
                    visited.add(state)
                    queue.append(state)

        return False

    def valid_actions(self, agent_pos: Position, block_positions: Boxes) -> List[int]:
        valid = []
        for action in range(self.action_space_n):
            _, _, _, _, info = self.transition(agent_pos, block_positions, action)
            if not info["invalid"]:
                valid.append(action)
        return valid or list(range(self.action_space_n))

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    def _get_obs(self) -> Dict[str, Any]:
        grid = np.zeros((self.rows, self.cols), dtype=np.int8)

        for w in self.walls:
            grid[w] = WALL

        # Targets first, boxes second so boxes on targets are rendered as *.
        for target in self.target_positions:
            grid[target] = TARGET

        for block in self.block_positions:
            grid[block] = BLOCK_ON_TARGET if block in self.target_positions else BLOCK

        ar, ac = self.agent_pos
        grid[ar, ac] = (
            AGENT_ON_TARGET if self.agent_pos in self.target_positions else AGENT
        )

        return {
            "grid": grid,
            "agent_pos": self.agent_pos,
            "block_positions": self.block_positions,
            "target_positions": self.target_positions,
            # Compatibility fields:
            "block_pos": self.block_positions[0],
            "target_pos": self.target_positions[0],
        }

    def render(self, mode: str = "human") -> Optional[str]:
        grid = self._get_obs()["grid"]
        text = "\n".join("".join(GLYPHS[cell] for cell in row) for row in grid)

        if mode == "ansi":
            return text
        print(text)
        return None

    def __repr__(self):
        return (
            f"SokobanEnv(grid={self.rows}x{self.cols}, "
            f"boxes={self.block_positions}, targets={self.target_positions}, "
            f"agent={self.agent_pos}, step={self._step_count}/{self.max_steps})"
        )
