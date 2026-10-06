import argparse
import os
import time
from typing import List, Tuple

from sokoban_env import SokobanEnv
from sprite_renderer import SpriteRenderer
from algorithms import AGENT_REGISTRY, BaseAgent

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def greedy_rollout_actions(
    env: SokobanEnv, agent: BaseAgent
) -> Tuple[List[int], bool]:
    """Run the learned greedy policy until all boxes are on targets."""
    env.reset()
    state = env.flat_state()
    actions = []

    for _ in range(env.max_steps):
        action = agent.greedy_action(state)
        actions.append(action)

        _, _, done, info = env.step(action)
        state = env.flat_state()

        if done:
            return actions, bool(info.get("solved", False))

    return actions, False


def visualize_agent(
    agent: BaseAgent,
    grid_size: int,
    max_steps: int,
    renderer: SpriteRenderer,
    agent_start: Tuple[int, int],  # <-- Add this parameter
    walls,
    block_starts,
    target_positions,
):
    out_dir = os.path.join(OUT_DIR, agent.name)
    os.makedirs(out_dir, exist_ok=True)

    env = SokobanEnv(
        grid_size=grid_size,
        max_steps=max_steps,
        seed=0,
        agent_start=agent_start,
        walls=walls,
        block_starts=block_starts,
        target_positions=target_positions,
    )

    actions, solved = greedy_rollout_actions(env, agent)
    status = "SOLVED ALL BOXES" if solved else "did NOT solve all boxes"
    print(f"  greedy rollout: {status} in {len(actions)} steps")

    # Animated GIF of the actual rollout.
    env.reset()
    gif_path = os.path.join(out_dir, "solution.gif")
    renderer.render_rollout_gif(
        env, actions, gif_path, duration_ms=2500, hold_last_ms=1600,title=f"Algorithm: {agent.name.upper()}",
    )

    # The old renderer's policy-map function assumes one fixed block, so it
    # is deliberately omitted for the multi-box environment.
    # The final frame is the useful static result for this version.
    env.reset()
    for a in actions:
        _, _, done, _ = env.step(a)
        if done:
            break

    renderer.render_frame(env, facing="down").save(
        os.path.join(out_dir, "solved_final.png")
    )

    print(f"  saved -> {out_dir}/(solution.gif, solved_final.png)")
    return solved, len(actions)


def build_agent(name: str, seed: int):
    cls = AGENT_REGISTRY[name]
    return cls(seed=seed)


def train_agent(name: str, agent, env, episodes: int):
    if name == "vi":
        agent.train(env)
    elif name == "mc":
        agent.train(env, episodes=max(episodes, 8000))
    else:
        agent.train(env, episodes=episodes)


def main():
    parser = argparse.ArgumentParser(
        description="Train & visualize multi-box Sokoban policies."
    )

    # Value iteration is excluded by default because its exact state-space
    # enumeration is deliberately limited to one box.
    default_algorithms = ["qlearning", "sarsa", "mc"]

    parser.add_argument(
        "--algo",
        nargs="+",
        default=default_algorithms,
        choices=list(AGENT_REGISTRY.keys()),
    )
    parser.add_argument(
        "--grid_size",
        type=int,
        default=8,
        help="Use 8x8 by default: large enough for 3 boxes, still practical for tabular RL.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=500,
        help="Maximum moves per episode.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=12000,
        help="Training episodes. Multi-box Sokoban generally needs more than single-box.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # 8x8, 3-box puzzle. The layout is intentionally open enough to remain
    # learnable while still requiring the agent to coordinate several boxes.
    walls = [
        (0,2),(0,3),(0,4),(0,5),(0,6),
        (1,0),(1,1),(1,2),(1,6),
        (2,0),(2,6),
        (3,0),(3,1),(3,2),(3,6),
        (4,0),(4,2),(4,3),(4,6),
        (5,0),(5,2),(5,6),
        (6,0),(6,7),
        (7,0),(7,7),

    ]

    # This fixed 8x8 instance has 3 interchangeable boxes and 3 targets.
    # It is solvable in about 19 moves under the environment's shortest-path
    # dynamics, while still being substantially larger than the old puzzle.
    agent_start = (2, 2)

    block_starts = [
        (2,3),
        (3,4),
        (4,4),
        (6,1),
        (6,3),
        (6,4),
        (6,5),
    ]

    target_positions = [
        (2, 1),
        (3, 5),
        (5, 4),
        (4, 1),
        (6,3),
        (7,4),
        (6,6),
   
    ]

    print("Multi-box Sokoban configuration")
    print(f"  grid: {args.grid_size}x{args.grid_size}")
    print(f"  boxes: {block_starts}")
    print(f" agent star {agent_start} !!!!!")
    print(f"  targets: {target_positions}")
    print(f"  walls: {len(walls)}")
    print()

    renderer = SpriteRenderer()
    results = {}

    for name in args.algo:
        print(f"\n=== {name} ===")

        if name == "vi":
            print(
                "  WARNING: Value Iteration is not suitable for this multi-box "
                "configuration. Use qlearning, sarsa or mc."
            )
            continue

        agent = build_agent(name, seed=args.seed)

        train_env = SokobanEnv(
            grid_size=args.grid_size,
            agent_start=agent_start,
            block_starts=block_starts,
            target_positions=target_positions,
            max_steps=args.max_steps,
            seed=args.seed,
            walls=walls,
        )

        t0 = time.time()
        train_agent(name, agent, train_env, args.episodes)
        train_time = time.time() - t0

        print(f"  trained in {train_time:.2f}s")

        solved, n_steps = visualize_agent(
            agent,
            args.grid_size,
            args.max_steps,
            renderer,
            agent_start,
            walls,
            block_starts,
            target_positions,
        )

        results[name] = (solved, n_steps, train_time)

    print("\n=== RESULTS ===")
    for name, (solved, n_steps, train_time) in results.items():
        status = "SOLVED" if solved else "NOT SOLVED"
        print(
            f"{name:12s} | {status:10s} | steps={n_steps:4d} | "
            f"train_time={train_time:.2f}s"
        )


if __name__ == "__main__":
    main()
