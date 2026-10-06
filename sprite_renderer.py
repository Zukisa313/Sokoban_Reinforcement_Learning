from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import imageio
import numpy as np
from PIL import Image, ImageDraw

from sokoban_env import SokobanEnv, UP, DOWN, LEFT, RIGHT

class SpriteRenderer:
    """
    Sprite renderer for the multi-box Sokoban environment.

    Compatible with visualize_policy.py.

    The environment is expected to contain:

        env.grid_size
        env.walls
        env.agent_pos
        env.block_positions
        env.target_positions

    Coordinates are represented as:

        (row, column)

    Multiple boxes and multiple targets are supported.
    """

    def __init__(
        self,
        tile_size=64,
        assets_dir=None,
    ):
        self.tile_size = tile_size
        self.cell_px = tile_size

        if assets_dir is None:
            # Assume assets/sprites is next to the project files.
            project_dir = os.path.dirname(
                os.path.abspath(__file__)
            )

            assets_dir = os.path.join(
                project_dir,
                "assets",
                "sprites",
            )

        self.assets_dir = assets_dir

        self.sprites = {}

        self._load_sprites()

    # =========================================================
    # Sprite loading
    # =========================================================

    def _load_sprites(self):
        """
        Load all Sokoban sprites.
        Compatible across Pillow versions and supports both directional
        and single-player sprites with fallback generation.
        """
        # Resolve resampling filter across Pillow versions (<9.1 vs >=9.1)
        resample_filter = getattr(getattr(Image, "Resampling", Image), "NEAREST", Image.NEAREST)

        sprite_names = [
            "floor",
            "wall",
            "target",
            "crate",
            "crate_on_target",
            "player",
        ]

        for name in sprite_names:
            path = os.path.join(self.assets_dir, f"{name}.png")

            if os.path.exists(path):
                try:
                    image = Image.open(path).convert("RGBA")
                    if image.size != (self.tile_size, self.tile_size):
                        image = image.resize(
                            (self.tile_size, self.tile_size),
                            resample_filter,
                        )
                    self.sprites[name] = image
                except Exception as exc:
                    print(f"Warning: could not load {path}: {exc}")
                    self.sprites[name] = self._create_fallback_sprite(name)
            else:
                print(f"Warning: sprite not found: {path}")
                self.sprites[name] = self._create_fallback_sprite(name)

        # Alias directional player sprites to self.sprites["player"]
        # so render_frame(...) can look up player_down, player_up, etc. without error.
        base_player = self.sprites["player"]
        for direction in ("down", "up", "left", "right"):
            dir_name = f"player_{direction}"
            dir_path = os.path.join(self.assets_dir, f"{dir_name}.png")
            if os.path.exists(dir_path):
                try:
                    dir_img = Image.open(dir_path).convert("RGBA")
                    if dir_img.size != (self.tile_size, self.tile_size):
                        dir_img = dir_img.resize(
                            (self.tile_size, self.tile_size),
                            resample_filter,
                        )
                    self.sprites[dir_name] = dir_img
                except Exception:
                    self.sprites[dir_name] = base_player
            else:
                self.sprites[dir_name] = base_player
    # =========================================================
    # Fallback sprites
    # =========================================================

    def _create_fallback_sprite(self, name):
        """
        Create a simple fallback sprite when an asset
        cannot be found.
        """

        size = self.tile_size

        image = Image.new(
            "RGBA",
            (size, size),
            (0, 0, 0, 255),
        )

        pixels = image.load()

        if name == "floor":

            colour = (220, 220, 220, 255)

        elif name == "wall":

            colour = (70, 70, 70, 255)

        elif name == "target":

            colour = (220, 180, 70, 255)

        elif name == "crate":

            colour = (170, 105, 45, 255)

        elif name == "crate_on_target":

            colour = (90, 160, 90, 255)

        elif name == "player":

            colour = (70, 110, 210, 255)

        else:

            colour = (255, 0, 255, 255)

        for y in range(size):

            for x in range(size):

                pixels[x, y] = colour

        return image

    # =========================================================
    # Utilities
    # =========================================================

    def _get_grid_size(self, env):
        """
        Get rows and columns from env.grid_size.
        """

        grid_size = env.grid_size

        if isinstance(grid_size, int):

            return grid_size, grid_size

        return (
            int(grid_size[0]),
            int(grid_size[1]),
        )

    def _normalise_position(self, position):
        """
        Convert a position into a tuple of integers.
        """

        return (
            int(position[0]),
            int(position[1]),
        )

    # =========================================================
    # Render a single frame
    # =========================================================

    def render_frame(self, env: SokobanEnv, facing: str = "down", title: Optional[str] = None,) -> Image.Image:
        """Render the current state of `env` as a sprite-based PIL Image."""
        px = self.cell_px
        board = Image.new("RGBA", (env.cols * px, env.rows * px), (0, 0, 0, 0))

        target_set = set(env.target_positions)
        box_set = set(env.block_positions)

        for r in range(env.rows):
            for c in range(env.cols):
                pos = (r, c)
                x, y = c * px, r * px

                if pos in env.walls:
                    board.paste(self.sprites["wall"], (x, y), self.sprites["wall"])
                    continue

                base = "target" if pos in target_set else "floor"
                board.paste(self.sprites[base], (x, y), self.sprites[base])

                if pos in box_set:
                    crate_sprite = "crate_on_target" if pos in target_set else "crate"
                    board.paste(self.sprites[crate_sprite], (x, y), self.sprites[crate_sprite])

        ar, ac = env.agent_pos
        player_sprite = self.sprites.get(f"player_{facing}", self.sprites["player_down"])
        board.paste(player_sprite, (ac * px, ar * px), player_sprite)
# --- Draw algorithm name banner if provided ---
        if title:
            draw = ImageDraw.Draw(board)
            # Banner background for high contrast
            draw.rectangle([(0, 0), (board.width, 26)], fill=(0, 0, 0, 160))
            # Text label
            draw.text((10, 6), title, fill=(255, 255, 255, 255))
        return board

    # =========================================================
    # Render rollout GIF
    # =========================================================

    def render_rollout_gif(
        self,
        env,
        actions,
        gif_path,
        duration_ms=250,
        hold_last_ms=1600,
        title: Optional[str] = None,
    ):
        """
        Render an action sequence as a GIF.

        Parameters
        ----------
        env:
            SokobanEnv instance.

        actions:
            List of integer actions produced by the learned policy.

        gif_path:
            Output GIF filename.

        duration_ms:
            Duration of each normal frame.

        hold_last_ms:
            How long the final frame remains visible.
        """

        frames = []

        # -----------------------------------------------------
        # Reset environment before replaying actions
        # -----------------------------------------------------

        env.reset()

        # -----------------------------------------------------
        # Initial frame
        # -----------------------------------------------------

        frame = self.render_frame(
            env,
            facing="down",
            title = title
        )

        frames.append(
            np.array(
                frame.convert("RGB")
            )
        )

        # -----------------------------------------------------
        # Replay actions
        # -----------------------------------------------------

        for action in actions:

            _, _, done, _ = env.step(
                int(action)
            )

            frame = self.render_frame(
                env,
                facing="down",
                title = title,
            )

            frames.append(
                np.array(
                    frame.convert("RGB")
                )
            )

            if done:

                break

        # -----------------------------------------------------
        # Make sure output directory exists
        # -----------------------------------------------------

        output_dir = os.path.dirname(
            os.path.abspath(gif_path)
        )

        os.makedirs(
            output_dir,
            exist_ok=True,
        )

        # -----------------------------------------------------
        # Convert milliseconds to seconds
        # -----------------------------------------------------

        normal_duration = (
            max(duration_ms, 1) / 1000.0
        )

        final_duration = (
            max(hold_last_ms, 1) / 1000.0
        )

        # -----------------------------------------------------
        # GIF durations
        # -----------------------------------------------------

        durations = [
            normal_duration
            for _ in frames
        ]

        if len(durations) > 0:

            durations[-1] = final_duration

        # -----------------------------------------------------
        # Save GIF
        # -----------------------------------------------------

        imageio.mimsave(
            gif_path,
            frames,
            duration=durations,
            loop=0,
        )

        return gif_path

    # =========================================================
    # Static render
    # =========================================================

    def render(
        self,
        env,
        save_path=None,
        facing="down",
    ):
        """
        Render the current environment state.

        Returns a PIL Image.
        """

        frame = self.render_frame(
            env,
            facing=facing,
        )

        if save_path is not None:

            output_dir = os.path.dirname(
                os.path.abspath(save_path)
            )

            os.makedirs(
                output_dir,
                exist_ok=True,
            )

            frame.save(save_path)

        return frame

    # =========================================================
    # Policy map
    # =========================================================

    def render_policy_map(
        self,
        *args,
        **kwargs,
    ):
        """
        Policy maps from the original single-box project
        are intentionally disabled.

        In multi-box Sokoban, the best action cannot be
        represented by a single arrow for each grid cell.

        The policy depends on:

            player position
            box 1 position
            box 2 position
            ...
            all target positions

        Therefore use render_rollout_gif() instead.
        """

        raise RuntimeError(
            "\n"
            "render_policy_map() is not supported for "
            "multi-box Sokoban.\n\n"
            "The policy depends on the complete state:\n"
            "(agent position + all box positions).\n\n"
            "Use render_rollout_gif() to visualize the "
            "learned policy instead."
        )


# =============================================================
# Convenience function
# =============================================================

def render_rollout_gif(
    env,
    actions,
    gif_path="rollout.gif",
    duration_ms=250,
    hold_last_ms=1600,
    tile_size=64,
    assets_dir=None,
):
    """
    Convenience wrapper for rendering a rollout.

    Example:

        render_rollout_gif(
            env,
            actions,
            "solution.gif"
        )
    """

    renderer = SpriteRenderer(
        tile_size=tile_size,
        assets_dir=assets_dir,
    )

    return renderer.render_rollout_gif(
        env,
        actions,
        gif_path,
        duration_ms=duration_ms,
        hold_last_ms=hold_last_ms,
    )