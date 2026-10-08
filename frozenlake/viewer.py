from importlib import resources
from typing import ClassVar, Optional, Sequence, Tuple

import matplotlib.animation
import matplotlib.pyplot as plt
import numpy as np
from jumanji.viewer import MatplotlibViewer
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from numpy.typing import NDArray

from frozenlake.types import State

# Grid rendering constants
LIGHT_BLUE = [137 / 255, 207 / 255, 240 / 255]
RED = [1, 0, 0]
DARK_BLUE = [25 / 255, 25 / 255, 112 / 255]

# Images for rendering
ELF_IMAGE = resources.files("frozenlake") / "img" / "elf.png"

class FrozenLakeViewer(MatplotlibViewer[State]):

    FIGURE_NAME = "Frozenlake"
    FIGURE_SIZE = (4.0, 4.0)

    FROZEN = 0
    HOLE = 1
    GOAL = 2

    COLORS: ClassVar = {
        FROZEN: LIGHT_BLUE,
        HOLE: DARK_BLUE,
        GOAL: RED,
    }

    def __init__(self, name: str, render_mode: str = "human") -> None:
        """Viewer for the Frozen Lake environment.

        Args:
            name: the window name to be used when initialising the matplotlib window.
            render_mode: the mode used to render the environment. Must be one of:
                - "human": render the environment on screen.
                - "rgb_array": return a numpy array frame representing the environment.
        """
        super().__init__(name, render_mode, figure_size=(4.0, 4.0))
        with ELF_IMAGE.open("rb") as f:
            self._elf_image = plt.imread(f)

    def render(self, state: State, save_path: Optional[str] = None) -> Optional[NDArray]:
        """Render the given state of the `FrozenLake` environment.

        Args:
            state: the environment state to render.
            save_path: Optional path to save the rendered environment image to.

        Returns:
            RGB array if the render_mode is 'rgb_array'.
        """
        self._clear_display()
        fig, ax = self._get_fig_ax()
        ax.clear()
        self.draw(state, ax)

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", pad_inches=0.2)

        return self._display(fig)

    def animate(self, states: Sequence[State], interval: int = 200,
                save_path: Optional[str] = None) -> matplotlib.animation.FuncAnimation:
        """Create an animation from a sequence of environment states.

        Args:
            states: sequence of environment states corresponding to consecutive timesteps.
            interval: delay between frames in milliseconds, default to 200.
            save_path: the path where the animation file should be saved. If it is None, the plot
                will not be saved.

        Returns:
            Animation that can be saved as a GIF, MP4, or rendered with HTML.
        """
        fig, ax = self._get_fig_ax(name_suffix="_animation", show=False)
        plt.close(fig=fig)

        def make_frame(state: State) -> Tuple[Artist]:
            ax.clear()
            self.draw(state, ax)
            return (ax,)

        self._animation = matplotlib.animation.FuncAnimation(
            fig,
            make_frame,
            frames=states,
            interval=interval,
        )

        # Save the animation as a gif.
        if save_path:
            self._animation.save(save_path)

        return self._animation

    def draw(self, state: State, ax: Axes) -> None:
        """Draw the lake and the player of the given state on the axes."""
        holes = np.asarray(state.holes)
        num_rows, num_cols = holes.shape

        # Lake tiles
        tiles = np.where(holes, self.HOLE, self.FROZEN)
        tiles[int(state.goal_position.row), int(state.goal_position.col)] = self.GOAL
        img = np.zeros((num_rows, num_cols, 3))
        for tile, color in self.COLORS.items():
            img[tiles == tile] = color
        ax.imshow(img)

        # Grid lines
        ax.hlines(np.arange(num_rows + 1) - 0.5, -0.5, num_cols - 0.5, color="black", linewidth=2)
        ax.vlines(np.arange(num_cols + 1) - 0.5, -0.5, num_rows - 0.5, color="black", linewidth=2)

        # Agent
        row, col = int(state.player_position.row), int(state.player_position.col)
        ax.imshow(self._elf_image, extent=(col - 0.4, col + 0.4, row + 0.4, row - 0.4), zorder=2)

        ax.set_xlim(-0.5, num_cols - 0.5)
        ax.set_ylim(num_rows - 0.5, -0.5)
        ax.set_axis_off()
        ax.set_title(f"step {int(state.step_count)}")
