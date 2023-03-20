
from enum import IntEnum
from typing import TYPE_CHECKING, NamedTuple

import chex
from typing_extensions import TypeAlias

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from chex import dataclass

Gridworld: TypeAlias = chex.Array

class Position(NamedTuple):
    row: chex.Array
    col: chex.Array

    def __eq__(self, other: "Position") -> chex.Array:  # type: ignore[override]
        if not isinstance(other, Position):
            return NotImplemented
        return (self.row == other.row) & (self.col == other.col)

    def __add__(self, other: "Position") -> "Position":  # type: ignore[override]
        if not isinstance(other, Position):
            return NotImplemented
        return Position(row=self.row + other.row, col=self.col + other.col)
    
class Actions(IntEnum):
    UP: int = 0
    RIGHT: int = 1
    DOWN: int = 2
    LEFT: int = 3

@dataclass
class State:
    """
    grid: the grid, each nonzero element in the array corresponds
    to a game tile.
    step_count: the number of steps taken so far.
    action_mask: array of booleans that indicate the feasible actions, i.e. valid
    directions to move in.
    states on board, is of length num_terminals
    goal: Position of reward
    elf: position of agent
    """
    grid: Gridworld
    step_count: chex.Numeric  # ()
    action_mask: chex.Array  # (4,)
    goal_position: Position
    elf_position: Position
    key: chex.PRNGKey
  

class Observation(NamedTuple):
    """
    grid: feature maps that include information about the goal, the elf.
    step_count: current number of steps in the episode.
    action_mask: array specifying which directions the agent can move in from its current position.
    """

    grid: chex.Array  # (num_rows, num_cols, 5)
    step_count: chex.Numeric  # Shape ()
    action_mask: chex.Array  # (4,)