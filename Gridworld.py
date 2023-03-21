from typing import Optional, Tuple
from frtypes import State, Observation, Position
import chex
import pygame
import jax
import jax.numpy as jnp
import matplotlib.animation as animation
from chex import Array, PRNGKey
import Visual
from jumanji import specs
from jumanji.env import Environment
from jumanji.types import TimeStep, restart, termination, transition

from jumanji import register



grid_size=4
# Define some colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (137, 207, 240)
RED = (255, 0, 0)
DBLUE= (25, 25, 112)
# This sets the WIDTH and HEIGHT of each grid location
WIDTH = 90
HEIGHT = 90
 
# This sets the margin between each cell
MARGIN = 5

class Frozenlake(Environment[State]):

    FIGURE_NAME = "Frozenlake"
    FIGURE_SIZE = (4.0, 4.0)
    MOVES = jnp.array([[-1, 0], [0, 1], [0, -1], [1, 0] ], jnp.int32) #List of actions 
    """Moves= List of actions
    0= Left
    1=Right
    2=Up
    3=Down
    """

    def __init__(self, grid_size: int=4) -> None:
        """Initialize the Gridworld.
        Args:
            grid_size: size of the grid. Defaults to 4.
            terminal: states where Agent ends episode
        
        Reset env to a grid 
        [0 0 0 0]
        [0 0 0 0]
        [0 0 0 0]
        [0 0 0 0]
        Initialise terminal states (Holes=-1 and Goal =1)
        [ 0  0 0  0]
        [ 0 -1 0 -1]
        [ 0  0 0 -1]
        [-1  0 0  1]
        """


        super().__init__()
        
        terminal= jnp.array([[1, grid_size-1],[grid_size-1,0] ,[grid_size-2, grid_size-1], [1, 1], [grid_size-1, grid_size-1]])
        self.grid = jnp.zeros((grid_size, grid_size), dtype=jnp.int32)
        x=terminal[..., 0]
        y=terminal[...,1]

        for i in range (0,grid_size):
             self.grid=self.grid.at[x[i], y[i]].set(-1)
        self.grid=self.grid.at[grid_size-1, grid_size-1].set(1)    
        self.grid=self.grid
        
        self.num_rows = grid_size
        self.num_cols = grid_size
        self.grid_shape = (grid_size, grid_size)
        

    def __repr__(self) -> str:
        """String representation of the environment.
        Returns:
        str: the string representation of the environment.
        """
        return f"Frozenlake(grid_size={grid_size})"
    
    def reset(self, key:chex.PRNGKey) -> Tuple[State, TimeStep[Observation]]:
        """Reset the environment to the initial state 
        Args:
          key= Random number so each initialisation is unique
        Returns:
          state: `State` object corresponding to the new state of the environment.
          timestep: `TimeStep` object corresponding to the first timestep returned by the
                   environment.
        """
     
        key, elf_key, goal_key= jax.random.split(key, 3)
        elf_coordinates = [0,0]
        elf_position = Position(*tuple(elf_coordinates))
        goal_coordinates= [grid_size, grid_size]
        goal_position= Position(*tuple(goal_coordinates))

        state = State(
            grid= self.grid,
            key = key,
            elf_position=elf_position,
            goal_position=goal_position,
            step_count=jnp.array(0, jnp.int32),
            action_mask=self._get_action_mask(elf_position),
            )
        timestep = restart(observation=self._state_to_observation(state))
        return state, timestep

    def step(self, state: State, action: chex.Numeric
             ) -> Tuple[State, TimeStep[Observation]]:
        """Run one timestep of the environment's dynamics.
        Args:
            state: `State` object containing the dynamics of the environment.
            action: Array containing the action to take:
                - 0 = Left
                - 1 = Right
                - 2 = Up
                - 3 = Down
        Returns:
            state, timestep: next state of the environment and timestep to be observed.
        """
             
        is_valid = state.action_mask[action]
        key, goal_key = jax.random.split(state.key, 2)

        elf_position = self._update_elf_position(state.elf_position, action)

        goal_achieved = elf_position == state.goal_position

        done = ~is_valid | goal_achieved 

        if state.grid[elf_position]==-1:
            done


        step_count = state.step_count + 1
        next_state = State(
            grid= self.grid,
            key=key,
            elf_position=elf_position,
            goal_position=state.goal_position,
            step_count=state.step_count + 1,
            action_mask=self._get_action_mask(elf_position),
        )


        if elf_position== state.goal_position:
            reward=1
        else:
            reward=0

        observation = self._state_to_observation(next_state)

        timestep = jax.lax.cond(
            done,
            termination,
            transition,
            reward,
            observation,
        )
        return next_state, timestep
    
    def observation_spec(self) -> specs.Spec[Observation]:
        """Returns the observation spec.
        Returns:
            Spec for the `Observation` whose fields are:
            - grid: BoundedArray (float) of shape (num_rows, num_cols, 5).
            - step_count: DiscreteArray (num_values = time_limit) of shape ().
            - action_mask: BoundedArray (bool) of shape (4,).
        """
        grid = specs.BoundedArray(
            shape=(self.grid_size, self.grid_size, 5),
            minimum=0.0,
            maximum=1.0,
            dtype=float,
            name="grid",
        )
        step_count = specs.DiscreteArray(
            self.time_limit, dtype=jnp.int32, name="step_count"
        )
        action_mask = specs.BoundedArray(
            shape=(4,),
            dtype=bool,
            minimum=False,
            maximum=True,
            name="action_mask",
        )
        return specs.Spec(
            Observation,
            "ObservationSpec",
            grid=grid,
            step_count=step_count,
            action_mask=action_mask,
        )

    def action_spec(self) -> specs.DiscreteArray:
        """Returns the action spec. 4 actions: [0,1,2,3] -> [Left, Right, Up, Down].
        Returns:
            action_spec: a `specs.DiscreteArray` spec.
        """
        return specs.DiscreteArray(4, name="action")
    
    def _state_to_observation(self, state: State) -> Observation:
        """Maps an environment state to an observation.
        Args:
            state: `State` object containing the dynamics of the environment.
        Returns:
            The observation derived from the state.
        """
        elf = jnp.array(state.elf_position)
        goal =jnp.array(state.goal_position)
        grid = grid = jnp.concatenate(
            jax.tree_util.tree_map(
                lambda x: x[..., None], [elf, goal]))

        return Observation(
            grid=grid,
            step_count=state.step_count,
            action_mask=state.action_mask,
        )


    def _get_action_mask( self, elf_position: Position,) -> chex.Array:
        """Checks whether the episode is over or not. Also checks the validity of the action 
        Args:
            Elf_position: Position of the Elf.
        Returns:
            action_mask: array (bool) of shape (4,).
        """

        def is_valid(move: chex.Array) -> chex.Array:
            new_elf_position = elf_position + Position(*tuple(move))
            outside_board = (
                (new_elf_position.row < 0)
                | (new_elf_position.row >= grid_size)
                | (new_elf_position.col < 0)
                | (new_elf_position.col >= grid_size)
            )
            return ~outside_board

        action_mask = jax.vmap(is_valid)(self.MOVES)
        return action_mask

    def _update_elf_position(
        self, elf_position: Position, action: chex.Numeric
    ) -> Position:
        """Give the new elf position after taking an action.
        Args:
            elf_position: `Position` of the elf.
            action: integer that tells in which direction to go.
        Returns:
            New elf position after taking the action.
        """
        # Possible moves are: Up, Right, Down, Left.
        row_move, col_move = self.MOVES[action]
        move_position = Position(row=row_move, col=col_move)
        next_elf_position = Position(*tuple(elf_position)) + move_position
        return next_elf_position
    
    def action_space_sample(
        self,
        body: chex.Array,
        key: chex.PRNGKey,
    ) -> Position:
        """Sample a random action.
        Args:
            Moves: 
            key: random key to generate a random Action
        Returns:
            action
        """
        action_index = jax.random.choice(
            key,
            jnp.int)
        return Position(row=row, col=col)

