import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Int32
from jumanji import specs
from jumanji.env import Environment
from jumanji.types import TimeStep, restart, termination, transition, truncation

from frozenlake.types import State, Observation, Position

# Grid rendering constants
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (137, 207, 240)
RED = (255, 0, 0)
DBLUE = (25, 25, 112)

WIDTH = 90
HEIGHT = 90
MARGIN = 5

class FrozenLake(Environment[State, specs.DiscreteArray, Observation]):
    """4x4 gridworld environment with a goal and different holes.

    Actions:
        0 = Left, 1 = Down, 2 = Right, 3 = Up
    """

    FIGURE_NAME = "Frozenlake"
    FIGURE_SIZE = (4.0, 4.0)
    GRID_SIZE = 4
    MOVES = jnp.array([[0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)

    def __init__(self, goal_reward: float = 1.0, hole_reward: float = 0.0,
                 step_reward: float = 0.0, time_limit: int = 100) -> None:
        """Initialize the Gridworld.

        Args:
            goal_reward: Reward for reaching the goal. Defaults to 1.
            hole_reward: Reward for falling into a hole. Defaults to 0.
            step_reward: Reward for any other step. Defaults to 0.
            time_limit: Maximum number of steps per episode. Defaults to 100.
        """
        super().__init__()
        self.time_limit = time_limit

        self.grid_size = self.GRID_SIZE
        terminal = jnp.array([
            [1, 1],
            [1, self.grid_size - 1],
            [self.grid_size - 2, self.grid_size - 1],
            [self.grid_size - 1, 0],
            [self.grid_size - 1, self.grid_size - 1],
        ])
        self.grid = jnp.zeros((self.grid_size, self.grid_size), dtype=jnp.int32)
        x = terminal[..., 0]
        y = terminal[..., 1]

        # Mark holes as -1, except the last terminal state which is the goal with +1
        for i in range(len(terminal) - 1):
            self.grid = self.grid.at[x[i], y[i]].set(-1)
        self.grid = self.grid.at[self.grid_size - 1, self.grid_size - 1].set(1)

        self.goal_reward = goal_reward
        self.hole_reward = hole_reward
        self.step_reward = step_reward
        self.num_rows = self.grid_size
        self.num_cols = self.grid_size
        self.grid_shape = (self.grid_size, self.grid_size)


    def __repr__(self) -> str:
        return f"Frozenlake(grid_size={self.grid_size})"

    def reset(self, key: Array) -> tuple[State, TimeStep[Observation]]:
        """Reset the environment to the initial state.

        Args:
            key: Random key for reproducibility.

        Returns:
            State and initial TimeStep.
        """
        key, _ = jax.random.split(key, 2)
        player_position = Position(0, 0)
        goal_position = Position(self.grid_size - 1, self.grid_size - 1)

        state = State(
            grid=self.grid,
            key=key,
            player_position=player_position,
            goal_position=goal_position,
            step_count=jnp.array(0, jnp.int32),
            action_mask=self._get_action_mask(player_position),
        )
        timestep = restart(observation=self._state_to_observation(state))
        return state, timestep

    def step(self, state: State, action: Array) -> tuple[State, TimeStep[Observation]]:
        """Run one timestep of the environment's dynamics.

        Args:
            state: Current state of the environment.
            action: Action to take (0=Left, 1=Down, 2=Right, 3=Up).

        Returns:
            Next state and timestep.
        """
        # If the chosen action is invalid, i.e. it leaves the lake, it is a no-op.
        is_valid = state.action_mask[action]
        move = jnp.where(is_valid, self.MOVES[action], 0)
        player_position = self._update_player_position(state.player_position, move)

        # Check whether the episode terminates or is truncated.
        goal_achieved = player_position == state.goal_position
        fell_in_hole = state.grid[player_position.row, player_position.col] == -1
        terminated = goal_achieved | fell_in_hole
        truncated = state.step_count + 1 >= self.time_limit

        # Build the (updated) state.
        key, _ = jax.random.split(state.key, 2)
        next_state = State(
            grid=self.grid,
            key=key,
            player_position=player_position,
            goal_position=state.goal_position,
            step_count=state.step_count + 1,
            action_mask=self._get_action_mask(player_position),
        )
        observation = self._state_to_observation(next_state)

        # Compute the reward.
        reward = jnp.where(
            goal_achieved,
            self.goal_reward,
            jnp.where(fell_in_hole, self.hole_reward, self.step_reward),
        ).astype(float)

        # Termination takes precedence over truncation.
        timestep = jax.lax.cond(
            terminated,
            termination,
            lambda reward, observation: jax.lax.cond(
                truncated, truncation, transition, reward, observation
            ),
            reward,
            observation,
        )
        return next_state, timestep

    def observation_spec(self) -> specs.Spec[Observation]:
        """Returns the observation spec."""
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
        """Returns the action spec (4 discrete actions)."""
        return specs.DiscreteArray(4, name="action")

    def _state_to_observation(self, state: State) -> Observation:
        """Convert state to observation."""
        pos = jnp.array(state.player_position)
        goal = jnp.array(state.goal_position)
        grid = jnp.concatenate(
            jax.tree_util.tree_map(lambda x: x[..., None], [pos, goal])
        )
        return Observation(
            grid=grid,
            step_count=state.step_count,
            action_mask=state.action_mask,
        )


    def _get_action_mask(self, player_position: Position) -> Bool[Array, "4"]:
        """Get boolean mask of valid actions from current position."""

        def is_valid(move: Int32[Array, "2"]) -> Bool[Array, ""]:
            new_pos = player_position + Position(*tuple(move))
            outside = (
                (new_pos.row < 0)
                | (new_pos.row >= self.grid_size)
                | (new_pos.col < 0)
                | (new_pos.col >= self.grid_size)
            )
            return ~outside

        return jax.vmap(is_valid)(self.MOVES)

    def _update_player_position(self, player_position: Position, move: Array) -> Position:
        """Compute new player position after taking a moven."""
        return Position(
            row=player_position.row + move[0],
            col=player_position.col + move[1],
        )

    def action_space_sample(self, key: Array) -> Int32[Array, ""]:
        """Sample a random action uniformly from valid actions.

        Args:
            key: Random key for sampling.

        Returns:
            Action index (0-3).
        """
        return jax.random.randint(key, (), 0, self.action_spec().num_values)

