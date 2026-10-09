"""Roll out a random policy in Frozen Lake and save the episode as a gif."""
from dataclasses import dataclass

import tyro
import jax
import jumanji

from frozenlake.env import FrozenLake
from frozenlake.viewer import FrozenLakeViewer

@dataclass
class Args:
    """ set rollout parameters"""

    seed: int = 1
    """seed of the experiment"""
    max_episode_steps: int = 100
    """the number of steps after which an episode is truncated"""
    render: bool = False
    """if toggled, save the animation of the simulated episode as a gif"""


def main() -> None:

    # load arguments
    args = tyro.cli(Args)

    # create Frozen Lake environment and jit the corresponding reset and step functions
    env = FrozenLake(time_limit=args.max_episode_steps)
    reset_fn, step_fn = jax.jit(env.reset), jax.jit(env.step)

    # reset environment
    key = jax.random.PRNGKey(args.seed)
    key, reset_key = jax.random.split(key)
    state, timestep = reset_fn(reset_key)

    states = [state]
    while not timestep.last():

        # sample action randomly
        key, action_key = jax.random.split(key)
        action = env.action_space_sample(action_key)

        # step dynamics forward
        state, timestep = step_fn(state, action)
        states.append(state)

    print(f"episode length: {len(states) - 1}, final reward: {timestep.reward}")

    if args.render:
        viewer = FrozenLakeViewer("Frozen Lake")
        viewer.animate(states, interval=500, save_path="data/random_rollout.gif")


if __name__ == "__main__":
    main()
