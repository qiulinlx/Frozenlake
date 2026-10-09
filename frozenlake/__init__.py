from jumanji import register

from frozenlake.env import FrozenLake
from frozenlake.types import Observation, Position, State
from frozenlake.viewer import FrozenLakeViewer

__all__ = [
    "FrozenLake",
    "FrozenLakeViewer",
    "Observation",
    "Position",
    "State",
]

# Frozen Lake with OpenAI Gym's 4x4 map and a time limit of 100.
register(
    id="FrozenLake-v0",
    entry_point="frozenlake:FrozenLake",
    kwargs={"time_limit": 100},
)
