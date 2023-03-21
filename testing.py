import Gridworld
from jax import random
from jax import jit
import Visual

key = random.PRNGKey(1)
env= Gridworld.Frozenlake()
env.reset(key)
state, timestep = jit(env.reset)(key) #Get the state and timestep 

action = env.action_spec().generate_value() #Generate a valid action given the state
state, timestep = jit(env.step)(state, action) #Tells the agent to take the action

grid=state.grid
elfpos=state.elf_position
Visual.render(state.grid, elfpos)
