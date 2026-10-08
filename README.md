# Frozen lake Environment in JAX & Jumanji 

This is an implementation of Gymnasium's [Frozen Lake](https://gymnasium.farama.org/environments/toy_text/frozen_lake/)  environment using `JAX` and `Jumanji`. Gymnasium Frozen Lake implementation has many functions, but it uses `numpy` which is quite a bit slower than JAX. This package uses `jax.numpy` which is 86 times faster than numpy when GPU is available. This package is built using `Jumanji`, meaning that users can write any functions that they need and seamelessly implement them within this or any Jumanji environment. 

The Frozen Lake environment is a playground for people to learn about RL algorithms. The agent (known as the elf) must reach the goal without falling into any holes. It represents a _simple gridworld_ problem where an agent must go from point A to point B. Unlike the Gymnasium version, the ice isn't slippery so the agent won't accidentally slide from one grid block to the next. This implementation is shown on the left and Gymnasium is on the right. Animation is also supported in this package. 

<p align="center">
<img width="250" height="250" src="https://user-images.githubusercontent.com/110373610/226537623-c6aafa7c-a7bf-4208-875c-e6645ffd1785.png">
<img width="250 height="250" src="https://user-images.githubusercontent.com/110373610/226541613-9707f3de-a707-40f5-a5b4-99303c8c410f.png">


The _dark blue squares_ represent terminal states or equivalently holes, and the _red squares_ represent the location of the gift (aka reward). All blocks that aren't light blue will have terminal states.    
## Installation
```bash
git clone https://github.com/riberaborrell/Frozenlake
pip install -e Frozenlake
```
                                                                                                             
## Quickstart
**TLDR: See the `frozenlake/examples/random_rollout.py` file**  
                                                                                                                                         
To get started, we need to install the necessary pacakges `jumanji` and `jaxlib` since this repo is built on top of these pacakges. This can be done by following instructions on the links
                                                                                   
                                                                                                                                         
https://github.com/instadeepai/jumanji (Jumanji) 

https://jax.readthedocs.io/en/latest/installation.html (Jax)

**Be Careful when installing JAX on Windows, if done incorrectly, it can lead to BSOD problems!!**                                                         

Once `Jaxlib` and `Jumanji` are running, we can clone this repo. 

```bash
git clone git@github.com:riberaborrell/Frozenlake.git
```
Now we can begin coding up the Frozen Lake environment easily. The environment and the elf agent is initialized and setup in the code below. 
                                                                                                                                        
```python
from frozenlake.env import FrozenLake

env = FrozenLake()
key = jax.random.PRNGKey(1)
state, timestep = env.reset(key)
```

Here are some simple functions that you'll need in your RL-algorithm. These functions will be able to return information about the environment and the agent such as the state of the env, the position of the agent and the reward and the action that the elf will take. 

```python
reset_fn, step_fn = jax.jit(env.reset), jax.jit(env.step)

state, timestep = reset_fn(key)
key, action_key = jax.random.split(key)
action = env.action_space_sample(action_key)
state, timestep = step_fn(state, action)
```
                                                                                                                                  
In order to visualise the environment and the position of the agent, we need to import the `FrozenLakeViewer` class from the `frozenlake.viewer` module, instantiate it and run the function `animate`.

```python
from frozenlake.viewer import FrozenLakeViewer

v = FrozenLakeViewer("Frozen Lake")
v.animate(states, interval=500, save_path="data/random_rollout.gif")
```                                                                                                                                         
                                                                                                                                        
        
