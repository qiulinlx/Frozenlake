# Frozenlake In JAX using Jumanji API

This is an implementation of OpenAI gyms Frozen Lake using JAX and Jumanji. This is a simple gridworld where the agent must get to the goal while avoiding the holes.

<p align="center">
<img width="256" height="250" src="https://user-images.githubusercontent.com/110373610/226537623-c6aafa7c-a7bf-4208-875c-e6645ffd1785.png">
</p>

The Dark blue squares represent terminal states or equivalently holes, and the red squares represent the location of the reward. All blocks that aren't light blue will have terminal states. A simple implementation with details can be found in the examples.py file. Some of the basic commands that can be used is shown below. 

```ruby
env.reset(key)
state, timestep = jit(env.reset)(key)
action = env.action_spec().generate_value()
Visual.render(env.grid)
```
