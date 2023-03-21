# Frozen lake Environment in JAX & Jumanji 

This is an implementation of OpenAI gyms Frozen Lake using JAX and Jumanji. OpenAIs Frozen Lake implementation has many functions, but it uses numpy whihc is quite a bit slower than jax. This package uses jax.numpy which is 86 times faster than numpy when GPU is available. This package is built using Jumanji, meaning that users can write any functions that they need and seamelessly implement them within this or any Jumanji environment. 

The Frozen Lake environment is a playground for people to learn about RL algorithms. The agent (known as the elf) must reach the goal without falling into any holes. It represents a simple gridworld problem where an agent must go from point A to point B. Unlike the OpenAI version, the ice isn't slippery so the agent won't accidentally slide from one grid block to the next. My implementation is shown on the left and OpenAIs is on the right. OpenAI supports multiple render forms but our package only supports one, which is shown below. 

<p align="center">
<img width="250" height="250" src="https://user-images.githubusercontent.com/110373610/226537623-c6aafa7c-a7bf-4208-875c-e6645ffd1785.png">
<img width="250 height="250" src="https://user-images.githubusercontent.com/110373610/226541613-9707f3de-a707-40f5-a5b4-99303c8c410f.png">
<\p>

The Dark blue squares represent terminal states or equivalently holes, and the red squares represent the location of the gift (aka reward). All blocks that aren't light blue will have terminal states.    
                                                                                                                                         
## Quickstart
To get started, we need to install jumanji and jaxlib. This can be done by following instructions on the links
                                                                                                                                         
https://github.com/instadeepai/jumanji (Jumanji) \                                                                                                         https://jax.readthedocs.io/en/latest/installation.html (Jax)\
**Be Careful when installing JAX on Windows, if done incorrectly, it can lead to BSOD problems!!**                                                         
                                                                                                                                         
                                                                                                                                        
```ruby
env.reset(key)
state, timestep = jit(env.reset)(key)
action = env.action_spec().generate_value()
Visual.render(env.grid)
```
