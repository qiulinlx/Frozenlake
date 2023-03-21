from jumanji import register

register(
    id="Frozenlake-v0",                            # format: (env_name)-v(version)
    entry_point="Gridworld: Frozenlake",
    kwargs={...},                                 # environment configuration
)