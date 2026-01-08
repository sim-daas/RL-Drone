import gymnasium
import PyFlyt.gym_envs # noqa

env = gymnasium.make("PyFlyt/QuadX-Hover-v4", render_mode="human")
obs, info = env.reset()

termination = False
truncation = False

while not (termination or truncation):
    action = env.action_space.sample()
    observation, reward, termination, truncation, info = env.step(action)

env.close()
