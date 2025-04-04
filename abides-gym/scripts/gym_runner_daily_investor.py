import gymnasium as gym
from tqdm import tqdm

# Import to register environments
import abides_gym

if __name__ == "__main__":

    env = gym.make(
        "markets-daily_investor-v0",
        background_config="rmsc04",
    )

    env.seed(0)
    state = env.reset()
    for i in tqdm(range(5)):
        state, reward, done, info = env.step(0)
