import gymnasium as gym
from stable_baselines3 import SAC, TD3, A2C, PPO
import os
import argparse

# Create directories to hold models and logs
model_dir = "models"
log_dir = "logs"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)


def train(env, sb3_algo):
    match sb3_algo:
        case "SAC":
            model = SAC(
                "MlpPolicy", env, verbose=1, device="cuda", tensorboard_log=log_dir
            )
        case "TD3":
            model = TD3(
                "MlpPolicy", env, verbose=1, device="cuda", tensorboard_log=log_dir
            )
        case "A2C":
            model = A2C(
                "MlpPolicy", env, verbose=1, device="cuda", tensorboard_log=log_dir
            )
        case "PPO":
            model = PPO(
                "MlpPolicy", env, verbose=1, device="cuda", tensorboard_log=log_dir
            )
        case _:
            print("Algorithm not found")
            return

    TIMESTEPS = 25000
    iters = 0
    while True:
        iters += 1
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
        model.save(f"{model_dir}/{sb3_algo}_{TIMESTEPS*iters}")
        if iters == 10:
            break


def test(env, sb3_algo, path_to_model):

    match sb3_algo:
        case "SAC":
            model = SAC.load(path_to_model, env=env)
        case "TD3":
            model = TD3.load(path_to_model, env=env)
        case "A2C":
            model = A2C.load(path_to_model, env=env)
        case "PPO":
            model = PPO.load(path_to_model, env=env)
        case _:
            print("Algorithm not found")
            return

    obs = env.reset()[0]
    done = False
    extra_steps = 500
    while True:
        action, _ = model.predict(obs)
        obs, _, done, _, _ = env.step(action)

        if done:
            extra_steps -= 1

            if extra_steps < 0:
                break


if __name__ == "__main__":
    gymenv = gym.make("Humanoid-v4", render_mode=None)
    train(gymenv, "PPO")

    testFile = "models/PPO_250000.zip"
    if os.path.isfile(testFile):
        gymenv = gym.make("Humanoid-v4", render_mode="human")
        test(gymenv, "PPO", testFile)
