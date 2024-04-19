import numpy as np
import gymnasium
from racecar_gym.env_origin import RaceEnv
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
from gymnasium.wrappers import GrayScaleObservation, ResizeObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, VecMonitor
from tqdm import tqdm
from utils import *
# scenario = 'austria_competition'
scenario = 'circle_cw_competition_collisionStop'


def make_env():
    env = RaceEnv(scenario=scenario,
        render_mode='rgb_array_birds_eye',
        reset_when_collision=False)
    env = ChannelLastObservation(env)
    env = GrayScaleObservation(env, keep_dim=True)
    env = ResizeObservation(env, 84)
    return env

env = DummyVecEnv([lambda: make_env() for i in range(1)])
env = VecFrameStack(env, 8, channels_order='last') 
env = VecMonitor(env)
# CHECKPOINT_DIR = './log/PPO_austria_40env_1024steps_with_higher_noise/'
# model = PPO.load(CHECKPOINT_DIR + 'best_model_38912.zip', env=env)
# model = PPO.load(CHECKPOINT_DIR + 'best_model_40960.zip', env=env)
# model = PPO.load('./log/PPO_austria_40env_1024steps/best_model_45056', device='cuda:1')
CHECKPOINT_DIR = './log/PPO_circle_80/'
model = PPO.load(CHECKPOINT_DIR + 'best_model_37888', env=env)

from PIL import Image
img_list = []
end = 0
for _ in range(5):
    obs = env.reset()
    done = False
    progress_bar = tqdm(range(5000))
    total_reward = np.zeros(1)
    for i in progress_bar:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, state = env.step(action)
        total_reward += reward[0]
        img_list.append(obs[0][:,:,-1])
        progress_bar.set_description(f'action: {action[0]}, progress: {state[0]["progress"]:.5f}')
        if done.all():
            break
    print(state[0]["lap"], state[0]["checkpoint"])
    print(total_reward)
# print(end)
imgs = [Image.fromarray(img) for img in img_list]
imgs[0].save("result.gif", save_all=True, append_images=imgs[1:], duration=10, loop=0)
