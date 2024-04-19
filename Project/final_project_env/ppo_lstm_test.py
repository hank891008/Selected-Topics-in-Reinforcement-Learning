import numpy as np
import gymnasium
from racecar_gym.env_origin import RaceEnv
from sb3_contrib import RecurrentPPO
import matplotlib.pyplot as plt
from gymnasium.wrappers import GrayScaleObservation, ResizeObservation
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from tqdm import tqdm
from utils import *
scenario = 'austria_competition'
# scenario = 'circle_cw_competition_collisionStop'


def make_env():
    env = RaceEnv(scenario=scenario,
        render_mode='rgb_array_birds_eye',
        reset_when_collision=True)
    env = ChannelLastObservation(env)
    env = GrayScaleObservation(env, keep_dim=True)
    env = ResizeObservation(env, 84)
    return env

env = DummyVecEnv([lambda: make_env() for i in range(1)])
env = VecMonitor(env)
CHECKPOINT_DIR = './log/PPO_lstm/'
model = RecurrentPPO.load(CHECKPOINT_DIR + 'best_model_154624', env=env)

from PIL import Image
img_list = []
obs = env.reset()
done = False
progress_bar = tqdm(range(5000))
total_reward = np.zeros(1)
lstm_states = None
for i in progress_bar:
    action, lstm_states = model.predict(obs, deterministic=True, state=lstm_states)
    obs, reward, done, state = env.step(action)
    total_reward += reward[0]
    img_list.append(obs[0][:,:,-1])
    progress_bar.set_description(f'action: {action[0]}, progress: {state[0]["progress"]:.5f}')
    if done.all():
        break


imgs = [Image.fromarray(img) for img in img_list]
imgs[0].save("result.gif", save_all=True, append_images=imgs[1:], duration=10, loop=0)
