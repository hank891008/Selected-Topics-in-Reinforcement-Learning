import numpy as np
import gymnasium
from racecar_gym.env_origin import RaceEnv
from stable_baselines3 import DQN
import matplotlib.pyplot as plt
from gymnasium.wrappers import GrayScaleObservation, ResizeObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, VecMonitor
from tqdm import tqdm

from utils import DiscreteActionWrapper, ChannelLastObservation, imgs_to_gif, evaluate

# scenario = 'austria_competition'
scenario = 'circle_cw_competition_collisionStop'
reset_when_collision = True if 'austria' in scenario else False
def make_env():
    env = RaceEnv(scenario=scenario,
        render_mode='rgb_array_birds_eye',
        reset_when_collision=reset_when_collision)
    env = ChannelLastObservation(env)
    env = DiscreteActionWrapper(env)
    env = GrayScaleObservation(env, keep_dim=True)
    env = ResizeObservation(env, 84)
    return env
env = make_env()
env = DummyVecEnv([lambda: make_env() for i in range(1)])
env = VecFrameStack(env, 8, channels_order='last')
env = VecMonitor(env)

# CHECKPOINT_DIR = './log/DQN_multienv_circle/'
# model = DQN.load(CHECKPOINT_DIR + 'best_model_190000.zip') 

CHECKPOINT_DIR = './log/DQN_multienv_circle/'
model = DQN.load(CHECKPOINT_DIR + 'best_model_50000.zip')

# evaluate(model, env, n_eval_episodes=10)

img_list = []
obs = env.reset()
done = False
progress_bar = tqdm(range(5000))
total_reward = np.zeros(1)
for i in progress_bar:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, state = env.step(action)
    total_reward += reward[0]
    img_list.append(obs[0][:,:,0])
    progress_bar.set_description(f'steering: {action[0] - 1}, progress: {state[0]["progress"]:.3f}')
    if done.all():
        break
print(state[0]['lap'], state[0]['progress'])

imgs_to_gif(img_list)