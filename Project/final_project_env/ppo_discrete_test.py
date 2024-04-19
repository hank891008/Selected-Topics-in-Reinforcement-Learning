import numpy as np
import gymnasium
from racecar_gym.env_origin import RaceEnv
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
from gymnasium.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, VecMonitor
from tqdm import tqdm
from stable_baselines3.common.evaluation import evaluate_policy
from utils import DiscreteActionWrapper_v2, ChannelLastObservation, imgs_to_gif, evaluate
scenario = 'austria_competition'
# scenario = 'circle_cw_competition_collisionStop'
reset_when_collision = True if 'austria' in scenario else False
def make_env():
    env = RaceEnv(scenario=scenario,
        render_mode='rgb_array_birds_eye',
        reset_when_collision=reset_when_collision)
    env = ChannelLastObservation(env)
    env = DiscreteActionWrapper_v2(env)
    env = GrayScaleObservation(env, keep_dim=True)
    env = ResizeObservation(env, 84)
    return env
env = make_env()
env = DummyVecEnv([lambda: make_env() for i in range(1)])
env = VecFrameStack(env, 12, channels_order='last') 
env = VecMonitor(env)
CHECKPOINT_DIR = './log/PPO_austria_discrete/'
model = PPO.load(CHECKPOINT_DIR + 'best_model_18432', env=env)

from PIL import Image
img_list = []
end = 0
for _ in range(1):
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
        # print(f'action: {action[0]}, progress: {state[0]["progress"]:.5f}, tota_reward: {reward[0]:.3f}')
        if done.all():
            if i == 4999:
                end += 1
            break
    print(state[0]["lap"], state[0]["checkpoint"])
    print(total_reward)
# print(end)
imgs = [Image.fromarray(img) for img in img_list]
imgs[0].save("result.gif", save_all=True, append_images=imgs[1:], duration=10, loop=0)