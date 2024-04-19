import os
import numpy as np
import gymnasium
from gymnasium.wrappers import GrayScaleObservation, ResizeObservation
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecFrameStack, SubprocVecEnv, VecMonitor
from racecar_gym.env_austria import RaceEnv
from utils import ChannelLastObservation, TrainAndLoggingCallback, DiscreteActionWrapper_v2


scenario = 'austria_competition_collisionStop'
def make_env():
    env = RaceEnv(scenario=scenario,
        render_mode='rgb_array_birds_eye',
        reset_when_collision=False)
    env = ChannelLastObservation(env)
    env = GrayScaleObservation(env, keep_dim=True)
    env = ResizeObservation(env, 84)
    env = DiscreteActionWrapper_v2(env)
    return env

if __name__ == "__main__":
    # scenario = 'circle_cw_competition_collisionStop'
    scenario = 'austria_competition_collisionStop'
    CPU = 40
    env = SubprocVecEnv([lambda: make_env() for i in range(CPU)])
    env = VecFrameStack(env, 8, channels_order='last')
    env = VecMonitor(env)
    CHECKPOINT_DIR = './log/DQN_austria/'
    LOG_DIR = './log/logs_DQN_austria/'

    callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)
    model = DQN('CnnPolicy', env, verbose=1, tensorboard_log=LOG_DIR, device='cuda:1', buffer_size=5000 * 50, batch_size=64)
    model.learn(total_timesteps=2e6, callback=callback, progress_bar=True)
    model.save(os.path.join(CHECKPOINT_DIR, 'final_model'))