from racecar_gym.env_austria import RaceEnv
from stable_baselines3 import PPO
import gymnasium
from gymnasium.wrappers import GrayScaleObservation, ResizeObservation
from stable_baselines3.common.vec_env import VecFrameStack, SubprocVecEnv, VecMonitor
import os
import numpy as np
from utils import *
scenario = 'circle_cw_competition_collisionStop'
# scenario = 'austria_competition_collisionStop'
reset_when_collision = True if 'austria' in scenario else False
if __name__ == "__main__":
    def make_env():
        env = RaceEnv(scenario='austria_competition_collisionStop',
            render_mode='rgb_array_birds_eye',
            reset_when_collision=False)
        env = ChannelLastObservation(env)
        env = GrayScaleObservation(env, keep_dim=True)
        env = ResizeObservation(env, 84)
        return env
    CPU = 80
    env = SubprocVecEnv([lambda: make_env() for i in range(CPU)])
    env = VecFrameStack(env, 8, channels_order='last')
    env = VecMonitor(env)
        
    CHECKPOINT_DIR = './log/PPO_circle_80/'
    LOG_DIR = './log/logs_PPO_circle_80/'

    callback = TrainAndLoggingCallback(check_freq=1024, save_path=CHECKPOINT_DIR)
    model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=LOG_DIR, device='cuda', learning_rate=3e-4, use_sde=True, n_steps=1024, batch_size=64, n_epochs=10, clip_range=0.2)
    model.learn(total_timesteps=5e6, callback=callback, progress_bar=True)
    model_path = os.path.join(CHECKPOINT_DIR, "final_model")
    model.save(model_path)