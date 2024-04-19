from racecar_gym.env_austria import RaceEnv
from sb3_contrib import RecurrentPPO
import gymnasium
from gymnasium.wrappers import GrayScaleObservation, ResizeObservation
from stable_baselines3.common.vec_env import VecFrameStack, SubprocVecEnv, VecMonitor
import os
import numpy as np
from utils import *
scenario = 'circle_cw_competition_collisionStop'
scenario = 'austria_competition_collisionStop'
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
    env = VecMonitor(env)
        
    CHECKPOINT_DIR = './log/PPO_lstm/'
    LOG_DIR = './log/logs_PPO_lstm/'

    callback = TrainAndLoggingCallback(check_freq=2048, save_path=CHECKPOINT_DIR)
    model = RecurrentPPO('CnnLstmPolicy', env, verbose=1, tensorboard_log=LOG_DIR, device='cuda', learning_rate=3e-4, use_sde=True, n_steps=2048, batch_size=64, n_epochs=10, clip_range=0.2)
    # model = RecurrentPPO.load("./log/PPO_lstm/final_model.zip", env=env)
    model.learn(total_timesteps=1e7, callback=callback, progress_bar=True)
    model_path = os.path.join(CHECKPOINT_DIR, "final_model")
    model.save(model_path)