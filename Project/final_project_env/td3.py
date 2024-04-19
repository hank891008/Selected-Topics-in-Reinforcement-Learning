from racecar_gym.env_circle import RaceEnv
from stable_baselines3 import TD3
from gymnasium.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
from stable_baselines3.common.vec_env import VecFrameStack, SubprocVecEnv, VecMonitor
from utils import *
# scenario = 'austria_competition'
scenario = 'circle_cw_competition_collisionStop'

if __name__ == "__main__":
    
    def make_env():
        env = RaceEnv(scenario=scenario,
            render_mode='rgb_array_birds_eye',
            reset_when_collision=False)
        env = ChannelLastObservation(env)
        env = GrayScaleObservation(env, keep_dim=True)
        env = ResizeObservation(env, 84)
        return env
    CPU = 40
    env = SubprocVecEnv([lambda: make_env() for i in range(CPU)])
    env = VecFrameStack(env, 8, channels_order='last')
    env = VecMonitor(env)
    
    CHECKPOINT_DIR = './log/TD3_circle_steering02/'
    LOG_DIR = './log/logs_TD3_circle_steering02/'

    callback = TrainAndLoggingCallback(check_freq=5000, save_path=CHECKPOINT_DIR)
    model = TD3('CnnPolicy', env, verbose=1, tensorboard_log=LOG_DIR, device='cuda:1', buffer_size=10000, train_freq=(512, "step"), learning_rate=4.5e-5)
    model.learn(total_timesteps=2e6, callback=callback, progress_bar=True)