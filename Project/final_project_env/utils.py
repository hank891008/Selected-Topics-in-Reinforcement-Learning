import os
import numpy as np
from PIL import Image
import gymnasium
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback

class CircleActionWrapper(gymnasium.ActionWrapper):
    def __init__(self, env):
        super(CircleActionWrapper, self).__init__(env)
        self.action_space = gymnasium.spaces.Box(low=np.array([0.15]), high=np.array([0.25]), shape=(1,), dtype=np.float32)
    
    def action(self, action):
        return action
        
class DiscreteActionWrapper(gymnasium.ActionWrapper):
    def __init__(self, env):
        super(DiscreteActionWrapper, self).__init__(env)
        self.action_space = gymnasium.spaces.Discrete(3)
    def action(self, action):
        if action == 0:
            return [1., -1.]
        elif action == 1:
            return [1., 0.]
        elif action == 2:
            return [1., 1.]
        else:
            raise ValueError(f'Invalid action: {action}')
        
class DiscreteActionWrapper_v2(gymnasium.ActionWrapper):
    def __init__(self, env):
        super(DiscreteActionWrapper_v2, self).__init__(env)
        self.action_space = gymnasium.spaces.Discrete(4)
    def action(self, action):
        if action == 0:
            return [-1, 0.]
        elif action == 1:
            return [1., -1.]
        elif action == 2:
            return [1., 0.]
        elif action == 3:
            return [1., 1.]
        else:
            raise ValueError('Invalid action value')
        
class ChannelLastObservation(gymnasium.ObservationWrapper):
    def __init__(self, env):
        super(ChannelLastObservation, self).__init__(env)
        obs_shape = self.observation_space.shape
        new_shape = (obs_shape[1], obs_shape[2], obs_shape[0])
        self.observation_space = gymnasium.spaces.Box(low=0, high=255, shape=new_shape, dtype=np.uint8)

    def observation(self, observation):
        return np.transpose(observation, (1, 2, 0))

class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if (self.n_calls) % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)
        return True
        
def imgs_to_gif(img_list, gif_name="result.gif"):
    imgs = [Image.fromarray(img) for img in img_list]
    imgs[0].save(gif_name, save_all=True, append_images=imgs[1:], duration=10, loop=0)

def evaluate(model, env, n_eval_episodes=10, deterministic=True):
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes, deterministic=deterministic)
    print(f"mean reward: {mean_reward}, std_reward: {std_reward}")