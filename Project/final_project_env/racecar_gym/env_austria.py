from collections import OrderedDict
import gymnasium as gym
import numpy as np
from numpy import array, float32
# noinspection PyUnresolvedReferences
import racecar_gym.envs.gym_api


class RaceEnv(gym.Env):
    camera_name = 'camera_competition'
    motor_name = 'motor_competition'
    steering_name = 'steering_competition'
    """The environment wrapper for RaceCarGym.
    
    - scenario: str, the name of the scenario.
        'austria_competition' or
        'plechaty_competition'
    
    Notes
    -----
    - Assume there are only two actions: motor and steering.
    - Assume the observation is the camera value.
    """
    def __init__(self,
                 scenario: str,
                 render_mode: str = 'rgb_array_birds_eye',
                 reset_when_collision: bool = True,
                 **kwargs):
        self.scenario = scenario.upper()[0] + scenario.lower()[1:]
        self.env_id = f'SingleAgent{self.scenario}-v0'
        self.env = gym.make(id=self.env_id,
                            render_mode=render_mode,
                            reset_when_collision=reset_when_collision,
                            **kwargs)
        self.render_mode = render_mode
        # Assume actions only include: motor and steering
        self.action_space = gym.spaces.box.Box(low=-1., high=1., shape=(2,), dtype=float32)
        # Assume observation is the camera value
        # noinspection PyUnresolvedReferences
        observation_spaces = {k: v for k, v in self.env.observation_space.items()}
        assert self.camera_name in observation_spaces, f'One of the sensors must be {self.camera_name}. Check the scenario file.'
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(3, 128, 128), dtype=np.uint8)
        #
        self.cur_step = 0
        self.prev_info = dict()

    def observation_postprocess(self, obs):
        obs = obs[self.camera_name].astype(np.uint8).transpose(2, 0, 1)
        return obs

    def reset(self, *args, **kwargs: dict):
        if kwargs.get("options"):
            kwargs["options"]["mode"] = 'random'
        else:
            kwargs["options"] = {"mode": 'random'}
        self.cur_step = 0
        obs, *others = self.env.reset(*args, **kwargs)
        obs = self.observation_postprocess(obs)
        self.prev_info['motor'] = 0
        self.prev_info['steering'] = 0
        self.prev_info['state'] = others[0].copy()
        return obs, others

    def step(self, actions):
        self.cur_step += 1
        motor_action, steering_action = actions

        # Add a small noise and clip the actions
        motor_scale = 0.001
        steering_scale = 0.01
        motor_action = np.clip(motor_action + np.random.normal(scale=motor_scale), -1., 1.)
        steering_action = np.clip(steering_action + np.random.normal(scale=steering_scale), -1., 1.)


        dict_actions = OrderedDict([(self.motor_name, array(motor_action, dtype=float32)),
                                    (self.steering_name, array(steering_action, dtype=float32))])
        obs, *others = self.env.step(dict_actions)
        obs = self.observation_postprocess(obs)
        reward, done, truncated, state = others
        reward = 0
        truncated |= (state['time'] >= 100)

        if state['checkpoint'] != self.prev_info['state']['checkpoint']:
            reward += 15

        # reward += 0.1 * motor_action # max: 0.4
        reward += 1 * motor_action
        reward -= 0.1 * (abs(motor_action - self.prev_info['motor']) + abs(steering_action - self.prev_info['steering'])) # max: 0.2
        
        if state['progress'] > self.prev_info['state']['progress']: # move forward
            reward += 1000 * (state['progress'] - self.prev_info['state']['progress'])  # max: 0.6
        elif state['progress'] == self.prev_info['state']['progress']:   # not moving
            reward -= 0.1
        if state['wall_collision'] == True:
            reward = -100
            done = True
        self.prev_info['motor'] = motor_action.copy()
        self.prev_info['steering'] = steering_action.copy()
        self.prev_info['state'] = state.copy()
        
        return obs, reward, done, truncated, state

    def render(self):
        return self.env.render()