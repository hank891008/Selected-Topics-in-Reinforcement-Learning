import torch
import torch.nn as nn
import numpy as np
import os
import time
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from replay_buffer.gae_replay_buffer import GaeSampleMemory
from base_agent import PPOBaseAgent
from models.atari_model import AtariNet
import gym

from gym.wrappers import atari_preprocessing
from gym.wrappers import FrameStack

class AtariPPOAgent(PPOBaseAgent):
	def __init__(self, config):
		super(AtariPPOAgent, self).__init__(config)
		### TODO ###
		# initialize env
		def make_env(render_mode="rgb_array", noop_max=30):
			env = gym.make(f'ALE/{config["env_id"]}', render_mode=render_mode)
			env = atari_preprocessing.AtariPreprocessing(env, frame_skip=1, noop_max=noop_max)
			env = FrameStack(env, 4)
			return env
		self.env = gym.vector.AsyncVectorEnv([
			lambda: make_env() for _ in range(self.num_envs)
		])

		### TODO ###
		# initialize test_env
		self.test_env = make_env("rgb_array", noop_max=0)

		self.net = AtariNet(self.env.single_action_space.n)
		self.net.to(self.device)
		self.lr = config["learning_rate"]
		self.update_count = config["update_ppo_epoch"]
		self.optim = torch.optim.Adam(self.net.parameters(), lr=self.lr, eps=1e-5)
		
	def decide_agent_actions(self, observation, eval=False):
		### TODO ###
		# add batch dimension in observation
		# get action, value, logp from net
		observation = observation.__array__()
		if len(observation.shape) == 3:
			observation = np.expand_dims(observation, axis=0)
		
		observation = torch.from_numpy(observation).to(self.device)
		if eval:
			with torch.no_grad():
				action, logp_pi, value = self.net(observation, eval=True)
		else:
			action, logp_pi, value = self.net(observation)
		action = action.detach().cpu().numpy()
		value = value.detach().cpu().numpy()
		logp_pi = logp_pi.detach().cpu().numpy()
		return action, value, logp_pi

	
	def update(self):
		loss_counter = 0.0001
		total_surrogate_loss = 0
		total_v_loss = 0
		total_entropy = 0
		total_loss = 0

		batches = self.gae_replay_buffer.extract_batch(self.discount_factor_gamma, self.discount_factor_lambda)
		sample_count = len(batches["action"])
		batch_index = np.random.permutation(sample_count)
		
		observation_batch = {}
		for key in batches["observation"]:
			observation_batch[key] = batches["observation"][key][batch_index]
		action_batch = batches["action"][batch_index]
		return_batch = batches["return"][batch_index]
		adv_batch = batches["adv"][batch_index]
		v_batch = batches["value"][batch_index]
		logp_pi_batch = batches["logp_pi"][batch_index]

		for _ in range(self.update_count):
			for start in range(0, sample_count, self.batch_size):
				ob_train_batch = {}
				for key in observation_batch:
					ob_train_batch[key] = observation_batch[key][start:start + self.batch_size]
				ac_train_batch = action_batch[start:start + self.batch_size]
				return_train_batch = return_batch[start:start + self.batch_size]
				adv_train_batch = adv_batch[start:start + self.batch_size]
				v_train_batch = v_batch[start:start + self.batch_size]
				logp_pi_train_batch = logp_pi_batch[start:start + self.batch_size]

				ob_train_batch = torch.from_numpy(ob_train_batch["observation_2d"])
				ob_train_batch = ob_train_batch.to(self.device, dtype=torch.float32)
				ac_train_batch = torch.from_numpy(ac_train_batch)
				ac_train_batch = ac_train_batch.to(self.device, dtype=torch.long)
				adv_train_batch = torch.from_numpy(adv_train_batch)
				adv_train_batch = adv_train_batch.to(self.device, dtype=torch.float32)
				logp_pi_train_batch = torch.from_numpy(logp_pi_train_batch)
				logp_pi_train_batch = logp_pi_train_batch.to(self.device, dtype=torch.float32)
				return_train_batch = torch.from_numpy(return_train_batch)
				return_train_batch = return_train_batch.to(self.device, dtype=torch.float32)

				### TODO ###
				# calculate loss and update network
				logp_pi, value, entropy = self.net.get_training_data(ob_train_batch, ac_train_batch)

				# calculate policy loss
				ratio = torch.exp(logp_pi - logp_pi_train_batch)
				pg_loss1 = -ratio * adv_train_batch
				pg_loss2 = -torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * adv_train_batch
				surrogate_loss = torch.mean(torch.max(pg_loss1, pg_loss2))

				# calculate value loss
				value_criterion = nn.MSELoss()
				v_loss = value_criterion(value, return_train_batch)
				
				# calculate entropy loss
				entropy = torch.mean(entropy)

				# calculate total loss
				loss = surrogate_loss + self.value_coefficient * v_loss - self.entropy_coefficient * entropy

				# update network
				self.optim.zero_grad()
				loss.backward()
				nn.utils.clip_grad_norm_(self.net.parameters(), self.max_gradient_norm)
				self.optim.step()

				total_surrogate_loss += surrogate_loss.item()
				total_v_loss += v_loss.item()
				total_entropy += entropy.item()
				total_loss += loss.item()
				loss_counter += 1

		self.writer.add_scalar('PPO/Loss', total_loss / loss_counter, self.total_time_step)
		self.writer.add_scalar('PPO/Surrogate Loss', total_surrogate_loss / loss_counter, self.total_time_step)
		self.writer.add_scalar('PPO/Value Loss', total_v_loss / loss_counter, self.total_time_step)
		self.writer.add_scalar('PPO/Entropy', total_entropy / loss_counter, self.total_time_step)
		print(f"Loss: {total_loss / loss_counter}\
			\tSurrogate Loss: {total_surrogate_loss / loss_counter}\
			\nValue Loss: {total_v_loss / loss_counter}\
			\tEntropy: {total_entropy / loss_counter}\
			")

	def train(self):
		observations, infos = self.env.reset()
		episode_rewards = np.zeros(self.num_envs)
		episode_lens = np.zeros(self.num_envs)

		while self.total_time_step <= self.training_steps:
			actions, values, logp_pis = self.decide_agent_actions(observations)
			next_observations, rewards, terminates, truncates, infos = self.env.step(actions)
			for i in range(self.num_envs):
				obs = {}
				obs["observation_2d"] = np.asarray(observations[i], dtype=np.float32)
				self.gae_replay_buffer.append(i, {
						"observation": obs,    # shape = (4,84,84)
						"action": np.array(actions[i]),      # shape = (1,)
						"reward": rewards[i],      # shape = ()
						"value": values[i],        # shape = ()
						"logp_pi": logp_pis[i],    # shape = ()
						"done": terminates[i],     # shape = ()
					})

				if len(self.gae_replay_buffer) >= self.update_sample_count:
					self.update()
					self.gae_replay_buffer.clear_buffer()
			episode_rewards += rewards
			episode_lens += 1		
				
			for i in range(self.num_envs):
				if terminates[i] or truncates[i]:
					if i == self.num_envs - 1:
						self.writer.add_scalar('Train/Episode Reward', episode_rewards[i], self.total_time_step)
						self.writer.add_scalar('Train/Episode Len', episode_lens[i], self.total_time_step)
					print(f"env[{i}]: [{len(self.gae_replay_buffer)}/{self.update_sample_count}][{self.total_time_step}/{self.training_steps}] episode reward: {episode_rewards[i]}  episode len: {episode_lens[i]}")
					episode_rewards[i] = 0
					episode_lens[i] = 0
			observations = next_observations
			self.total_time_step += self.num_envs
			
			if self.total_time_step % self.eval_interval == 0:
				# save model checkpoint
				avg_score = self.evaluate()
				self.save(os.path.join(self.writer.log_dir, f"model_{self.total_time_step}_{int(avg_score)}.pth"))
				self.writer.add_scalar('Evaluate/Episode Reward', avg_score, self.total_time_step)