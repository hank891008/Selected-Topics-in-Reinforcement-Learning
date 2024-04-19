from ppo_agent_atari import AtariPPOAgent

if __name__ == '__main__':

	config = {
		"gpu": True,
		"training_steps": 1e9,
		"update_sample_count": 10000,
		"discount_factor_gamma": 0.99,
		"discount_factor_lambda": 0.95,
		"clip_epsilon": 0.2,
		"max_gradient_norm": 0.5,
		"batch_size": 512,
		"logdir": 'log/Enduro/',
		"update_ppo_epoch": 3,
		"learning_rate": 2.5e-4,
		"value_coefficient": 0.5,
		"entropy_coefficient": 0.01,
		"horizon": 512,
		"env_id": 'Enduro-v5',
		"eval_interval": int(2**20),
		"eval_episode": 3,
		"num_envs": 256,
	}
	agent = AtariPPOAgent(config)
	agent.train()



