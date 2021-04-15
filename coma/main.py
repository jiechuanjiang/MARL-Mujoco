import os, sys
import gym
import numpy as np
import tensorflow as tf
from model import Agent
from buffer import ReplayBuffer
from config import *
config_tf = tf.ConfigProto()
config_tf.gpu_options.allow_growth = True
sess = tf.Session(config=config_tf)

env = gym.make('Hopper-v3')
n_ant = 3
state_space = 11
n_actions = 1

agents = Agent(sess,state_space,n_actions,n_ant,sample_num)
agents.target_init()
buff = ReplayBuffer(buffer_size+max_ep_len,state_space,n_actions,n_ant)
states = np.zeros((mini_batch,state_space))
next_states = np.zeros((mini_batch,state_space))
actions = np.zeros((n_ant,mini_batch,n_actions))
returns = np.zeros((mini_batch,1))
dones = np.zeros((mini_batch,1))
gammas = np.zeros((mini_batch,1))

reward_list = []
sum_reward = 0
obs = env.reset()
while setps<max_steps:

	outs = agents.acting([np.array([obs])])
	p = []
	for i in range(n_ant):
		p.append(outs[i][0])

	next_obs, reward, terminated, info = env.step(np.hstack(p))
	sum_reward += reward
	setps += 1
	buff.add(obs, p, reward, next_obs, terminated)
	obs = next_obs

	if terminated:
		obs = env.reset()
		terminated = False
		reward_list.append(sum_reward)
		sum_reward = 0
		if buff.pointer > buffer_size:

			print(np.mean(reward_list))
			reward_list = []

			for k in range(num_ite):
				states, actions, returns, next_states, dones, gammas = buff.getBatch(mini_batch)
				Q_target = agents.compute_target([next_states])[0]
				Q_target = returns + Q_target*gammas*(1 - dones)
				agents.train_critic(states, actions, Q_target)
				agents.update()

			states, actions, returns, next_states, dones, gammas = buff.getBatch(2000)
			advantages = agents.compute_advantage([states]+[actions[i] for i in range(n_ant)])
			if advantage_norm:
				for i in range(n_ant):
					advantages[i] = (advantages[i] - advantages[i].mean())/(advantages[i].std()+1e-8)
			agents.train_actors(states, actions, advantages)

			buff.reset()
