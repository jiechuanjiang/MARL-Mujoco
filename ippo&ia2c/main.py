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

agents = Agent(sess,state_space,n_actions,n_ant,method)
buff = []
for i in range(n_ant):
	buff.append(ReplayBuffer(buffer_size+max_ep_len,state_space,n_actions,gamma,lam))
states = np.zeros((n_ant,mini_batch,state_space))
actions = np.zeros((n_ant,mini_batch,n_actions))
oldlogps = np.zeros((n_ant,mini_batch,1))
advantages = np.zeros((n_ant,mini_batch,1))
returns = np.zeros((n_ant,mini_batch,1))

reward_list = []
sum_reward = 0
obs = env.reset()
while setps<max_steps:

	outs = agents.acting([np.array([obs])])
	p = []
	for i in range(n_ant):
		p.append(outs[3*i][0])
	next_obs, reward, terminated, info = env.step(np.hstack(p))
	sum_reward += reward
	setps += 1
	for i in range(n_ant):
		buff[i].add(obs, p[i], reward, terminated, outs[3*i+1][0], outs[3*i+2][0][0])
	obs = next_obs

	if terminated:
		obs = env.reset()
		terminated = False
		reward_list.append(sum_reward)
		sum_reward = 0
		if buff[0].pointer > buffer_size:

			print(np.mean(reward_list))
			reward_list = []

			for i in range(n_ant):
				buff[i].compute_targets()
			for k in range(num_ite):
				for i in range(n_ant):
					states[i], actions[i], oldlogps[i], advantages[i], returns[i] = buff[i].getBatch(mini_batch)
				agents.train_critics(states, returns)
				agents.train_actors(states, actions, advantages, oldlogps)

			for i in range(n_ant):
				buff[i].reset()
