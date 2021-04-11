import os, sys  
import numpy as np
import gym
import tensorflow as tf
from model import Agent
from buffer import ReplayBuffer
from config import *
config_tf = tf.ConfigProto()
config_tf.gpu_options.allow_growth = True
sess = tf.Session(config=config_tf)

env = gym.make('HalfCheetah-v3')
test_env = gym.make('HalfCheetah-v3')
n_ant = 6
state_space = 17
n_actions = 1

agents = Agent(sess,state_space,n_actions,n_ant,method)
agents.target_init()
buff = ReplayBuffer(capacity)

X = np.zeros((batch_size,state_space))
next_X = np.zeros((batch_size,state_space))
A = np.zeros((n_ant,batch_size,n_actions))

def test_agent():
	sum_reward = 0
	for m in range(10):
		o, d, ep_l = test_env.reset(), False, 0
		while not(d or (ep_l == max_ep_len)):
			p = agents.acting.predict(np.array([o]))
			for i in range(n_ant):
				p[i] = p[i][0]
			o, r, d, _ = test_env.step(np.hstack(p))
			sum_reward += r
			ep_l += 1
	return sum_reward/10

obs = env.reset()
while setps<max_steps:

	p = agents.acting.predict(np.array([obs]))
	for i in range(n_ant):
		if setps < 10000:
			p[i] = 2*np.random.rand(n_actions) - 1
		else:
			p[i] = np.clip(p[i][0] + 0.1*np.random.randn(n_actions),-1,1)
	next_obs, reward, terminated, info = env.step(np.hstack(p))
	setps += 1
	ep_len += 1
	buff.add(obs, p, reward, next_obs, terminated)
	obs = next_obs

	if (terminated)|(ep_len == max_ep_len):
		obs = env.reset()
		terminated = False
		ep_len = 0

	if setps%10000==0:
		print(test_agent())

	if (setps < 1000)|(setps%50!=0):
		continue

	for e in range(50):
		batch = buff.getBatch(batch_size)
		for j in range(batch_size):
			X[j] = batch[j][0]
			next_X[j] = batch[j][3]
			for i in range(n_ant):
				A[i][j] = batch[j][1][i]

		Q_target = agents.Q_tot_tar.predict(next_X,batch_size = batch_size)
		for j in range(batch_size):
			Q_target[j] = batch[j][2] + Q_target[j]*gamma*(1 - batch[j][4])

		agents.train_critic(X, A, Q_target)
		agents.train_actors(X)
		agents.update()
