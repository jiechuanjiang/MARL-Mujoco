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

env = gym.make('Hopper-v3')
test_env = gym.make('Hopper-v3')
n_ant = 3
state_space = 11
n_actions = 1

agents = Agent(sess,state_space,n_actions,n_ant,alpha)
agents.target_init()
buff = []
for i in range(n_ant):
	buff.append(ReplayBuffer(capacity,state_space,n_actions))

X = np.zeros((n_ant,batch_size,state_space))
next_X = np.zeros((n_ant,batch_size,state_space))
A = np.zeros((n_ant,batch_size,n_actions))
R = np.zeros((n_ant,batch_size,1))
D = np.zeros((n_ant,batch_size,1))
Q_target = np.zeros((n_ant,batch_size,1))

def test_agent():
	sum_reward = 0
	for m in range(10):
		o, d, ep_l = test_env.reset(), False, 0
		while not(d or (ep_l == max_ep_len)):
			out = agents.acting_test([np.array([o])])
			p = []
			for i in range(n_ant):
				p.append(out[i][0])
			o, r, d, _ = test_env.step(np.hstack(p))
			sum_reward += r
			ep_l += 1
	return sum_reward/10

obs = env.reset()
while setps<max_steps:

	out = agents.acting_train([np.array([obs])])
	p = []
	for i in range(n_ant):
		if setps < 10000:
			p.append(2*np.random.rand(n_actions) - 1)
		else:
			p.append(out[i][0])
	next_obs, reward, terminated, info = env.step(np.hstack(p))
	setps += 1
	ep_len += 1
	for i in range(n_ant):
		buff[i].add(obs, p[i], reward, next_obs, terminated)
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

		for i in range(n_ant):
			X[i], A[i], R[i], next_X[i], D[i] = buff[i].getBatch(batch_size)
		q_e = agents.compute_target([next_X[i] for i in range(n_ant)])
		for i in range(n_ant):
			Q_target[i] = R[i] + (q_e[i] - alpha*q_e[i+n_ant])*gamma*(1 - D[i])

		agents.train_critics(X, A, Q_target)
		agents.train_actors(X)
		agents.update()