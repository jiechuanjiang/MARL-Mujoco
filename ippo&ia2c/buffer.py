import numpy as np
class ReplayBuffer(object):

	def __init__(self, buffer_size, state_space, n_action, gamma, lamda):
		self.buffer_size = buffer_size
		self.pointer = 0
		self.states = np.zeros((self.buffer_size,state_space))
		self.actions = np.zeros((self.buffer_size,n_action))
		self.oldlogps = np.zeros((self.buffer_size,1))
		self.rewards = np.zeros((self.buffer_size,1))
		self.values = np.zeros((self.buffer_size,1))
		self.dones = np.zeros((self.buffer_size,1))
		self.advantages = np.zeros((self.buffer_size,1))
		self.returns = np.zeros((self.buffer_size,1))
		self.deltas = np.zeros((self.buffer_size,1))
		self.gamma = gamma
		self.lamda = lamda
		self.advantage_norm = True
		self.EPS = 1e-8

	def getBatch(self, batch_size):

		index = np.random.choice(self.pointer, batch_size, replace=False)
		return self.states[index], self.actions[index], self.oldlogps[index], self.advantages[index], self.returns[index]

	def add(self, state, action, reward, done, oldlogp, value):

		self.states[self.pointer] = state
		self.actions[self.pointer] = action
		self.rewards[self.pointer] = reward
		self.dones[self.pointer] = done
		self.oldlogps[self.pointer] = oldlogp
		self.values[self.pointer] = value
		self.pointer += 1

	def reset(self):

		self.pointer = 0

	def compute_targets(self):
		pre_return = 0
		pre_value = 0
		pre_advantage = 0
		for i in reversed(range(self.pointer)):

			self.returns[i] = self.rewards[i] + self.gamma*pre_return*(1-self.dones[i])
			self.deltas[i] = self.rewards[i] + self.gamma*pre_value*(1-self.dones[i]) - self.values[i]
			self.advantages[i] = self.deltas[i] + self.gamma*self.lamda*pre_advantage*(1 - self.dones[i])

			pre_return = self.returns[i]
			pre_value = self.values[i]
			pre_advantage = self.advantages[i]

		if self.advantage_norm:
			self.advantages[0:self.pointer] = (self.advantages[0:self.pointer] - self.advantages[0:self.pointer].mean()) / (self.advantages[0:self.pointer].std() + self.EPS)