import numpy as np
class ReplayBuffer(object):

	def __init__(self, buffer_size, state_space, n_action, n_ant):
		self.buffer_size = buffer_size
		self.pointer = 0
		self.n_ant = n_ant
		self.state_space = state_space
		self.states = np.zeros((self.buffer_size,state_space))
		self.next_states = np.zeros((self.buffer_size,state_space))
		self.actions = np.zeros((n_ant,self.buffer_size,n_action))
		self.rewards = np.zeros((self.buffer_size,1))
		self.dones = np.zeros((self.buffer_size,1))
		self.horizen = 10
		self.gamma = 0.99

	def getBatch(self, batch_size):

		index = np.random.choice(self.pointer - self.horizen - 1, batch_size, replace=False)
		
		states_t = self.states[index]
		actions_t = self.actions[:,index]
		returns_t = np.zeros((batch_size,1))
		next_states_t = np.zeros((batch_size,self.state_space))
		dones_t = np.zeros((batch_size,1))
		gammas_t = np.zeros((batch_size,1))
		for k in range(batch_size):
			i = index[k]
			r = 0
			t = 1
			for j in range(self.horizen):
				r += self.rewards[i+j]*t
				t *= self.gamma
				if (self.dones[i+j] == 1):
					break
			returns_t[k] = r
			next_states_t[k] = self.next_states[i+j]
			dones_t[k] = self.dones[i+j]
			gammas_t[k] = t

		return states_t, actions_t, returns_t, next_states_t, dones_t, gammas_t


	def add(self, state, action, reward, next_state, done):

		self.states[self.pointer] = state
		self.next_states[self.pointer] = next_state
		for i in range(self.n_ant):
			self.actions[i][self.pointer] = action[i]
		self.rewards[self.pointer] = reward
		self.dones[self.pointer] = done
		self.pointer += 1

	def reset(self):

		self.pointer = 0
