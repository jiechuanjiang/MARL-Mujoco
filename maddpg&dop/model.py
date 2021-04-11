import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Lambda, Input, Dense, Concatenate, Add, Reshape
from keras.models import Model

def build_actor(num_features,n_actions):

	I1 = Input(shape = (num_features,))
	h1 = Dense(256,activation = 'relu')(I1) 
	h2 = Dense(256,activation = 'relu')(h1)
	V = Dense(n_actions,activation = 'tanh')(h2)
	model = Model(I1, V)

	return model

def build_critic(state_space,n_ant,n_actions):

	Inputs = []
	for i in range(n_ant):
		Inputs.append(Input(shape = (n_actions,)))
	Inputs.append(Input(shape = (state_space,)))

	I = Concatenate(axis=1)(Inputs)
	h = Dense(256,activation = 'relu')(I)
	h = Dense(256,activation = 'relu')(h)
	q_total = Dense(1)(h)
	model = Model(Inputs, q_total)

	return model

def build_acting(state_space,actors,n_ant):

	Inputs = Input(shape=[state_space])
	actions = []
	for i in range(n_ant):
		actions.append(actors[i](Inputs))
	model = Model(Inputs, actions)

	return model

def build_Q_tot(state_space,actors,critic,n_ant):

	Inputs = Input(shape=[state_space])
	actions = []
	for i in range(n_ant):
		actions.append(actors[i](Inputs))
	actions.append(Inputs)
	q_value = critic(actions)
	model = Model(Inputs, q_value)

	return model

def build_smallQ(num_features,n_actions):

	Inputs = []
	Inputs.append(Input(shape = (n_actions,)))
	Inputs.append(Input(shape = (num_features,)))

	I = Concatenate(axis=1)(Inputs)
	h = Dense(256,activation = 'relu')(I)
	h = Dense(256,activation = 'relu')(h)
	q = Dense(1)(h)

	model = Model(Inputs, q)
	return model

def build_mixer(num_features,n_ant):

	I1 = Input(shape = (n_ant,))
	I2 = Input(shape = (num_features,))

	W1 = Dense(256,activation = 'relu')(I2)
	W1 = Dense(n_ant)(W1)
	W1 = Lambda(lambda x: K.abs(x))(W1)
	W1 = Reshape((n_ant, 1))(W1)
	b1 = Dense(256,activation = 'relu')(I2)
	b1 = Dense(1)(b1)

	h = Lambda(lambda x: K.batch_dot(x[0],x[1]))([I1,W1])
	q_total = Add()([h, b1])
	model = Model([I1,I2], q_total)

	return model

def build_dop_critic(state_space,n_ant,n_actions):

	smallQ = []
	for i in range(n_ant):
		smallQ.append(build_smallQ(state_space,n_actions))
	mixer = build_mixer(state_space,n_ant)

	Inputs = []
	for i in range(n_ant):
		Inputs.append(Input(shape = (n_actions,)))
	Inputs.append(Input(shape = (state_space,)))

	q_values = []
	for i in range(n_ant):
		q_values.append(smallQ[i]([Inputs[i],Inputs[n_ant]]))
	q_values = Concatenate(axis=1)(q_values)

	q_total = mixer([q_values,Inputs[n_ant]])
	model = Model(Inputs, q_total)
	
	return model

class Agent(object):
	def __init__(self,sess,state_space,n_actions,n_ant,method):
		super(Agent, self).__init__()
		self.sess = sess
		self.n_actions = n_actions
		self.n_ant = n_ant
		self.state_space = state_space
		self.actors = []
		self.method = method
		K.set_session(sess)
		
		for i in range(self.n_ant):
			self.actors.append(build_actor(self.state_space,self.n_actions))
		if self.method == 'maddpg':
			self.critic = build_critic(self.state_space,self.n_ant,self.n_actions)
		else:
			self.critic = build_dop_critic(self.state_space,self.n_ant,self.n_actions)
		self.Q_tot = build_Q_tot(self.state_space,self.actors,self.critic,self.n_ant)
		self.acting = build_acting(self.state_space,self.actors,self.n_ant)
		
		self.actors_tar = []
		for i in range(self.n_ant):
			self.actors_tar.append(build_actor(self.state_space,self.n_actions))
		if self.method == 'maddpg':
			self.critic_tar = build_critic(self.state_space,self.n_ant,self.n_actions)
		else:
			self.critic_tar = build_dop_critic(self.state_space,self.n_ant,self.n_actions)
		self.Q_tot_tar = build_Q_tot(self.state_space,self.actors_tar,self.critic_tar,self.n_ant)
		
		self.label = tf.placeholder(tf.float32,[None, 1])
		self.optimize = []
		for i in range(self.n_ant):
			self.optimize.append(tf.train.AdamOptimizer(0.001).minimize(-tf.reduce_mean(self.Q_tot.output),var_list = self.actors[i].trainable_weights))

		self.opt_critic = tf.train.AdamOptimizer(0.001).minimize(tf.reduce_mean((self.label - self.critic.get_output_at(0))**2),var_list = self.critic.trainable_weights)
		
		self.opt_actor = tf.group(self.optimize)
		self.sess.run(tf.global_variables_initializer())

	def train_actors(self, X):

		dict_t = {}
		dict_t[self.Q_tot.input] = X
		return self.sess.run(self.opt_actor, feed_dict=dict_t)

	def train_critic(self, S, A, label):

		dict_t = {}
		for i in range(self.n_ant):
			dict_t[self.critic.inputs[i]] = A[i]
		dict_t[self.critic.inputs[self.n_ant]] = S
		dict_t[self.label] = label
		return self.sess.run(self.opt_critic, feed_dict=dict_t)

	def update(self):

		weights = self.Q_tot.get_weights()
		target_weights = self.Q_tot_tar.get_weights()
		for w in range(len(weights)):
			target_weights[w] = (1 - 0.995)* weights[w] + 0.995* target_weights[w]
		self.Q_tot_tar.set_weights(target_weights)

	def target_init(self):

		weights = self.Q_tot.get_weights()
		self.Q_tot_tar.set_weights(weights)
