import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Lambda, Input, Dense, Concatenate, Add, Reshape
from keras.models import Model
EPS = 1e-8

def build_actor(num_features,n_actions):

	I1 = Input(shape = (num_features,))
	h1 = Dense(64,activation = 'tanh')(I1) 
	h2 = Dense(64,activation = 'tanh')(h1)

	mu = Dense(n_actions)(h2)
	log_std = Dense(n_actions)(Lambda(lambda x: K.zeros_like(x))(I1))
	std = Lambda(lambda x: K.exp(x))(log_std)
	pi = Lambda(lambda x: x[0] + x[1]*K.random_normal(shape=(K.shape(x[0]))))([mu, std])

	model = Model(I1, [mu, log_std, pi])

	return model

def build_critic(state_space,n_actions,n_ant):

	Inputs = []
	Inputs.append(Input(shape = (state_space,)))
	for i in range(n_ant):
		Inputs.append(Input(shape = (n_actions,)))
	I = Concatenate(axis=1)(Inputs)
	h = Dense(256,activation = 'relu')(I)
	h = Dense(256,activation = 'relu')(h)
	V = Dense(1)(h)
	model = Model(Inputs, V)

	return model

def build_acting(state_space,actors,n_ant):

	Inputs = Input(shape = (state_space,))
	Outputs = []
	for i in range(n_ant):
		Outputs.append(actors[i](Inputs)[2])

	return K.function([Inputs], Outputs)

def build_compute_target(state_space,actors,critic,n_ant,sample_num):

	S = Input(shape = (state_space,))
	S_t = Lambda(lambda x: K.repeat_elements(x,sample_num*(n_ant**2),axis = 0))(S)
	action_t = []
	for i in range(n_ant):
		action_t.append(actors[i](S_t)[2])

	q = critic([S_t]+action_t)
	q = Lambda(lambda x: K.reshape(x,(-1,sample_num*(n_ant**2),1)))(q)
	q = Lambda(lambda x: K.mean(x,axis = 1))(q)

	return K.function([S], [q])

def build_compute_advantage(state_space,actors,critic,n_ant,n_actions,sample_num):

	S = Input(shape = (state_space,))
	Inputs = []
	for i in range(n_ant):
		Inputs.append(Input(shape = (n_actions,)))

	S_t = Lambda(lambda x: K.repeat_elements(x,sample_num,axis = 0))(S)
	Inputs_t = []
	for i in range(n_ant):
		Inputs_t.append(Lambda(lambda x: K.repeat_elements(x,sample_num,axis = 0))(Inputs[i]))

	q = critic([S]+Inputs)

	Outputs = []
	for i in range(n_ant):
		q_i = critic([S_t]+Inputs_t[0:i]+[actors[i](S_t)[2]]+Inputs_t[i+1:n_ant])
		q_i = Lambda(lambda x: K.reshape(x,(-1,sample_num,1)))(q_i)
		Outputs.append(Lambda(lambda x: x[0] - K.mean(x[1],axis = 1))([q,q_i]))
	
	return K.function([S]+Inputs, Outputs)

class Agent(object):
	def __init__(self,sess,state_space,n_actions,n_ant,sample_num):
		super(Agent, self).__init__()
		self.sess = sess
		self.n_actions = n_actions
		self.n_ant = n_ant
		self.state_space = state_space
		self.actors = []
		self.entropy_coeff = 0.0
		self.sample_num = sample_num
		K.set_session(sess)
		
		for i in range(self.n_ant):
			self.actors.append(build_actor(self.state_space,self.n_actions))
		self.critic = build_critic(self.state_space,self.n_actions,self.n_ant)
		self.critic_tar = build_critic(self.state_space,self.n_actions,self.n_ant)
		self.acting = build_acting(self.state_space,self.actors,self.n_ant)
		self.compute_target = build_compute_target(self.state_space,self.actors,self.critic_tar,self.n_ant,self.sample_num)
		self.compute_advantage = build_compute_advantage(self.state_space,self.actors,self.critic,self.n_ant,self.n_actions,self.sample_num)
		
		self.label = tf.placeholder(tf.float32, [None, 1])
		self.adv = []
		self.a = []

		for i in range(self.n_ant):
			self.adv.append(tf.placeholder(tf.float32, [None, 1]))
			self.a.append(tf.placeholder(tf.float32, [None, self.n_actions]))
		self.opt_actors = []

		for i in range(self.n_ant):
			logp = tf.reshape(tf.reduce_sum(-0.5*(((self.a[i] - self.actors[i].outputs[0])/(tf.exp(self.actors[i].outputs[1])+EPS))**2 + 2*self.actors[i].outputs[1] + np.log(2*np.pi)), axis=1), (-1,1))
			pi_loss = -tf.reduce_mean(logp*self.adv[i])
			entropy_loss = tf.reduce_mean(logp)
			self.opt_actors.append(tf.train.AdamOptimizer(0.0003).minimize(pi_loss + self.entropy_coeff*entropy_loss,var_list = self.actors[i].trainable_weights))
		self.opt_critic = tf.train.AdamOptimizer(0.001).minimize(tf.reduce_mean((self.label - self.critic.outputs[0])**2),var_list = self.critic.trainable_weights)
		
		self.opt_actors = tf.group(self.opt_actors)
		self.sess.run(tf.global_variables_initializer())

	def train_actors(self, S, A, Adv):

		dict_t = {}
		for i in range(self.n_ant):
			dict_t[self.actors[i].inputs[0]] = S
			dict_t[self.adv[i]] = Adv[i]
			dict_t[self.a[i]] = A[i]
		return self.sess.run(self.opt_actors, feed_dict=dict_t)

	def train_critic(self, S, A, label):

		dict_t = {}
		dict_t[self.critic.inputs[0]] = S
		for i in range(self.n_ant):
			dict_t[self.critic.inputs[i+1]] = A[i]
		dict_t[self.label] = label
		return self.sess.run(self.opt_critic, feed_dict=dict_t)

	def update(self):

		weights = self.critic.get_weights()
		target_weights = self.critic_tar.get_weights()
		for w in range(len(weights)):
			target_weights[w] = (1 - 0.995)* weights[w] + 0.995* target_weights[w]
		self.critic_tar.set_weights(target_weights)

	def target_init(self):

		weights = self.critic.get_weights()
		self.critic_tar.set_weights(weights)