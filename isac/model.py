import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Lambda, Input, Dense, Concatenate, Add, Reshape
from keras.models import Model
EPS = 1e-8
LOG_STD_MAX = 2
LOG_STD_MIN = -20
def build_actor(num_features,n_actions):

	I1 = Input(shape = (num_features,))
	h1 = Dense(256,activation = 'relu')(I1) 
	h2 = Dense(256,activation = 'relu')(h1)

	mu = Dense(n_actions)(h2)
	log_std = Dense(n_actions)(h2)
	log_std = Lambda(lambda x: K.clip(x, LOG_STD_MIN, LOG_STD_MAX))(log_std)
	std = Lambda(lambda x: K.exp(x))(log_std)
	pi = Lambda(lambda x: x[0] + x[1]*K.random_normal(shape=(K.shape(x[0]))))([mu, std])
	logp_pi = Lambda(lambda x: K.reshape(K.sum(-0.5*(((x[0]-x[1])/(K.exp(x[2])+EPS))**2 + 2*x[2] + np.log(2*np.pi)), axis=1) - K.sum(2*(np.log(2) - x[0] - K.softplus(-2*x[0])), axis=1), (-1,1)))([pi, mu, log_std])

	mu = Lambda(lambda x: K.tanh(x))(mu)
	pi = Lambda(lambda x: K.tanh(x))(pi)
	model = Model(I1, [mu, pi, logp_pi])

	return model

def build_critic(state_space,n_actions):

	S = Input(shape = (state_space,))
	A = Input(shape = (n_actions,))
	I = Concatenate(axis=1)([S,A])
	h = Dense(256,activation = 'relu')(I)
	h = Dense(256,activation = 'relu')(h)
	q = Dense(1)(h)

	model = Model([S,A], q)

	return model

def build_acting_train(state_space,actors,n_ant):

	Inputs = Input(shape = (state_space,))
	Outputs = []
	for i in range(n_ant):
		Outputs.append(actors[i](Inputs)[0])

	return K.function([Inputs], Outputs)

def build_acting_test(state_space,actors,n_ant):

	Inputs = Input(shape = (state_space,))
	Outputs = []
	for i in range(n_ant):
		Outputs.append(actors[i](Inputs)[1])

	return K.function([Inputs], Outputs)

def build_compute_target(state_space,actors,critics,n_ant):

	S = []
	Q = []
	E = []
	for i in range(n_ant):
		S.append(Input(shape = (state_space,)))
		Q.append(critics[i]([S[i],actors[i](S[i])[1]]))
		E.append(actors[i](S[i])[2])

	return K.function(S, Q + E)

class Agent(object):
	def __init__(self,sess,state_space,n_actions,n_ant,alpha):
		super(Agent, self).__init__()
		self.sess = sess
		self.n_actions = n_actions
		self.n_ant = n_ant
		self.state_space = state_space
		self.actors = []
		self.critics = []
		self.critics_tar = []
		self.alpha = alpha
		K.set_session(sess)
		
		for i in range(self.n_ant):
			self.actors.append(build_actor(self.state_space,self.n_actions))
			self.critics.append(build_critic(self.state_space,self.n_actions))
			self.critics_tar.append(build_critic(self.state_space,self.n_actions))
		self.acting_train = build_acting_train(self.state_space,self.actors,self.n_ant)
		self.acting_test = build_acting_test(self.state_space,self.actors,self.n_ant)
		self.compute_target = build_compute_target(self.state_space,self.actors,self.critics_tar,self.n_ant)
		
		self.label = []
		for i in range(self.n_ant):
			self.label.append(tf.placeholder(tf.float32, [None, 1]))
		
		self.opt_actors = []
		self.opt_critics = []
		for i in range(self.n_ant):

			pi_loss = - tf.reduce_mean(self.critics[i]([self.actors[i].inputs[0],self.actors[i].outputs[1]]))
			entropy_loss = tf.reduce_mean(self.actors[i].outputs[2])
			self.opt_actors.append(tf.train.AdamOptimizer(0.001).minimize(pi_loss + self.alpha*entropy_loss,var_list = self.actors[i].trainable_weights))
			self.opt_critics.append(tf.train.AdamOptimizer(0.001).minimize(tf.reduce_mean((self.label[i] - self.critics[i].outputs[0])**2),var_list = self.critics[i].trainable_weights))
		
		self.opt_actors = tf.group(self.opt_actors)
		self.opt_critics = tf.group(self.opt_critics)
		self.sess.run(tf.global_variables_initializer())

	def train_actors(self, S):

		dict_t = {}
		for i in range(self.n_ant):
			dict_t[self.actors[i].inputs[0]] = S[i]
		return self.sess.run(self.opt_actors, feed_dict=dict_t)

	def train_critics(self, S, A, label):

		dict_t = {}
		for i in range(self.n_ant):
			dict_t[self.critics[i].inputs[0]] = S[i]
			dict_t[self.critics[i].inputs[1]] = A[i]
			dict_t[self.label[i]] = label[i]
		return self.sess.run(self.opt_critics, feed_dict=dict_t)

	def update(self):

		for i in range(self.n_ant):
			weights = self.critics[i].get_weights()
			target_weights = self.critics_tar[i].get_weights()
			for w in range(len(weights)):
				target_weights[w] = (1 - 0.995)* weights[w] + 0.995* target_weights[w]
			self.critics_tar[i].set_weights(target_weights)

	def target_init(self):

		for i in range(self.n_ant):
			weights = self.critics[i].get_weights()
			self.critics_tar[i].set_weights(weights)
