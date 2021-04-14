import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Lambda, Input, Dense, Concatenate, Add, Reshape
from keras.models import Model
EPS = 1e-8
clip_ratio = 0.2
def build_actor(num_features,n_actions):

	I1 = Input(shape = (num_features,))
	h1 = Dense(64,activation = 'tanh')(I1) 
	h2 = Dense(64,activation = 'tanh')(h1)

	mu = Dense(n_actions)(h2)
	log_std = Dense(n_actions)(Lambda(lambda x: K.zeros_like(x))(I1))
	std = Lambda(lambda x: K.exp(x))(log_std)
	pi = Lambda(lambda x: x[0] + x[1]*K.random_normal(shape=(K.shape(x[0]))))([mu, std])
	logp_pi = Lambda(lambda x: K.sum(-0.5*(((x[0]-x[1])/(K.exp(x[2])+EPS))**2 + 2*x[2] + np.log(2*np.pi)), axis=1))([pi, mu, log_std])

	model = Model(I1, [mu, log_std, pi, logp_pi])

	return model

def build_critic(state_space):

	Inputs = Input(shape = (state_space,))
	h = Dense(64,activation = 'tanh')(Inputs)
	h = Dense(64,activation = 'tanh')(h)
	V = Dense(1)(h)
	model = Model(Inputs, V)

	return model

def build_acting(state_space,actors,critics,n_ant):

	Inputs = Input(shape = (state_space,))
	Outputs = []
	for i in range(n_ant):
		Outputs.append(actors[i](Inputs)[2])
		Outputs.append(actors[i](Inputs)[3])
		Outputs.append(critics[i](Inputs))

	return K.function([Inputs], Outputs)

class Agent(object):
	def __init__(self,sess,state_space,n_actions,n_ant,method):
		super(Agent, self).__init__()
		self.sess = sess
		self.n_actions = n_actions
		self.n_ant = n_ant
		self.state_space = state_space
		self.actors = []
		self.critics = []
		self.method = method
		self.entropy_coeff = 0.00
		K.set_session(sess)
		
		for i in range(self.n_ant):
			self.actors.append(build_actor(self.state_space,self.n_actions))
			self.critics.append(build_critic(self.state_space))
		self.acting = build_acting(self.state_space,self.actors,self.critics,self.n_ant)
		
		self.label = []
		self.adv = []
		self.old_logp_pi = []
		self.a = []
		for i in range(self.n_ant):
			self.label.append(tf.placeholder(tf.float32, [None, 1]))
			self.adv.append(tf.placeholder(tf.float32, [None, 1]))
			self.old_logp_pi.append(tf.placeholder(tf.float32, [None, 1]))
			self.a.append(tf.placeholder(tf.float32, [None, self.n_actions]))
		self.opt_actors = []
		self.opt_critics = []
		for i in range(self.n_ant):

			logp = tf.reshape(tf.reduce_sum(-0.5*(((self.a[i] - self.actors[i].outputs[0])/(tf.exp(self.actors[i].outputs[1])+EPS))**2 + 2*self.actors[i].outputs[1] + np.log(2*np.pi)), axis=1), (-1,1))
			ratio = tf.exp(logp - self.old_logp_pi[i])
			clipped_ratio = tf.clip_by_value(ratio, 1 - clip_ratio, 1 + clip_ratio)
			if self.method == 'ippo':
				pi_loss = -tf.reduce_mean(tf.minimum(ratio*self.adv[i],clipped_ratio*self.adv[i]))
			else:
				pi_loss = -tf.reduce_mean(logp*self.adv[i])
			entropy_loss = tf.reduce_mean(logp)
			self.opt_actors.append(tf.train.AdamOptimizer(0.0003).minimize(pi_loss + self.entropy_coeff*entropy_loss,var_list = self.actors[i].trainable_weights))
			self.opt_critics.append(tf.train.AdamOptimizer(0.001).minimize(tf.reduce_mean((self.label[i] - self.critics[i].outputs[0])**2),var_list = self.critics[i].trainable_weights))
		
		self.opt_actors = tf.group(self.opt_actors)
		self.opt_critics = tf.group(self.opt_critics)
		self.sess.run(tf.global_variables_initializer())

	def train_actors(self, S, A, Adv, Old_Logp_Pi):

		dict_t = {}
		for i in range(self.n_ant):
			dict_t[self.actors[i].inputs[0]] = S[i]
			dict_t[self.adv[i]] = Adv[i]
			dict_t[self.old_logp_pi[i]] = Old_Logp_Pi[i]
			dict_t[self.a[i]] = A[i]
		return self.sess.run(self.opt_actors, feed_dict=dict_t)

	def train_critics(self, S, label):

		dict_t = {}
		for i in range(self.n_ant):
			dict_t[self.critics[i].inputs[0]] = S[i]
			dict_t[self.label[i]] = label[i]
		return self.sess.run(self.opt_critics, feed_dict=dict_t)