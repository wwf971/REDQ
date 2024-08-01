import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

def build_mlp(layer_shape, hidden_activation, output_activation):
	'''Build net with for loop'''
	layers = []
	for j in range(len(layer_shape)-1):
		act = hidden_activation if j < len(layer_shape)-2 else output_activation
		layers += [nn.Linear(layer_shape[j], layer_shape[j+1]), act()]
	return nn.Sequential(*layers)

class mlp_concat_layer(nn.Module):
	def __init__(self, weight, bias, act):
		super(mlp_concat_layer, self).__init__()
		self.weight = nn.Parameter(weight) # (mlp_num, input_num, output_num)
		self.bias = nn.Parameter(bias) # (mlp_num, output_num)
		self.act = act
	def forward(self, x): # x: (batch_size, mlp_num, input_dim)
		# x_unsqueeze = x.unsqueeze(1) # (batch_size, mlp_num, input_dim)
		# y = torch.bmm(x_unsqueeze, self.weight)
		y = torch.einsum('bmi,mio->bmo', x, self.weight) # (batch_size, mlp_num, output_dim)
		z = self.act(y).squeeze(1) + self.bias
		return z

class mlp_concat_preprocess(nn.Module):
	def __init__(self, mlp_num):
		super(mlp_concat_preprocess, self).__init__()
		self.mlp_num = mlp_num
	def forward(self, x): # (batch_size, input_dim)
		y = x.unsqueeze(1).repeat(1, self.mlp_num, 1) # (batch_size, mlp_num, input_dim)
		return y

class mlp_concat_afterprocess(nn.Module):
	def __init__(self):
		super(mlp_concat_afterprocess, self).__init__()
	def forward(self, x): # (batch_size, mlp_num, 1)
		y = x.squeeze(2)
		return y


def build_mlp_concat(layer_shape, hidden_activation, output_activation, mlp_num):
	'''Build net with for loop'''
	layer_list = [mlp_concat_preprocess(mlp_num)]
	weight_list = []
	bias_list = []
	act_list = []
	for j in range(len(layer_shape)-1):
		act = hidden_activation if j < len(layer_shape)-2 else output_activation
		linear_layer_list = []
		for i in range(mlp_num):
			linear_layer = nn.Linear(layer_shape[j], layer_shape[j+1])
			linear_layer_list.append(linear_layer)
		weight_concat = torch.stack(
			[linear_layer.weight for linear_layer in linear_layer_list], axis=0
		).permute(0, 2, 1) # (mlp_num, layer_shape[j], layer_shape[j+1])

		bias_concat = torch.stack([linear_layer.bias for linear_layer in linear_layer_list])
		layer_list.append(mlp_concat_layer(weight_concat, bias_concat, act=act()))
	layer_list.append(mlp_concat_afterprocess())
	return nn.Sequential(*layer_list)

class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, hid_shape, hidden_activation=nn.ReLU, output_activation=nn.ReLU):
		super(Actor, self).__init__()
		layers = [state_dim] + list(hid_shape)

		self.a_net = build_mlp(layers, hidden_activation, output_activation)
		self.mu_layer = nn.Linear(layers[-1], action_dim)
		self.log_std_layer = nn.Linear(layers[-1], action_dim)

		self.LOG_STD_MAX = 2
		self.LOG_STD_MIN = -20

	def forward(self, state, deterministic, with_logprob):
		'''Network with Enforcing Action Bounds'''
		net_out = self.a_net(state)
		mu = self.mu_layer(net_out)
		log_std = self.log_std_layer(net_out)
		log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)  #总感觉这里clamp不利于学习
		# we learn log_std rather than std, so that exp(log_std) is always > 0
		std = torch.exp(log_std)
		dist = Normal(mu, std)
		if deterministic: u = mu
		else: u = dist.rsample()

		# enforce action bound
		a = torch.tanh(u)
		if with_logprob:
			# Get probability density of logp_pi_a from probability density of u:
			# logp_pi_a = (dist.log_prob(u) - torch.log(1 - a.pow(2) + 1e-6)).sum(dim=1, keepdim=True)
			# Derive from the above equation. No a, thus no tanh(h), thus less gradient vanish and more stable.
			logp_pi_a = dist.log_prob(u).sum(axis=1, keepdim=True) - (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(axis=1, keepdim=True)
		else:
			logp_pi_a = None

		return a, logp_pi_a

class Double_Q_Critic(nn.Module):
	def __init__(self, state_dim, action_dim, hid_shape):
		super(Double_Q_Critic, self).__init__()
		layers = [state_dim + action_dim] + list(hid_shape) + [1]

		self.Q_1 = build_mlp(layers, nn.ReLU, nn.Identity)
		self.Q_2 = build_mlp(layers, nn.ReLU, nn.Identity)

	def forward(self, state, action):
		sa = torch.cat([state, action], 1)
		q1 = self.Q_1(sa)
		q2 = self.Q_2(sa)
		return q1, q2

class Multi_Q_Critic_Simple(nn.Module):
	def __init__(self, state_dim, action_dim, hid_shape, critic_num):
		super(Multi_Q_Critic_Simple, self).__init__()
		layers = [state_dim + action_dim] + list(hid_shape) + [1]
		self.Q_list = []
		self.critic_num = critic_num
		for Q_index in range(self.critic_num):
			self.add_module(
				"Q_%d"%Q_index, build_mlp(layers, nn.ReLU, nn.Identity)
			)
			self.Q_list.append(getattr(self, "Q_%d"%Q_index))
		
	def forward(self, state, action):
		sa = torch.cat([state, action], 1)
		q_list = []
		for Q_func in self.Q_list:
			q_list.append(Q_func(sa))
		return q_list
	
	def forward_indices(self, state, action, indices):
		sa = torch.cat([state, action], 1)
		q_list = []
		for index in indices:
			Q_func = self.Q_list[index]
			q_list.append(Q_func(sa))
		return q_list

class Multi_Q_Critic(nn.Module):
	def __init__(self, state_dim, action_dim, hid_shape, critic_num):
		super(Multi_Q_Critic, self).__init__()
		layers = [state_dim + action_dim] + list(hid_shape) + [1]
		
		self.Q_emsemble = build_mlp_concat(layers, nn.ReLU, nn.Identity, critic_num)
		self.critic_num = critic_num
	
	def forward(self, state, action):
		sa = torch.cat([state, action], 1) # (batch_size, state_dim + action_dim)
		q_vec = self.Q_emsemble(sa)
		return q_vec

def to_a_env(a, a_max):
	#from [-1,1] to [-max,max]
	return  a * a_max

def norm_a_env(act, a_max):
	#from [-max,max] to [-1,1]
	return  act / a_max

def evaluate(env, agent, a_max, turns = 3):
	total_scores = 0
	for j in range(turns):
		s, info = env.reset()
		done = False
		while not done:
			# Take deterministic actions at test time
			a = agent.select_action(s, deterministic=True)
			a_env = to_a_env(a, a_max)
			s_next, r, dw, tr, info = env.step(a_env)
			done = (dw or tr)

			total_scores += r
			s = s_next
	return int(total_scores/turns)

def str2bool(v):
	'''transfer str to bool for argparse'''
	if isinstance(v, bool):
		return v
	if v.lower() in ('yes', 'True', 'true', 'TRUE', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'False', 'false', 'FALSE', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')