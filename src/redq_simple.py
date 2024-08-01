from utils import Actor, Double_Q_Critic, Multi_Q_Critic_Simple
import torch.nn.functional as F
import numpy as np
import torch
import copy
import random

class REDQ_continuous():
	def __init__(self, **kwargs):
		# Init hyperparameters for agent, just like "self.gamma = opt.gamma, self.lambd = opt.lambd, ..."
		self.__dict__.update(kwargs)
		self.tau = 0.005

		self.actor = Actor(self.state_dim, self.action_dim, (self.net_width,self.net_width)).to(self.device)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.a_lr)

		self.q_critic = Multi_Q_Critic_Simple(
			self.state_dim, self.action_dim, (self.net_width,self.net_width), self.critic_num
		).to(self.device)
		self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=self.c_lr)
		self.q_critic_target = copy.deepcopy(self.q_critic)
		# Freeze target networks with respect to optimizers (only update via polyak averaging)
		for p in self.q_critic_target.parameters():
			p.requires_grad = False

		self.replay_buffer = ReplayBuffer(self.state_dim, self.action_dim, max_size=int(1e6), device=self.device)

		if self.adaptive_alpha:
			# Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
			self.target_entropy = torch.tensor(-self.action_dim, dtype=float, requires_grad=True, device=self.device)
			# We learn log_alpha instead of alpha to ensure alpha>0
			self.log_alpha = torch.tensor(np.log(self.alpha), dtype=float, requires_grad=True, device=self.device)
			self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.c_lr)

	def select_action(self, state, deterministic):
		# only used when interact with the env
		with torch.no_grad():
			state = torch.FloatTensor(state[np.newaxis,:]).to(self.device)
			a, _ = self.actor(state, deterministic, with_logprob=False)
		return a.cpu().numpy()[0]

	def train(self,):
		s, a, r, s_next, dw = self.replay_buffer.sample(self.batch_size)

		#----------------------------- ↓↓↓↓↓ Update Q Net ↓↓↓↓↓ ------------------------------#
		with torch.no_grad():
			a_next, log_pi_a_next = self.actor(s_next, deterministic=False, with_logprob=True)
			
			# random select from Q ensembles, for each sample in batch
			Q_indices = random.sample(range(self.critic_num), self.critic_num_select)
			target_Q_list = self.q_critic_target.forward_indices(s_next, a_next, Q_indices) # (batch_size, critic_num)
			target_Q_vec = torch.cat(target_Q_list, dim=1) # (batch_size, critic_num_select)
			target_Q, _ = torch.min(target_Q_vec, dim=1, keepdim=True) # (batch_size, 1)
			target_Q = r + (~dw) * self.gamma * (target_Q - self.alpha * log_pi_a_next) # Dead or Done is tackled by Randombuffer

		# get current Q estimates
		current_Q_list = self.q_critic(s, a) # (batch_size, critic_num)
		q_loss = sum(F.mse_loss(current_Q, target_Q) for current_Q in current_Q_list)
		self.q_critic_optimizer.zero_grad()
		q_loss.backward()
		self.q_critic_optimizer.step()

		"""update actor net"""
		# Freeze critic so you don't waste computational effort computing gradients for them when update actor
		for params in self.q_critic.parameters(): params.requires_grad = False

		a, log_pi_a = self.actor(s, deterministic=False, with_logprob=True)
			# estimate entropy of pi(·|s): sample a from pi(·|s), and calculate -log(pi(a|s))
		target_Q_list = self.q_critic(s, a)
		target_Q_vec = torch.cat(target_Q_list, dim=1) # (batch_size, critic_num)
		Q = torch.mean(target_Q_vec, axis=1, keepdim=True) # use mean of all Q_func in ensemble to update pi

		a_loss = (self.alpha * log_pi_a - Q).mean() # SAC中的\pi(·|s)的entropy就是早这里实现的
		self.actor_optimizer.zero_grad()
		a_loss.backward()
		self.actor_optimizer.step()

		for params in self.q_critic.parameters(): params.requires_grad = True

		#----------------------------- ↓↓↓↓↓ Update alpha ↓↓↓↓↓ ------------------------------#
		if self.adaptive_alpha:
			# We learn log_alpha instead of alpha to ensure alpha>0
			alpha_loss = -(self.log_alpha * (log_pi_a + self.target_entropy).detach()).mean()
			self.alpha_optim.zero_grad()
			alpha_loss.backward()
			self.alpha_optim.step()
			self.alpha = self.log_alpha.exp()

		#----------------------------- ↓↓↓↓↓ Update Target Net ↓↓↓↓↓ ------------------------------#
		for param, target_param in zip(self.q_critic.parameters(), self.q_critic_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

	def save(self, timestep, save_dir_path):
		torch.save(self.actor.state_dict(), save_dir_path + "model/{}-actor.pth".format(timestep))
		torch.save(self.q_critic.state_dict(), save_dir_path + "model/{}-q_critic.pth".format(timestep))

	def load(self, env_name, timestep):
		self.actor.load_state_dict(torch.load("./model/{}_actor{}.pth".format(env_name, timestep), map_location=self.device))
		self.q_critic.load_state_dict(torch.load("./model/{}_q_critic{}.pth".format(env_name, timestep), map_location=self.device))

class ReplayBuffer():
	def __init__(self, state_dim, action_dim, max_size, device):
		self.max_size = max_size
		self.device = device
		self.ptr = 0
		self.size = 0

		self.s = torch.zeros((max_size, state_dim) ,dtype=torch.float,device=self.device)
		self.a = torch.zeros((max_size, action_dim) ,dtype=torch.float,device=self.device)
		self.r = torch.zeros((max_size, 1) ,dtype=torch.float,device=self.device)
		self.s_next = torch.zeros((max_size, state_dim) ,dtype=torch.float,device=self.device)
		self.dw = torch.zeros((max_size, 1) ,dtype=torch.bool,device=self.device)

	def add(self, s, a, r, s_next, dw):
		self.s[self.ptr] = torch.from_numpy(s).to(self.device)
		self.a[self.ptr] = torch.from_numpy(a).to(self.device) # Note that a is numpy.array
		self.r[self.ptr] = r
		self.s_next[self.ptr] = torch.from_numpy(s_next).to(self.device)
		self.dw[self.ptr] = dw

		self.ptr = (self.ptr + 1) % self.max_size # overwrite when overflow
		self.size = min(self.size + 1, self.max_size)

	def sample(self, batch_size):
		ind = torch.randint(0, self.size, device=self.device, size=(batch_size,))
		return self.s[ind], self.a[ind], self.r[ind], self.s_next[ind], self.dw[ind]
	



