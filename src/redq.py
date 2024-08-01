from utils import Actor, Multi_Q_Critic_Simple, ReplayBuffer, DRL_Continuous
import torch.nn.functional as F
import numpy as np
import torch
import random

class REDQ_continuous(DRL_Continuous):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		"""build q_critic"""
		self.q_critic = Multi_Q_Critic_Simple(
			self.state_dim, self.action_dim, (self.net_width,self.net_width), self.critic_num
		).to(self.device)
		
		self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=self.c_lr)
		
		"""build q_critic_target"""
		import copy
		self.q_critic_target = copy.deepcopy(self.q_critic)
		# freeze q_critic_target params
		for p in self.q_critic_target.parameters(): p.requires_grad = False
			# only update via polyak averaging

	def train(self):
		"""sample a batch from replay_buffer"""
		s, a, r, s_next, dw = self.replay_buffer.sample(self.batch_size)

		"""calculate target Q value"""
		with torch.no_grad():
			a_next, log_pi_a_next = self.actor(s_next, is_deterministic=False, with_logprob=True)
			
			# random select from Q ensembles, for each sample in batch
			Q_indices = random.sample(range(self.critic_num), self.critic_num_select)
			target_Q_list = self.q_critic_target.forward_indices(s_next, a_next, Q_indices) # (batch_size, critic_num)
			target_Q_vec = torch.cat(target_Q_list, dim=1) # (batch_size, critic_num_select)
			target_Q, _ = torch.min(target_Q_vec, dim=1, keepdim=True) # (batch_size, 1)
			target_Q = r + (~dw) * self.gamma * (target_Q - self.alpha * log_pi_a_next) # Dead or Done is tackled by Randombuffer

		"""update q_critic params"""
		current_Q_list = self.q_critic(s, a) # (batch_size, critic_num)
		q_loss = sum(F.mse_loss(current_Q, target_Q) for current_Q in current_Q_list)
		self.q_critic_optimizer.zero_grad()
		q_loss.backward()
		self.q_critic_optimizer.step()

		"""update actor param"""
		for params in self.q_critic.parameters(): params.requires_grad = False
			# freeze q_critic param to avoid waster computation

		a, log_pi_a = self.actor(s, is_deterministic=False, with_logprob=True)
			# estimate entropy of pi(·|s): sample a from pi(·|s), and calculate -log(pi(a|s))
		target_Q_list = self.q_critic(s, a)
		target_Q_vec = torch.cat(target_Q_list, dim=1) # (batch_size, critic_num)
		Q = torch.mean(target_Q_vec, axis=1, keepdim=True) # use mean of all Q_func in ensemble to update pi

		a_loss = (self.alpha * log_pi_a - Q).mean() # self.alpha * log_pi_a: entropy term in SAC.
		self.actor_optimizer.zero_grad()
		a_loss.backward()
		self.actor_optimizer.step()
		
		# unfreeze critic param
		for params in self.q_critic.parameters(): params.requires_grad = True

		if self.adaptive_alpha:
			# We learn log_alpha instead of alpha to ensure alpha>0
			alpha_loss = -(self.log_alpha * (log_pi_a + self.target_entropy).detach()).mean()
			self.alpha_optim.zero_grad()
			alpha_loss.backward()
			self.alpha_optim.step()
			self.alpha = self.log_alpha.exp()
		
		# update q_critic_target param
		for param, target_param in zip(self.q_critic.parameters(), self.q_critic_target.parameters()):
			target_param.data.copy_(
				self.critic_target_update_ratio * param.data +
				(1.0 - self.critic_target_update_ratio) * target_param.data
			)

	def train_2(self,): # using concatenated mlp
		s, a, r, s_next, dw = self.replay_buffer.sample(self.batch_size)

		"""update q_critic params"""
		with torch.no_grad():
			a_next, log_pi_a_next = self.actor(s_next, deterministic=False, with_logprob=True)
			target_Q_vec = self.q_critic_target(s_next, a_next) # (batch_size, critic_num)
			batch_size = target_Q_vec.size(0)

			# random select from Q ensembles, for each sample in batch
			indices = torch.randint(0, self.critic_num, (batch_size, self.critic_num_select)).to(self.device)
			target_Q_selected = target_Q_vec.gather(1, indices)  # (batch_size, critic_num_select)
			target_Q, _ = torch.min(target_Q_selected, dim=1, keepdim=True) # (batch_size, 1)
			target_Q = r + (~dw) * self.gamma * (target_Q - self.alpha * log_pi_a_next) # (batch_size, 1)
				# Dead or Done is tackled by Randombuffer
		# get current Q estimates
		current_Q_vec = self.q_critic(s, a) # (batch_size, critic_num)
		# q_loss = F.mse_loss(current_Q_vec, target_Q) * self.critic_num
		q_loss = torch.mean((current_Q_vec - target_Q) ** 2) * self.critic_num
		self.q_critic_optimizer.zero_grad()
		q_loss.backward()
		self.q_critic_optimizer.step()

		"""update actor param"""
		for params in self.q_critic.parameters(): params.requires_grad = False
			# freeze q_critic param to avoid waster computation

		a, log_pi_a = self.actor(s, deterministic=False, with_logprob=True) # 如何估计1个概率的entropy？采1个样, 然后求-log值
		target_Q_vec = self.q_critic(s, a)
		Q, _ = torch.min(target_Q_vec, dim=1, keepdim=True)

		a_loss = (self.alpha * log_pi_a - Q).mean() # self.alpha * log_pi_a: entropy term in SAC.
		self.actor_optimizer.zero_grad()
		a_loss.backward()
		self.actor_optimizer.step()
		
		# unfreeze critic param
		for params in self.q_critic.parameters(): params.requires_grad = True

		if self.adaptive_alpha:
			# We learn log_alpha instead of alpha to ensure alpha>0
			alpha_loss = -(self.log_alpha * (log_pi_a + self.target_entropy).detach()).mean()
			self.alpha_optim.zero_grad()
			alpha_loss.backward()
			self.alpha_optim.step()
			self.alpha = self.log_alpha.exp()

		# update target critic networks(target Q functions)
		for param, target_param in zip(self.q_critic.parameters(), self.q_critic_target.parameters()):
			target_param.data.copy_(
				self.critic_target_update_ratio * param.data +
				(1.0 - self.critic_target_update_ratio) * target_param.data
			)






