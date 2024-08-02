from utils import Double_Q_Critic, DRL_Continuous
import numpy as np
import torch
import torch.nn.functional as F

class SAC_countinuous(DRL_Continuous):
	def __init__(self, **kwargs):
		super().__init__(**kwargs) # build actor, replay_buffer
		"""build q_critic"""
		self.q_critic = Double_Q_Critic(
			self.state_dim, self.action_dim, (self.net_width,self.net_width)
		).to(self.device)
		
		self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=self.c_lr)
		
		"""build q_critic_target"""
		import copy
		self.q_critic_target = copy.deepcopy(self.q_critic)
		# freeze q_critic_target params
		for p in self.q_critic_target.parameters(): p.requires_grad = False
			# only update via polyak averaging

	def train(self):
		"""sample a mini-batch from replay_buffer"""
		s, a, r, s_next, dw = self.replay_buffer.sample(self.batch_size)

		"""update q_critic params"""
		with torch.no_grad():
			a_next, log_pi_a_next = self.actor(s_next, is_deterministic=False, with_logprob=True)
			target_Q1, target_Q2 = self.q_critic_target(s_next, a_next)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = r + (~dw) * self.gamma * (target_Q - self.alpha * log_pi_a_next) #Dead or Done is tackled by Randombuffer

		# get current Q estimates
		current_Q1, current_Q2 = self.q_critic(s, a)

		q_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
		self.q_critic_optimizer.zero_grad()
		q_loss.backward()
		self.q_critic_optimizer.step()

		# update actor param
		for params in self.q_critic.parameters(): params.requires_grad = False
			# freeze q_critic param to avoid waster computation
		a, log_pi_a = self.actor(s, is_deterministic=False, with_logprob=True)
			# estimate entropy of pi(·|s): sample a from pi(·|s), and calculate -log(pi(a|s))
		
		current_Q1, current_Q2 = self.q_critic(s, a)
		Q = torch.min(current_Q1, current_Q2)

		a_loss = (self.alpha * log_pi_a - Q).mean() # self.alpha * log_pi_a: entropy term in SAC.
		self.actor_optimizer.zero_grad()
		a_loss.backward()
		self.actor_optimizer.step()

		for params in self.q_critic.parameters(): params.requires_grad = True

		# update actor alpha
		if self.adaptive_alpha:
			# We learn log_alpha instead of alpha to ensure alpha > 0.0
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