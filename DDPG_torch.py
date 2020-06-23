import os
import numpy as np
import torch as T
import torch.nn.functional as F

from DDPG_torch_utils import ReplayBuffer, OUActionNoise, CriticNetwork, ActorNetwork, OUNoise

class Agent():
    def __init__(self, agent_type, s_dim, a_dim, a_bound, memory_capacity, batch_size, gamma, lr_a, lr_c, theta, sigma, hidden_sizes=(400,300)):
        self.agent_type = agent_type
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.a_bound = a_bound
        self.gamma = gamma
        self.batch_size = batch_size
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.tau = 0.995 #0.001 

        self.memory = ReplayBuffer(memory_capacity, s_dim, a_bound)

        #self.noise = OUActionNoise(sigma, theta, mu=np.zeros(a_bound))
        self.OUD = OUNoise(a_dim, theta, sigma)

        self.actor = ActorNetwork(lr_a, s_dim, hidden_sizes[0], hidden_sizes[1],
                                n_actions=a_bound, name='actor')

        self.critic = CriticNetwork(lr_c, s_dim, hidden_sizes[0], hidden_sizes[1],
                                n_actions=a_bound, name='critic')

        self.target_actor = ActorNetwork(lr_a, s_dim, hidden_sizes[0], hidden_sizes[1],
                                n_actions=a_bound, name='target_actor')

        self.target_critic = CriticNetwork(lr_c, s_dim, hidden_sizes[0], hidden_sizes[1],
                                n_actions=a_bound, name='target_critic')

        self.update_network_parameters(tau=1)

    def choose_action(self, observation, deterministic=False, random_action=False):
        if not random_action:
            self.actor.eval()
            state = T.tensor([observation], dtype=T.float).to(self.actor.device)
            mu = self.actor.forward(state).to(self.actor.device)
            mu_prime = mu + T.tensor(self.OUD.sample(), #.noise()
                                                dtype=T.float).to(self.actor.device)
            self.actor.train()
            a = mu_prime.cpu().detach().numpy()[0]
        else:
            a = np.array( (2.0*self.a_bound)*np.random.rand(self.a_dim) - self.a_bound).reshape((self.a_dim,) )

        return np.clip(a, -self.a_bound, self.a_bound)

    def remember(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        states, actions, rewards, states_, done = \
                self.memory.sample_buffer(self.batch_size)

        states = T.tensor(states, dtype=T.float).to(self.actor.device)
        states_ = T.tensor(states_, dtype=T.float).to(self.actor.device)
        actions = T.tensor(actions, dtype=T.float).to(self.actor.device)
        rewards = T.tensor(rewards, dtype=T.float).to(self.actor.device)
        done = T.tensor(done).to(self.actor.device)

        target_actions = self.target_actor.forward(states_)
        critic_value_ = self.target_critic.forward(states_, target_actions)
        critic_value = self.critic.forward(states, actions)

        critic_value_[done] = 0.0
        critic_value_ = critic_value_.view(-1)

        target = rewards + self.gamma*critic_value_
        target = target.view(self.batch_size, 1)

        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()

        self.actor.optimizer.zero_grad()
        actor_loss = -self.critic.forward(states, self.actor.forward(states))
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_state_dict = dict(target_critic_params)
        target_actor_state_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                                (1-tau)*target_critic_state_dict[name].clone()

        for name in actor_state_dict:
             actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                                 (1-tau)*target_actor_state_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)
        #self.target_critic.load_state_dict(critic_state_dict, strict=False)
        #self.target_actor.load_state_dict(actor_state_dict, strict=False)
