from copy import deepcopy
import numpy as np
import torch
from torch.optim import Adam
#import gym
import time
import DDPG_torch_open_ai_core as core
#from spinup.utils.logx import EpochLogger
import pickle
import lzma


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DDPG agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, theta, sigma,mu=None):
        """Initialize parameters and noise process."""
        self.size = size
        self.mu = mu if mu is not None else np.zeros(self.size)
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.size) * self.mu
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = self.mu

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

class Agent(object):
    def __init__(self, agent_type, s_dim, a_dim, a_bound, memory_capacity, batch_size, gamma, lr_a, lr_c, thetas, sigmas, hidden_sizes=(400,300)):
        self.agent_type = agent_type
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.a_bound = a_bound
        self.memory_capacity = memory_capacity
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.polyak = 0.995

        self.ac = core.MLPActorCritic(s_dim, a_dim, a_bound, hidden_sizes=hidden_sizes)
        self.ac_targ = deepcopy(self.ac)

         # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False

        # Experience buffer
        self.replay_buffer = ReplayBuffer(obs_dim=s_dim, act_dim=a_dim, size=memory_capacity)

        # Exploration strategy
        self.OUD = OUNoise(a_dim, thetas, sigmas)

        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=self.lr_a)
        self.q_optimizer = Adam(self.ac.q.parameters(), lr=self.lr_c) 

        # Count variables (protip: try to get a feel for how different size networks behave!)
        var_counts = tuple(core.count_vars(module) for module in [self.ac.pi, self.ac.q])
        print('\nNumber of parameters: \t pi: %d, \t q: %d\n'%var_counts)

        self.duration_getting_batch = []
        self.duration_gradient_qfunction = []
        self.duration_gradient_pi = []
        self.duration_polyak = []

    # Set up function for computing DDPG Q-loss
    def compute_loss_q(self, data):

        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q = self.ac.q(o,a)

        # Bellman backup for Q function
        with torch.no_grad():
            q_pi_targ = self.ac_targ.q(o2, self.ac_targ.pi(o2))
            backup = r + self.gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q = ((q - backup)**2).mean()

        # Useful info for logging
        loss_info = dict(QVals=q.detach().numpy())

        return loss_q, loss_info

    # Set up function for computing DDPG pi loss
    def compute_loss_pi(self, data):

        o = data['obs']
        q_pi = self.ac.q(o, self.ac.pi(o))
        return -q_pi.mean()  

    def update(self):

        beginn_update_time = time.time()

        batch = self.replay_buffer.sample_batch(self.batch_size)
        data = batch

        self.duration_getting_batch.append(time.time() - beginn_update_time)

        self.q_optimizer.zero_grad()
        loss_q, loss_info = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()

        self.duration_gradient_qfunction.append(time.time() - beginn_update_time)

        # Freeze Q-network so you don't waste computational effort 
        # computing gradients for it during the policy learning step.
        for p in self.ac.q.parameters():
            p.requires_grad = False

        #print("Beginn gradientdescent step pi-Funktion")

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        self.duration_gradient_pi.append(time.time() - beginn_update_time)

        # Unfreeze Q-network so you can optimize it at next DDPG step.
        for p in self.ac.q.parameters():
            p.requires_grad = True
        
        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)
        
        self.duration_polyak.append(time.time() - beginn_update_time)
    
    def get_action(self, o, deterministic=False, random_action=False):
        if not random_action:
            a = self.ac.act(torch.as_tensor(o, dtype=torch.float32))
            if not deterministic:
                a += self.OUD.sample()
        else:
            a = np.array( (2.0*self.a_bound)*np.random.rand(self.a_dim) - self.a_bound).reshape((self.a_dim,) )

        return np.clip(a, -self.a_bound, self.a_bound)

    def store_transition(self, s, a, r, s_, d):
        # Store experience to replay buffer
        self.replay_buffer.store(s, a, r, s_, d)

    def save_to_disk(self, checkpoint_dir):
        self.saver.save(self.sess, checkpoint_dir+'agent.ckpt', write_meta_graph=True)
        #with open(checkpoint_dir+'replay_buffer.pkl', 'wb') as output_file:
        #    pickle.dump(self.replay_buffer, output_file, pickle.HIGHEST_PROTOCOL)
        with lzma.open(checkpoint_dir+'replay_buffer.pkl.xz', 'wb') as output_file:
            pickle.dump(self.replay_buffer, output_file, pickle.HIGHEST_PROTOCOL)
        
    def restore_from_disk(self, checkpoint_dir, checkpoint_with_buffer=True):
        self.saver.restore(self.sess, tf.train.latest_checkpoint(checkpoint_dir))
        print('Restored Pretrained Agent.')
    
        if checkpoint_with_buffer:
            try:
                with lzma.open(checkpoint_dir+'replay_buffer.pkl.xz', 'rb') as input_file:
                    self.replay_buffer = pickle.load(input_file)
            except:
                with open(checkpoint_dir+'replay_buffer.pkl', 'rb') as input_file:
                    self.replay_buffer = pickle.load(input_file)
            print('Restored Replaybuffer. Size:', self.replay_buffer.size)
        else:
            print('Replaybuffer NOT Restored.')