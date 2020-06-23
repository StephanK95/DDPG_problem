# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 06:07:41 2019

@author: Alexander
"""

seed = 7

import numpy as np
np.random.seed(seed)

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.set_random_seed(seed)

import DDPG_utils as core
#from SAC_utils import get_vars

import pickle
import lzma

class DDPG(object):
    def __init__(self, agent_type, sess, s_dim, a_dim, a_bound, memory_capacity, batch_size, gamma, lr_a, lr_c, thetas, sigmas, hidden_sizes=(400,300)):
        self.agent_type = agent_type
        self.sess = sess
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.a_bound = a_bound
        self.memory_capacity = memory_capacity
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr_a = lr_a
        self.lr_c = lr_c

        self.duration_getting_batch = []
        self.duration_gradient_qfunction = []
        self.duration_gradient_pi = []
        self.duration_polyak = []
        
        # Inputs to computation graph
        self.x_ph, self.a_ph, self.x2_ph, self.r_ph, self.d_ph = core.placeholders(s_dim, a_dim, s_dim, None, None)
    
        # Main outputs from computation graph
        with tf.variable_scope('main'):
            self.pi, self.q, q_pi = core.mlp_actor_critic(self.x_ph, self.a_ph, 
                                                hidden_sizes=hidden_sizes, 
                                                activation=tf.nn.relu, 
                                                output_activation=tf.tanh, 
                                                a_bound=a_bound)
        
        # Target networks
        with tf.variable_scope('target'):
            # Note that the action placeholder going to actor_critic here is 
            # irrelevant, because we only need q_targ(s, pi_targ(s)).
            pi_targ, _, q_pi_targ  = core.mlp_actor_critic(self.x2_ph, self.a_ph,
                                                           hidden_sizes=hidden_sizes, 
                                                           activation=tf.nn.relu, 
                                                           output_activation=tf.tanh, 
                                                           a_bound=a_bound)
    
        # Experience buffer
        self.replay_buffer = ReplayBuffer(obs_dim=s_dim, act_dim=a_dim, size=memory_capacity)
        
        # Exploration strategy
        self.OUD = OUNoise(a_dim, thetas, sigmas)
    
        # Count variables
        var_counts = tuple(core.count_vars(scope) for scope in ['main/pi', 'main/q', 'main'])
        print('\nNumber of parameters: \t pi: %d, \t q: %d, \t total: %d\n'%var_counts)
    
        # Bellman backup for Q function
        backup = tf.stop_gradient(self.r_ph + gamma*(1-self.d_ph)*q_pi_targ)
    
        # DDPG losses
        self.pi_loss = -tf.reduce_mean(q_pi)
        self.q_loss = tf.reduce_mean((self.q-backup)**2)
    
        # Separate train ops for pi, q
        pi_optimizer = tf.train.AdamOptimizer(learning_rate=lr_a)
        q_optimizer = tf.train.AdamOptimizer(learning_rate=lr_c)
        self.train_pi_op = pi_optimizer.minimize(self.pi_loss, var_list=core.get_vars('main/pi'))
        self.train_q_op = q_optimizer.minimize(self.q_loss, var_list=core.get_vars('main/q'))
    
        # Polyak averaging for target variables
        polyak=0.995
        self.target_update = tf.group([tf.assign(v_targ, polyak*v_targ + (1-polyak)*v_main)
                                  for v_main, v_targ in zip(core.get_vars('main'), core.get_vars('target'))])
    
        # Initializing targets to match main variables
        target_init = tf.group([tf.assign(v_targ, v_main)
                                  for v_main, v_targ in zip(core.get_vars('main'), core.get_vars('target'))])
    
        #self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(target_init)
        #self.sess.graph.finalize()
        
        # after graph is built: create TensorFlow Saver to Save and Restore Agent
        self.saver = tf.train.Saver()
    
    def get_action(self, o, deterministic=False, random_action=False):
        if not random_action:
            a = self.sess.run(self.pi, feed_dict={self.x_ph: o.reshape(1,-1)})[0]
            if not deterministic:
                a += self.OUD.sample()
        else:
            a = np.array( (2.0*self.a_bound)*np.random.rand(self.a_dim) - self.a_bound).reshape((self.a_dim,) )
        
        return np.clip(a, -self.a_bound, self.a_bound)
    
    def store_transition(self, s, a, r, s_, d):
        # Store experience to replay buffer
        self.replay_buffer.store(s, a, r, s_, d)
    
    def train(self ,j):    
        batch = self.replay_buffer.sample_batch(self.batch_size)
        feed_dict = {self.x_ph: batch['obs1'],
                     self.x2_ph: batch['obs2'],
                     self.a_ph: batch['acts'],
                     self.r_ph: batch['rews'],
                     self.d_ph: batch['done']
                    }

        # Q-learning update
        self.sess.run([self.q_loss, self.q, self.train_q_op], feed_dict) # outs = 

        # Policy update
        self.sess.run([self.pi_loss, self.train_pi_op, self.target_update], feed_dict) # outs = 
    
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
            

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])
        
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