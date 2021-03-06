# -*- coding: utf8 -*-

#  Policy Gradient Implementation
#  Adapted for Tensorflow
#  Other differences:
#  - Always choose the action with the highest probability
#  Source: http://rl-gym-doc.s3-website-us-west-2.amazonaws.com/mlss/lab2.html

import numpy as np
import os

import tensorflow as tf
from gym import wrappers

from agents.agent import Agent
from misc.utils import discount_rewards, preprocess_image, flatten
from misc.network_ops import conv2d, mu_sigma_layer
from misc.reporter import Reporter
from agents.env_runner import EnvRunner

class REINFORCE(Agent):
    """
    REINFORCE with baselines
    """
    def __init__(self, env, monitor_path, video=True, **usercfg):
        super(REINFORCE, self).__init__(**usercfg)
        self.env = wrappers.Monitor(env, monitor_path, force=True, video_callable=(None if video else False))
        self.env_runner = EnvRunner(self.env, self, usercfg)
        self.monitor_path = monitor_path
        # Default configuration. Can be overwritten using keyword arguments.
        self.config.update(dict(
            batch_update="timesteps",
            timesteps_per_batch=1000,
            n_iter=100,
            gamma=0.99,  # Discount past rewards by a percentage
            decay=0.9,  # Decay of RMSProp optimizer
            epsilon=1e-9,  # Epsilon of RMSProp optimizer
            learning_rate=0.05,
            n_hidden_units=20,
            repeat_n_actions=1,
            save_model=False
        ))
        self.config.update(usercfg)

        self.build_network()
        self.make_trainer()

        init = tf.global_variables_initializer()
        # Launch the graph.
        self.session = tf.Session()
        self.session.run(init)
        if self.config["save_model"]:
            tf.add_to_collection("action", self.action)
            tf.add_to_collection("states", self.states)
            self.saver = tf.train.Saver()
        self.rewards = tf.placeholder("float", name="Rewards")
        self.episode_lengths = tf.placeholder("float", name="Episode_lengths")
        summary_loss = tf.summary.scalar("Loss", self.summary_loss)
        summary_rewards = tf.summary.scalar("Rewards", self.rewards)
        summary_episode_lengths = tf.summary.scalar("Episode_lengths", self.episode_lengths)
        self.summary_op = tf.summary.merge([summary_loss, summary_rewards, summary_episode_lengths])
        self.writer = tf.summary.FileWriter(os.path.join(self.monitor_path, "task0"), self.session.graph)

    def choose_action(self, state):
        """Choose an action."""
        action = self.session.run([self.action], feed_dict={self.states: [state]})[0]
        return action

    def learn(self):
        """Run learning algorithm"""
        reporter = Reporter()
        config = self.config
        total_n_trajectories = 0
        for iteration in range(config["n_iter"]):
            # Collect trajectories until we get timesteps_per_batch total timesteps
            trajectories = self.env_runner.get_trajectories()
            total_n_trajectories += len(trajectories)
            all_state = np.concatenate([trajectory["state"] for trajectory in trajectories])
            # Compute discounted sums of rewards
            rets = [discount_rewards(trajectory["reward"], config["gamma"]) for trajectory in trajectories]
            max_len = max(len(ret) for ret in rets)
            padded_rets = [np.concatenate([ret, np.zeros(max_len - len(ret))]) for ret in rets]
            # Compute time-dependent baseline
            baseline = np.mean(padded_rets, axis=0)
            # Compute advantage function
            advs = [ret - baseline[:len(ret)] for ret in rets]
            all_action = np.concatenate([trajectory["action"] for trajectory in trajectories])
            all_adv = np.concatenate(advs)
            # Do policy gradient update step
            episode_rewards = np.array([trajectory["reward"].sum() for trajectory in trajectories])  # episode total rewards
            episode_lengths = np.array([len(trajectory["reward"]) for trajectory in trajectories])  # episode lengths
            summary, _ = self.session.run([self.summary_op, self.train], feed_dict={
                self.states: all_state,
                self.a_n: all_action,
                self.adv_n: all_adv,
                self.episode_lengths: np.mean(episode_lengths),
                self.rewards: np.mean(episode_rewards)
            })
            self.writer.add_summary(summary, iteration)
            self.writer.flush()

            reporter.print_iteration_stats(iteration, episode_rewards, episode_lengths, total_n_trajectories)
        if self.config["save_model"]:
            self.saver.save(self.session, os.path.join(self.monitor_path, "model"))

class REINFORCEDiscrete(REINFORCE):
    def __init__(self, env, monitor_path, video=True, **usercfg):
        super(REINFORCEDiscrete, self).__init__(env, monitor_path, video=video, **usercfg)

    def make_trainer(self):
        good_probabilities = tf.reduce_sum(tf.multiply(self.probs, tf.one_hot(tf.cast(self.a_n, tf.int32), self.env_runner.nA)), reduction_indices=[1])
        eligibility = tf.log(good_probabilities) * self.adv_n
        loss = -tf.reduce_sum(eligibility)
        self.summary_loss = loss
        optimizer = tf.train.RMSPropOptimizer(learning_rate=self.config["learning_rate"], decay=0.9, epsilon=1e-9)
        self.train = optimizer.minimize(loss)

    def build_network(self):
        # Symbolic variables for observation, action, and advantage
        self.states = tf.placeholder(tf.float32, [None, self.env_runner.nO], name="states")  # Observation
        self.a_n = tf.placeholder(tf.float32, name="a_n")  # Discrete action
        self.adv_n = tf.placeholder(tf.float32, name="adv_n")  # Advantage

        L1 = tf.contrib.layers.fully_connected(
            inputs=self.states,
            num_outputs=self.config["n_hidden_units"],
            activation_fn=tf.tanh,
            weights_initializer=tf.random_normal_initializer(),
            biases_initializer=tf.zeros_initializer())

        self.probs = tf.contrib.layers.fully_connected(
            inputs=L1,
            num_outputs=self.env_runner.nA,
            activation_fn=tf.nn.softmax,
            weights_initializer=tf.random_normal_initializer(),
            biases_initializer=tf.zeros_initializer())

        self.action = tf.squeeze(tf.multinomial(tf.log(self.probs), 1), name="action")

class REINFORCEDiscreteCNN(REINFORCEDiscrete):
    def __init__(self, env, monitor_path, video=True, **usercfg):
        usercfg["n_hidden_units"] = 200
        super(REINFORCEDiscreteCNN, self).__init__(env, monitor_path, video=video, **usercfg)
        self.config.update(usercfg)
        self.env_runner.state_preprocessor = preprocess_image

    def build_network(self):
        image_size = 80
        image_depth = 1  # aka nr. of feature maps. Eg 3 for RGB images. 1 here because we use grayscale images

        self.states = tf.placeholder(tf.float32, [None, image_size, image_size, image_depth], name="states")
        self.a_n = tf.placeholder(tf.float32, name="a_n")
        self.N = tf.placeholder(tf.int32, name="N")
        self.adv_n = tf.placeholder(tf.float32, name="adv_n")  # Advantage

        x = self.states
        # Convolution layers
        for i in range(4):
            x = tf.nn.elu(conv2d(x, 32, "l{}".format(i + 1), [3, 3], [2, 2]))

        # Flatten
        shape = x.get_shape().as_list()
        reshape = tf.reshape(x, [-1, shape[1] * shape[2] * shape[3]])  # -1 for the (unknown) batch size

        # Fully connected layer 1
        self.L3 = tf.contrib.layers.fully_connected(
            inputs=reshape,
            num_outputs=self.config["n_hidden_units"],
            activation_fn=tf.nn.relu,
            weights_initializer=tf.random_normal_initializer(stddev=0.01),
            biases_initializer=tf.zeros_initializer())

        # Fully connected layer 2
        self.probs = tf.contrib.layers.fully_connected(
            inputs=self.L3,
            num_outputs=self.env_runner.nA,
            activation_fn=tf.nn.softmax,
            weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.02),
            biases_initializer=tf.zeros_initializer())

        self.action = tf.squeeze(tf.multinomial(tf.log(self.probs), 1), name="action")

class REINFORCEDiscreteRNN(REINFORCEDiscrete):
    def __init__(self, env, monitor_path, video=True, **usercfg):
        super(REINFORCEDiscreteRNN, self).__init__(env, monitor_path, video=video, **usercfg)

    def build_network(self):
        self.rnn_state = None
        self.states = tf.placeholder(tf.float32, [None] + list(self.env.observation_space.shape), name="states")  # Observation
        # self.n_states = tf.placeholder(tf.float32, shape=[None], name="n_states")  # Observation
        self.a_n = tf.placeholder(tf.float32, name="a_n")  # Discrete action
        self.adv_n = tf.placeholder(tf.float32, name="adv_n")  # Advantage

        n_states = tf.shape(self.states)[:1]

        states = tf.expand_dims(flatten(self.states), [0])

        enc_cell = tf.contrib.rnn.GRUCell(self.config["n_hidden_units"])
        self.rnn_state_in = enc_cell.zero_state(1, tf.float32)
        L1, self.rnn_state_out = tf.nn.dynamic_rnn(cell=enc_cell,
                                                   inputs=states,
                                                   sequence_length=n_states,
                                                   initial_state=self.rnn_state_in,
                                                   dtype=tf.float32)
        self.probs = tf.contrib.layers.fully_connected(
            inputs=L1[0],
            num_outputs=self.env_runner.nA,
            activation_fn=tf.nn.softmax,
            weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.02),
            biases_initializer=tf.zeros_initializer())
        self.action = tf.squeeze(tf.multinomial(tf.log(self.probs), 1), name="action")

    def new_trajectory(self):
        self.rnn_state = None

    def choose_action(self, state):
        """Choose an action."""
        feed_dict = {
            self.states: [state]
        }
        if self.rnn_state is not None:
            feed_dict[self.rnn_state_in] = self.rnn_state
        action, self.rnn_state = self.session.run([self.action, self.rnn_state_out], feed_dict=feed_dict)
        return action

class REINFORCEDiscreteCNNRNN(REINFORCEDiscreteRNN):
    def __init__(self, env, monitor_path, video=True, **usercfg):
        super(REINFORCEDiscreteCNNRNN, self).__init__(env, monitor_path, video=video, **usercfg)
        self.env_runner.state_preprocessor = preprocess_image

    def build_network(self):
        self.rnn_state = None
        self.a_n = tf.placeholder(tf.float32, name="a_n")  # Discrete action
        self.adv_n = tf.placeholder(tf.float32, name="adv_n")  # Advantage

        image_size = 80
        image_depth = 1  # aka nr. of feature maps. Eg 3 for RGB images. 1 here because we use grayscale images

        self.states = tf.placeholder(tf.float32, [None, image_size, image_size, image_depth], name="states")
        self.N = tf.placeholder(tf.int32, name="N")

        x = self.states
        # Convolution layers
        for i in range(4):
            x = tf.nn.elu(conv2d(x, 32, "l{}".format(i + 1), [3, 3], [2, 2]))

        # Flatten
        shape = x.get_shape().as_list()
        reshape = tf.reshape(x, [-1, shape[1] * shape[2] * shape[3]])  # -1 for the (unknown) batch size

        reshape = tf.expand_dims(flatten(reshape), [0])
        self.enc_cell = tf.contrib.rnn.BasicLSTMCell(self.config["n_hidden_units"])
        self.rnn_state_in = self.enc_cell.zero_state(1, tf.float32)
        self.L3, self.rnn_state_out = tf.nn.dynamic_rnn(cell=self.enc_cell,
                                                        inputs=reshape,
                                                        initial_state=self.rnn_state_in,
                                                        dtype=tf.float32)

        self.probs = tf.contrib.layers.fully_connected(
            inputs=self.L3[0],
            num_outputs=self.env_runner.nA,
            activation_fn=tf.nn.softmax,
            weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.02),
            biases_initializer=tf.zeros_initializer())
        self.action = tf.squeeze(tf.multinomial(tf.log(self.probs), 1), name="action")

class REINFORCEContinuous(REINFORCE):
    def __init__(self, env, RNN, monitor_path, video=True, **usercfg):
        self.rnn = RNN
        super(REINFORCEContinuous, self).__init__(env, monitor_path, video=video, **usercfg)

    def build_network(self):
        if self.rnn:
            self.build_network_rnn()
        else:
            self.build_network_normal()

    def make_trainer(self):
        loss = -self.normal_dist.log_prob(self.a_n) * self.adv_n
        # Add cross entropy cost to encourage exploration
        loss -= 1e-1 * self.normal_dist.entropy()
        loss = tf.clip_by_value(loss, -1e10, 1e10)
        self.summary_loss = tf.reduce_mean(loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.config["learning_rate"])
        self.train = optimizer.minimize(loss, global_step=tf.contrib.framework.get_global_step())

    def build_network_normal(self):
        # Symbolic variables for observation, action, and advantage
        self.states = tf.placeholder(tf.float32, [None, self.env_runner.nO], name="states")  # Observation
        self.a_n = tf.placeholder(tf.float32, name="a_n")  # Continuous action
        self.adv_n = tf.placeholder(tf.float32, name="adv_n")  # Advantage

        L1 = tf.contrib.layers.fully_connected(
            inputs=self.states,
            num_outputs=self.config["n_hidden_units"],
            activation_fn=tf.tanh,
            weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.02),
            biases_initializer=tf.zeros_initializer())

        mu, sigma = mu_sigma_layer(L1, 1)

        self.normal_dist = tf.contrib.distributions.Normal(mu, sigma)
        self.action = self.normal_dist.sample(1)
        self.action = tf.clip_by_value(self.action, self.env.action_space.low[0], self.env.action_space.high[0])

    def build_network_rnn(self):
        self.states = tf.placeholder(tf.float32, [None] + list(self.env.observation_space.shape), name="states")  # Observation
        # self.n_states = tf.placeholder(tf.float32, shape=[None], name="n_states")  # Observation
        self.a_n = tf.placeholder(tf.float32, name="a_n")  # Discrete action
        self.adv_n = tf.placeholder(tf.float32, name="adv_n")  # Advantage

        n_states = tf.shape(self.states)[:1]

        states = tf.expand_dims(flatten(self.states), [0])

        enc_cell = tf.contrib.rnn.GRUCell(self.config["n_hidden_units"])
        L1, _ = tf.nn.dynamic_rnn(cell=enc_cell, inputs=states,
                                  sequence_length=n_states, dtype=tf.float32)

        L1 = L1[0]

        mu, sigma = mu_sigma_layer(L1, 1)

        self.normal_dist = tf.contrib.distributions.Normal(mu, sigma)
        self.action = self.normal_dist.sample(1)
        self.action = tf.clip_by_value(self.action, self.env.action_space.low[0], self.env.action_space.high[0])
