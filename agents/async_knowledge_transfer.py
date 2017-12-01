# -*- coding: utf8 -*-

import os
import numpy as np
import tensorflow as tf
import logging
from threading import Thread
import signal
import queue

from agents.agent import Agent
from agents.env_runner import EnvRunner
from agents.a3c import RunnerThread
from misc.utils import discount_rewards
from misc.reporter import Reporter
from agents.knowledge_transfer import TaskPolicy
from misc.network_ops import conv2d, flatten

class AKTThread(Thread):
    """Asynchronous knowledge transfer learner thread. Used to learn using one specific variation of a task."""
    def __init__(self, master, env, task_id, n_iter, start_at_iter=0):
        super(AKTThread, self).__init__()
        self.master = master
        self.env = env
        self.config = self.master.config
        self.task_id = task_id
        self.nA = env.action_space.n
        self.n_iter = n_iter
        self.start_at_iter = start_at_iter
        self.add_accum_grad = None  # To be filled in later

        # Only used (and overwritten) by agents that use an RNN
        self.initial_features = None

        with tf.variable_scope("task{}".format(self.task_id)):
            self.build_networks()
        self.states = self.master.states
        self.session = self.master.session
        self.task_runner = EnvRunner(env, TaskPolicy(self.action, self), self.master.config)

        # Write the summary of each task in a different directory
        self.writer = tf.summary.FileWriter(os.path.join(self.master.monitor_path, "task" + str(self.task_id)), self.master.session.graph)

        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.config["learning_rate"], decay=self.config["decay"], epsilon=self.config["epsilon"])
        self.runner = RunnerThread(self.env, self, 20, task_id == 0 and self.master.video)

    def build_networks(self):
        self.sparse_representation = tf.Variable(tf.truncated_normal([self.master.config["n_sparse_units"], self.nA], mean=0.0, stddev=0.02))
        self.sparse_representations = [self.sparse_representation]

        self.probs = tf.nn.softmax(tf.matmul(self.master.L1, tf.matmul(self.master.knowledge_base, self.sparse_representation)))

        self.action = tf.squeeze(tf.multinomial(tf.log(self.probs), 1), name="action")

        good_probabilities = tf.reduce_sum(tf.multiply(self.probs, tf.one_hot(tf.cast(self.master.action_taken, tf.int32), self.nA)), reduction_indices=[1])
        eligibility = tf.log(good_probabilities + 1e-10) * self.master.advantage
        self.loss = -tf.reduce_sum(eligibility)

    def run(self):
        """Run the appropriate learning algorithm."""
        if self.master.learning_method == "REINFORCE":
            self.learn_reinforce()
        else:
            self.learn_Karpathy()

    def learn_reinforce(self):
        """Learn using updates like in the REINFORCE algorithm."""
        reporter = Reporter()
        total_n_trajectories = 0
        iteration = self.start_at_iter
        while iteration < self.n_iter and not self.master.stop_requested:
            iteration += 1
            # Collect trajectories until we get timesteps_per_batch total timesteps
            trajectories = self.task_runner.get_trajectories()
            total_n_trajectories += len(trajectories)
            all_state = np.concatenate([trajectory["state"] for trajectory in trajectories])
            # Compute discounted sums of rewards
            rets = [discount_rewards(trajectory["reward"], self.config["gamma"]) for trajectory in trajectories]
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
            results = self.master.session.run([self.loss, self.train_op], feed_dict={
                self.master.states: all_state,
                self.master.action_taken: all_action,
                self.master.advantage: all_adv
            })
            print("Task:", self.task_id)
            reporter.print_iteration_stats(iteration, episode_rewards, episode_lengths, total_n_trajectories)
            summary = self.master.session.run([self.master.summary_op], feed_dict={
                self.master.loss: results[0],
                self.master.reward: np.mean(episode_rewards),
                self.master.episode_length: np.mean(episode_lengths)
            })
            self.writer.add_summary(summary[0], iteration)
            self.writer.flush()

    def learn_Karpathy(self):
        """Learn using updates like in the Karpathy algorithm."""
        iteration = self.start_at_iter
        while iteration < self.n_iter and not self.master.stop_requested:  # Keep executing episodes until the master requests a stop (e.g. using SIGINT)
            iteration += 1
            trajectory = self.task_runner.get_trajectory()
            reward = sum(trajectory["reward"])
            action_taken = trajectory["action"]

            discounted_episode_rewards = discount_rewards(trajectory["reward"], self.config["gamma"])
            # standardize
            discounted_episode_rewards -= np.mean(discounted_episode_rewards)
            std = np.std(discounted_episode_rewards)
            std = std if std > 0 else 1
            discounted_episode_rewards /= std
            feedback = discounted_episode_rewards

            results = self.master.session.run([self.loss, self.train_op], feed_dict={
                self.master.states: trajectory["state"],
                self.master.action_taken: action_taken,
                self.master.advantage: feedback
            })
            results = self.master.session.run([self.master.summary_op], feed_dict={
                self.master.loss: results[0],
                self.master.reward: reward,
                self.master.episode_length: trajectory["steps"]
            })
            self.writer.add_summary(results[0], iteration)
            self.writer.flush()

class AKTThreadDiscreteCNNRNN(AKTThread):
    """A3CThread for a discrete action space."""
    def __init__(self, master, env, thread_id, n_iter):
        super(AKTThreadDiscreteCNNRNN, self).__init__(master, env, thread_id, n_iter)
        self.rnn_state = None
        self.initial_features = self.state_init
        self.actor_states = self.critic_states = self.master.states

    def build_networks(self):
        self.adv = tf.placeholder(tf.float32, name="advantage")
        self.actions_taken = tf.placeholder(tf.float32, name="actions_taken")
        self.r = tf.placeholder(tf.float32, [None], name="r")

        lstm_size = 256
        self.enc_cell = tf.contrib.rnn.BasicLSTMCell(lstm_size)
        lstm_state_size = self.enc_cell.state_size
        c_init = np.zeros((1, lstm_state_size.c), np.float32)
        h_init = np.zeros((1, lstm_state_size.h), np.float32)
        self.state_init = [c_init, h_init]
        self.rnn_state_in = self.enc_cell.zero_state(1, tf.float32)
        tf.add_to_collection("rnn_state_in_c", self.rnn_state_in.c)
        tf.add_to_collection("rnn_state_in_h", self.rnn_state_in.h)
        L3, self.rnn_state_out = tf.nn.dynamic_rnn(cell=self.enc_cell,
                                                   inputs=self.master.L1,
                                                   initial_state=self.rnn_state_in,
                                                   dtype=tf.float32)
        tf.add_to_collection("rnn_state_out_c", self.rnn_state_out.c)
        tf.add_to_collection("rnn_state_out_h", self.rnn_state_out.h)
        self.L2 = tf.reshape(L3, [-1, lstm_size])

        self.sparse_representation_action = tf.Variable(tf.truncated_normal([self.master.config["n_sparse_units"], self.nA], mean=0.0, stddev=0.02))
        self.logits = tf.matmul(self.L2, tf.matmul(self.master.knowledge_base, self.sparse_representation_action), name="logits")

        self.sparse_representation_value = tf.Variable(tf.truncated_normal([self.master.config["n_sparse_units"], 1], mean=0.0, stddev=0.02))
        self.value = tf.matmul(self.L2, tf.matmul(self.master.knowledge_base, self.sparse_representation_value), name="logits")

        self.sparse_representations = [self.sparse_representation_action, self.sparse_representation_value]

        self.probs = tf.nn.softmax(self.logits)

        self.action = tf.squeeze(tf.multinomial(self.logits - tf.reduce_max(self.logits, [1], keep_dims=True), 1), [1], name="action")
        self.action = tf.one_hot(self.action, self.nA)[0, :]

        log_probs = tf.nn.log_softmax(self.logits)
        self.actor_loss = - tf.reduce_sum(tf.reduce_sum(log_probs * self.actions_taken, [1]) * self.adv)

        self.critic_loss = 0.5 * tf.reduce_sum(tf.square(self.value - self.r))

        self.entropy = - tf.reduce_sum(self.probs * log_probs)

        self.loss = self.actor_loss + 0.5 * self.critic_loss - self.entropy * 0.01

        self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

        self.losses = [self.loss, self.actor_loss, self.critic_loss]
        # return self.action, self.value, actor_states, critic_states, actions_taken, [loss, actor_loss, critic_loss], adv, r, n_steps

    def choose_action(self, state, features):
        """Choose an action."""
        feed_dict = {
            self.actor_states: [state]
        }
        if self.rnn_state is not None:
            feed_dict[self.rnn_state_in] = features
        action, rnn_state, value = self.master.session.run([self.action, self.rnn_state_out, self.value], feed_dict=feed_dict)
        return action, value, rnn_state

    def pull_batch_from_queue(self):
        """
        Take a trajectory from the queue.
        Also immediately try to extend it if the episode
        wasn't over and more transitions are available
        """
        trajectory = self.runner.queue.get(timeout=600.0)
        while not trajectory.terminal:
            try:
                trajectory.extend(self.runner.queue.get_nowait())
            except queue.Empty:
                break
        return trajectory

    def get_env_action(self, action):
        return np.argmax(action)

    def get_critic_value(self, states, features):
        feed_dict = {
            self.critic_states: states
        }
        if self.rnn_state is not None:
            feed_dict[self.rnn_state_in] = features
        return self.master.session.run(self.value, feed_dict=feed_dict)[0]

    def run(self):
        # Assume global shared parameter vectors θ and θv and global shared counter T = 0
        # Assume thread-specific parameter vectors θ' and θ'v
        sess = self.master.session
        self.runner.start_runner(sess, self.writer)
        t = 1  # thread step counter
        while self.master.global_step.eval(session=sess) < self.config["T_max"] and not self.master.stop_requested:
            # Synchronize thread-specific parameters θ' = θ and θ'v = θv
            trajectory = self.pull_batch_from_queue()
            v = 0 if trajectory.terminal else self.get_critic_value(np.asarray(trajectory.states)[None, -1], trajectory.features[-1][0])
            rewards_plus_v = np.asarray(trajectory.rewards + [v])
            vpred_t = np.asarray(trajectory.values + [v])
            delta_t = trajectory.rewards + self.config["gamma"] * vpred_t[1:] - vpred_t[:-1]
            batch_r = discount_rewards(rewards_plus_v, self.config["gamma"])[:-1]
            batch_adv = discount_rewards(delta_t, self.config["gamma"])
            fetches = self.losses + [self.train_op, self.master.global_step]
            states = np.asarray(trajectory.states)
            feed_dict = {
                self.actor_states: states,
                self.critic_states: states,
                self.actions_taken: np.asarray(trajectory.actions),
                self.adv: batch_adv,
                self.r: np.asarray(batch_r)
            }
            feature = trajectory.features[0][0]
            if feature != []:
                feed_dict[self.rnn_state_in] = feature
            results = sess.run(fetches, feed_dict)
            n_states = states.shape[0]
            feed_dict = dict(zip(self.master.losses, map(lambda x: x / n_states, results)))
            summary = sess.run([self.master.summary_op], feed_dict)
            self.writer.add_summary(summary[0], results[-1])
            self.writer.flush()
            t += 1
        self.runner.stop_requested = True

class AsyncKnowledgeTransfer(Agent):
    """Asynchronous learner for variations of a task."""
    def __init__(self, envs, monitor_path, learning_method="REINFORCE", video=False, **usercfg):
        super(AsyncKnowledgeTransfer, self).__init__(**usercfg)
        self.envs = envs
        self.learning_method = learning_method
        self.video = video
        self.monitor_path = monitor_path
        self.config.update(dict(
            timesteps_per_batch=10000,
            trajectories_per_batch=10,
            batch_update="timesteps",
            n_iter=200,
            switch_at_iter=None,  # None to deactivate, otherwise an iteration at which to switch
            gamma=0.99,  # Discount past rewards by a percentage
            decay=0.9,  # Decay of RMSProp optimizer
            epsilon=1e-9,  # Epsilon of RMSProp optimizer
            learning_rate=0.005,
            n_hidden_units=10,
            repeat_n_actions=1,
            n_task_variations=3,
            n_sparse_units=10,
            feature_extraction=False
        ))
        self.config.update(usercfg)

        self.stop_requested = False

        self.session = tf.Session(config=tf.ConfigProto(
            log_device_placement=False,
            allow_soft_placement=True))

        with tf.variable_scope("global"):
            self.global_step = tf.get_variable("global_step", [], tf.int32, initializer=tf.constant_initializer(0, dtype=tf.int32), trainable=False)
            self.build_networks()
            self.losses, loss_summaries = self.create_summary_losses()
            self.reward = tf.placeholder("float", name="reward")
            tf.summary.scalar("Reward", self.reward)
            self.episode_length = tf.placeholder("float", name="episode_length")
            tf.summary.scalar("Episode_length", self.episode_length)
            self.summary_op = tf.summary.merge(loss_summaries)

        self.jobs = []
        for i, env in enumerate(self.envs):
            self.jobs.append(
                self.make_thread(
                    env,
                    i,
                    self.config["switch_at_iter"] if self.config["switch_at_iter"] is not None and i != len(self.envs) - 1 else self.config["n_iter"],
                    start_at_iter=(0 if self.config["switch_at_iter"] is None or i != len(self.envs) - 1 else self.config["switch_at_iter"])))

        for i, job in enumerate(self.jobs):
            only_sparse = (self.config["switch_at_iter"] is not None and i == len(self.jobs) - 1)
            grads = tf.gradients(job.loss, (self.shared_vars if not(only_sparse) else []) + job.sparse_representations)
            apply_grads = job.optimizer.apply_gradients(
                zip(
                    grads,
                    (self.shared_vars if not(only_sparse) else []) + job.sparse_representations
                )
            )
            inc_step = self.global_step.assign_add(self.n_steps)
            job.train_op = tf.group(apply_grads, inc_step)

        self.session.run(tf.global_variables_initializer())

        if self.config["save_model"]:
            for job in self.jobs:
                tf.add_to_collection("action", job.action)
            tf.add_to_collection("states", self.states)
            self.saver = tf.train.Saver()

    def build_networks(self):
        self.states = tf.placeholder(tf.float32, [None] + list(self.envs[0].observation_space.shape), name="states")
        self.action_taken = tf.placeholder(tf.float32, name="action_taken")
        self.advantage = tf.placeholder(tf.float32, name="advantage")

        if self.config["feature_extraction"]:
            self.L1 = tf.contrib.layers.fully_connected(
                inputs=self.states,
                num_outputs=self.config["n_hidden_units"],
                activation_fn=tf.tanh,
                weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.02),
                biases_initializer=tf.zeros_initializer(),
                scope="L1")
        else:
            self.L1 = self.states
        self.knowledge_base = tf.Variable(tf.truncated_normal([self.L1.get_shape()[-1].value, self.config["n_sparse_units"]], mean=0.0, stddev=0.02), name="knowledge_base")

        self.shared_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    def signal_handler(self, signal, frame):
        """When a (SIGINT) signal is received, request the threads (via the master) to stop after completing an iteration."""
        logging.info("SIGINT signal received: Requesting a stop...")
        self.stop_requested = True

    def learn(self):
        signal.signal(signal.SIGINT, self.signal_handler)
        if self.config["switch_at_iter"] is None:
            idx = None
        else:
            total_T_max = self.config["T_max"]
            self.config["T_max"] = self.config["switch_at_iter"]
            idx = -1
        for job in self.jobs[:idx]:
            job.start()
        for job in self.jobs[:idx]:
            job.join()
        try:
            self.config["T_max"] = total_T_max
            self.jobs[idx].start()
            self.jobs[idx].join()
        except TypeError:  # idx is None
            pass

        if self.config["save_model"]:
            self.saver.save(self.session, os.path.join(self.monitor_path, "model"))

    def make_thread(self, env, task_id, n_iter, start_at_iter=0):
        return AKTThread(self, env, task_id, n_iter, start_at_iter=start_at_iter)

class AsyncKnowledgeTransferRNNCNN(AsyncKnowledgeTransfer):
    """Asynchronous knowledge transfer learner that uses an RNN and CNN."""
    def __init__(self, envs, monitor_path, **usercfg):
        self.thread_type = AKTThreadDiscreteCNNRNN
        super(AsyncKnowledgeTransferRNNCNN, self).__init__(envs, monitor_path, **usercfg)
        self.config["RNN"] = True
        self.config["n_sparse_units"] = self.config.get("n_sparse_units", 20)

    def build_networks(self):
        with tf.variable_scope("global"):
            self.states = tf.placeholder(tf.float32, [None] + list(self.envs[0].observation_space.shape), name="states")
            self.n_steps = tf.shape(self.states)[0]
            x = self.states
            # Convolution layers
            for i in range(4):
                x = tf.nn.elu(conv2d(x, 32, "l{}".format(i + 1), [3, 3], [2, 2]))

            # Flatten
            self.L1 = tf.expand_dims(flatten(x), [0])

            # 256 is the LSTM size
            self.knowledge_base = tf.Variable(tf.truncated_normal([256, self.config["n_sparse_units"]], mean=0.0, stddev=0.02), name="knowledge_base")
            self.shared_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

            self.global_step = tf.get_variable("global_step", [], tf.int32, initializer=tf.constant_initializer(0, dtype=tf.int32), trainable=False)

    def make_thread(self, env, task_id, n_iter, start_at_iter=0):
        return AKTThreadDiscreteCNNRNN(self, env, task_id, n_iter)

    def create_summary_losses(self):
        self.actor_loss = tf.placeholder("float", name="actor_loss")
        actor_loss_summary = tf.summary.scalar("Actor_loss", self.actor_loss)
        self.critic_loss = tf.placeholder("float", name="critic_loss")
        critic_loss_summary = tf.summary.scalar("Critic_loss", self.critic_loss)
        self.loss = tf.placeholder("float", name="loss")
        loss_summary = tf.summary.scalar("loss", self.loss)
        return [self.actor_loss, self.critic_loss, self.loss], [actor_loss_summary, critic_loss_summary, loss_summary]
