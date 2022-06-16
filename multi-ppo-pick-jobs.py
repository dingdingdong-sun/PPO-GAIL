import numpy as np
import tensorflow as tf
import gym
import os
import sys
import time
from src.spinup.spinup.utils.logx import EpochLogger
from src.spinup.spinup.utils.mpi_tf import MpiAdamOptimizer, sync_all_params
from src.spinup.spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
from src.spinup.spinup.utils.logx import restore_tf_graph
import os.path as osp
from multi_HPCSimPickJobs import *

USER_NUM = 0

def load_policy(model_path, itr='last'):
    # handle which epoch to load from
    if itr == 'last':
        saves = [int(x[11:]) for x in os.listdir(model_path) if 'simple_save' in x and len(x) > 11]
        itr = '%d' % max(saves) if len(saves) > 0 else ''
    else:
        itr = '%d' % itr

    # load the things!
    sess = tf.Session()
    model = restore_tf_graph(sess, osp.join(model_path, 'simple_save' + itr))

    # get the correct op for executing actions
    pi = model['pi']
    v = model['v']

    # make function for producing an action given a single state
    get_probs = lambda x, y: sess.run(pi, feed_dict={model['x']: x.reshape(-1, MAX_QUEUE_SIZE * JOB_FEATURES),
                                                     model['mask']: y.reshape(-1, MAX_QUEUE_SIZE)})
    get_v = lambda x: sess.run(v, feed_dict={model['x']: x.reshape(-1, MAX_QUEUE_SIZE * JOB_FEATURES)})
    return get_probs, get_v


def critic_mlp(job_f, user_f, cluster_node_num, last_state):
    job_f = tf.reshape(job_f, [-1, MAX_QUEUE_SIZE, JOB_FEATURES + cluster_node_num])
    x = tf.layers.dense(job_f, units=32, activation=tf.nn.sigmoid)
    x = tf.layers.dense(x, units=16, activation=tf.nn.sigmoid)
    x = tf.layers.dense(x, units=1, activation=tf.nn.sigmoid)
    x = tf.squeeze(x, axis=-1)

    u, state = rnn(user_f, last_state, cluster_node_num, 16)
    delay = tf.squeeze(tf.layers.dense(u, units=1), axis=-1)
    index = job_f[:, :, 4] * USER_NUM
    index = tf.squeeze(tf.reshape(index, [-1]))
    index = tf.cast(index, dtype=tf.int32)
    delay_out = tf.gather(delay, index)
    delay_out = tf.reshape(delay_out, [-1, MAX_QUEUE_SIZE])
    x = x * delay_out
    v_out = tf.reduce_mean(x, axis=1)


    return v_out, state


def mlp_v1(x, act_dim):
    x = tf.reshape(x, shape=[-1, MAX_QUEUE_SIZE * JOB_FEATURES])
    x = tf.layers.dense(x, units=128, activation=tf.nn.relu)
    x = tf.layers.dense(x, units=128, activation=tf.nn.relu)
    x = tf.layers.dense(x, units=128, activation=tf.nn.relu)
    return tf.layers.dense(x, units=act_dim)


def mlp_v2(x, act_dim):
    x = tf.reshape(x, shape=[-1, MAX_QUEUE_SIZE * JOB_FEATURES])
    x = tf.layers.dense(x, units=32, activation=tf.nn.relu)
    x = tf.layers.dense(x, units=16, activation=tf.nn.relu)
    x = tf.layers.dense(x, units=8, activation=tf.nn.relu)
    return tf.layers.dense(x, units=act_dim)


def mlp_v3(x, act_dim):
    x = tf.reshape(x, shape=[-1, MAX_QUEUE_SIZE * JOB_FEATURES])
    x = tf.layers.dense(x, units=32, activation=tf.nn.relu)
    x = tf.layers.dense(x, units=32, activation=tf.nn.relu)
    x = tf.layers.dense(x, units=32, activation=tf.nn.relu)
    x = tf.layers.dense(x, units=32, activation=tf.nn.relu)
    x = tf.layers.dense(x, units=32, activation=tf.nn.relu)
    return tf.layers.dense(x, units=act_dim)


def rnn(x, last_state, max_nodes, num_unit):
    x = tf.reshape(x, [-1, max_nodes])
    last_state = tf.reshape(last_state, [-1, 16])
    cell = tf.nn.rnn_cell.BasicRNNCell(num_unit, activation=tf.nn.sigmoid, reuse=tf.AUTO_REUSE)
    output, states = cell(x, last_state)
    return output, states



def rl_kernel(job_f, user_f, cluster_node_num, last_state):



    # x = tf.reshape(job_f, shape=[-1, JOB_FEATURES])
    # job_f = tf.reshape(job_f, shape=[-1, JOB_FEATURES])
    x = tf.layers.dense(job_f, units=32, activation=tf.nn.sigmoid, kernel_initializer=tf.random_normal_initializer())
    x = tf.layers.dense(x, units=16, activation=tf.nn.sigmoid, kernel_initializer=tf.random_normal_initializer())
    x = tf.layers.dense(x, units=1, activation=tf.nn.sigmoid, kernel_initializer=tf.random_normal_initializer())
    x = tf.squeeze(x, axis=-1)

    u, state = rnn(user_f, last_state, cluster_node_num, 16)
    delay = tf.squeeze(tf.layers.dense(u, units=1), axis=-1)
    index = job_f[:, :, 4] * USER_NUM
    index = tf.reshape(index, [-1])
    index = tf.cast(index, dtype=tf.int32)
    delay_out = tf.gather(delay, index)
    delay_out = tf.reshape(delay_out, [-1, MAX_QUEUE_SIZE])
    # delay_out = tf.map_fn(lambda z: delay[z[4]], job_f)
    x = x * delay_out

    return x, state


def attention(x, act_dim):
    x = tf.reshape(x, shape=[-1, MAX_QUEUE_SIZE, JOB_FEATURES])
    # x = tf.layers.dense(x, units=32, activation=tf.nn.relu)
    q = tf.layers.dense(x, units=32, activation=tf.nn.relu)
    k = tf.layers.dense(x, units=32, activation=tf.nn.relu)
    v = tf.layers.dense(x, units=32, activation=tf.nn.relu)
    score = tf.matmul(q, tf.transpose(k, [0, 2, 1]))
    score = tf.nn.softmax(score, -1)
    attn = tf.reshape(score, (-1, MAX_QUEUE_SIZE, MAX_QUEUE_SIZE))
    x = tf.matmul(attn, v)
    x = tf.layers.dense(x, units=16, activation=tf.nn.relu)

    x = tf.layers.dense(x, units=8, activation=tf.nn.relu)
    x = tf.squeeze(tf.layers.dense(x, units=1), axis=-1)
    # x = tf.layers.dense(x, units=128, activation=tf.nn.relu)
    # x = tf.layers.dense(x, units=64, activation=tf.nn.relu)
    # x = tf.layers.dense(x, units=64, activation=tf.nn.relu)
    return x


def lenet(x_ph, act_dim):
    m = int(np.sqrt(MAX_QUEUE_SIZE))
    x = tf.reshape(x_ph, shape=[-1, m, m, JOB_FEATURES])
    x = tf.layers.conv2d(inputs=x, filters=32, kernel_size=[1, 1], strides=1)
    x = tf.layers.max_pooling2d(x, [2, 2], 2)
    x = tf.layers.conv2d(inputs=x, filters=64, kernel_size=[1, 1], strides=1)
    x = tf.layers.max_pooling2d(x, [2, 2], 2)
    x = tf.layers.flatten(x)
    x = tf.layers.dense(x, units=64)

    return tf.layers.dense(
        inputs=x,
        units=act_dim,
        activation=None
    )


"""
Policies
"""


def categorical_policy(job_f, user_f, a, mask, max_nodes, last_state):
    job_f = tf.reshape(job_f, [-1, MAX_QUEUE_SIZE, JOB_FEATURES + max_nodes])
    mask = tf.reshape(mask, [-1, MAX_QUEUE_SIZE])
    act_dim = MAX_QUEUE_SIZE
    output_layer, next_state = rl_kernel(job_f, user_f, max_nodes, last_state)
    # logp = tf.reduce_sum(output_layer)
    # output_layer = tf.expand_dims(output_layer, axis=0)
    output_layer = output_layer + (mask - 1) * 1000000
    logp_all = tf.nn.log_softmax(output_layer)

    indexs1 = tf.cast(tf.multinomial(output_layer, 1), dtype=tf.int32)
    indexs0 = tf.expand_dims(tf.range(tf.shape(mask)[0]), axis=1)
    indexs = tf.concat([indexs0, indexs1], axis=1)
    pi = tf.squeeze(indexs1, axis=1)
    logp = tf.reduce_sum(tf.one_hot(a, depth=act_dim) * logp_all, axis=1)
    # logp = tf.reduce_sum(logp_all)
    logp_pi = tf.reduce_sum(tf.one_hot(pi, depth=act_dim) * logp_all, axis=1)
    prio = tf.gather_nd(output_layer, indexs)
    return pi, logp, logp_pi, prio, next_state


"""
Actor-Critics
"""



def actor_critic(x, a, mask, user_f, max_nodes, last_state_a, last_state_v):
    job_f = x
    user_f = tf.reshape(user_f, [-1, max_nodes])
    # cluster_f = tf.reshape(cluster_f, [agent_num])
    with tf.variable_scope('pi'):
        pi, logp, logp_pi, prio, next_state_a = categorical_policy(job_f, user_f, a, mask, max_nodes, last_state_a)
    with tf.variable_scope('v'):
        v, next_state_v = critic_mlp(job_f, user_f, max_nodes, last_state_v)
        # v = tf.squeeze(v)
    return pi, logp, logp_pi, v, prio, next_state_a, next_state_v

class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0., agent_num=1):
        size = size * 100  # assume the traj can be really long
        # self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.obs_buf = []
        self.state_actor_buf = []
        self.state_critic_buf = []
        self.cobs_buf = None
        self.act_buf = []
        self.mask_buf = []
        self.adv_buf = np.zeros((size, agent_num), dtype=np.float32)
        self.rew_buf = []
        self.ret_buf = np.zeros((size, agent_num), dtype=np.float32)
        self.val_buf = []
        self.logp_buf = []
        self.user_info = []
        self.user_index_buf = []
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
        self.a_num = agent_num

    def store(self, obs, state_actor, state_critic,cobs, act, mask, rew, val, logp, user_infomation,u_i):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf.append(obs)
        self.state_actor_buf.append(state_actor)
        self.state_critic_buf.append(state_critic)
        # self.cobs_buf[self.ptr] = cobs
        self.act_buf.append(act)
        self.mask_buf.append(mask)
        self.rew_buf.append(rew)
        self.val_buf.append(val)
        self.logp_buf.append(logp)
        self.user_info.append(user_infomation)
        self.user_index_buf.append(u_i)
        self.ptr += 1

    def finish_path(self, last_reward=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)

        last_val = np.ones((1, self.a_num), dtype=np.float32)
        last_val = last_val * last_reward
        # print(self.rew_buf[path_slice])
        rew_path = np.zeros((self.ptr-self.path_start_idx,self.a_num), np.float32)
        val_path = np.zeros((self.ptr - self.path_start_idx, self.a_num), np.float32)
        for x in range(self.ptr-self.path_start_idx):
            i = 0
            for u in sorted(self.user_index_buf[x]):
                rew_path[x, u] = self.rew_buf[x+self.path_start_idx][i]
                val_path[x, u] = self.val_buf[x+self.path_start_idx][i]
        # rew_path = np.array(self.rew_buf[path_slice], np.float32)
        # val_path = np.array(self.val_buf[path_slice], np.float32)
        print(val_path.shape)
        print(rew_path.shape)
        # no_val_agent = np.zeros((self.ptr - self.path_start_idx, self.a_num - val_path.shape[1]))
        # val_path = np.append(val_path, no_val_agent, axis=1)



        rews = np.append(rew_path, last_val, axis=0)
        vals = np.append(val_path, last_val, axis=0)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1, :] + self.gamma * vals[1:, :] - vals[:-1, :]
        for i in range(self.a_num):
            self.adv_buf[path_slice, i] = discount_cumsum(deltas[:, i], self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        for i in range(self.a_num):
            self.ret_buf[path_slice, i] = discount_cumsum(rews[:, i], self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr < self.max_size
        actual_size = self.ptr
        self.ptr, self.path_start_idx = 0, 0

        actual_adv_buf = []
        actual_ret_buf = []
        for x in range(actual_size):
            actual_adv = []
            actual_ret = []
            for u in sorted(self.user_index_buf[x]):
                actual_adv.append(self.adv_buf[x, u])
                actual_ret.append(self.ret_buf[x, u])
            actual_adv_buf.append(actual_adv)
            actual_ret_buf.append(actual_ret)
        actual_adv_buf = np.concatenate(actual_adv_buf, dtype=np.float32)
        # actual_adv_buf = actual_adv_buf[:actual_size]
        # print ("-----------------------> actual_adv_buf: ", actual_adv_buf)
        adv_sum = np.sum(actual_adv_buf, axis=0)
        adv_n = len(actual_adv_buf)
        adv_mean = adv_sum / adv_n
        adv_sum_sq = np.sum((actual_adv_buf - adv_mean) ** 2, axis=0)
        adv_std = np.sqrt(adv_sum_sq / adv_n)
        # print ("-----------------------> adv_std:", adv_std)
        actual_adv_buf = (actual_adv_buf - adv_mean) / adv_std
        # print(self.obs_buf[0])

        return [self.obs_buf[:actual_size],
                self.state_actor_buf[:actual_size],
                self.state_critic_buf[:actual_size],
                self.act_buf[:actual_size],
                self.mask_buf[:actual_size], actual_adv_buf,
                actual_ret_buf, self.logp_buf[:actual_size], self.user_info[:actual_size]]

        # return [np.array(self.obs_buf[:actual_size], dtype=np.float32),
        #         np.array(self.state_actor_buf[:actual_size], dtype=np.float32),
        #         np.array(self.state_critic_buf[:actual_size], dtype=np.float32),
        #         np.array(self.all_user_info_buf[:actual_size], dtype=np.float32),
        #         np.array(self.act_buf[:actual_size], dtype=np.float32),
        #         np.array(self.mask_buf[:actual_size], dtype=np.float32), actual_adv_buf,
        #         self.ret_buf[:actual_size], np.array(self.logp_buf[:actual_size], dtype=np.float32)]


def maxminnorm(array):
    # print(array)
    s = np.sum(array)
    if s == 0:
        array_out = np.ones_like(array)
        array_out = array_out/len(array)
    else:
        array_out = array/s
    return array_out

def sort_dict_by_key(d):
    k = sorted(d.keys())
    out = []
    for i in k:
        out.append(list(d[i]))
    return out

def change_dict_by_key(d, l):
    k = sorted(d.keys())
    x = 0
    for i in k:
        d[i] = l[x]
        x += 1

"""

Proximal Policy Optimization (by clipping), 

with early stopping based on approximate KL

"""


def multi_ppo(workload_file, model_path, ac_kwargs=dict(), seed=0,
        traj_per_epoch=400, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=1e-2,
        vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000,
        target_kl=0.003, logger_kwargs=dict(), save_freq=10, pre_trained=0, trained_model=None, attn=False,
        shuffle=False,
        backfil=False, skip=False, score_type=0, batch_job_slice=0, filter=False):
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    tf.set_random_seed(seed)
    np.random.seed(seed)

    env = HPCEnv(shuffle=shuffle, backfil=backfil, skip=skip, job_score_type=score_type,
                 batch_job_slice=batch_job_slice, build_sjf=False, filter=filter)
    env.seed(seed)
    env.my_init(workload_file=workload_file, sched_file=model_path)

    USER_NUM = len(env.loads.users_id.keys())

    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Share information about action space with policy architecture
    ac_kwargs['action_space'] = env.action_space
    ac_kwargs['attn'] = attn
    agent_num = env.agent_num
    max_nodes = env.loads.max_nodes


    # Inputs to computation graph

    buf = PPOBuffer(obs_dim, act_dim, traj_per_epoch * JOB_SEQUENCE_SIZE, gamma, lam, agent_num)

    if pre_trained:
        sess = tf.Session()
        model = restore_tf_graph(sess, trained_model)
        logger.log('load pre-trained model')
        # Count variables
        var_counts = tuple(count_vars(scope) for scope in ['pi', 'v'])
        logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n' % var_counts)

        x_ph = model['x']
        a_ph = model['a']
        last_state_a = model['a_last_state']
        last_state_v= model['v_last_state']
        user_f = model['user']
        mask_ph = model['mask']
        adv_ph = model['adv']
        ret_ph = model['ret']
        logp_old_ph = model['logp_old_ph']

        pi = model['pi']
        v = model['v']
        prio = model['priotity']
        next_state_a = model['a_next_state']
        next_state_v = model['v_next_state']
        # logits = model['logits']
        logp = model['logp']
        logp_pi = model['logp_pi']
        pi_loss = model['pi_loss']
        v_loss = model['v_loss']
        approx_ent = model['approx_ent']
        approx_kl = model['approx_kl']
        clipfrac = model['clipfrac']
        clipped = model['clipped']

        # Optimizers
        # graph = tf.get_default_graph()
        # op = sess.graph.get_operations()
        # [print(m.values()) for m in op]
        # train_pi = graph.get_tensor_by_name('pi/conv2d/kernel/Adam:0')
        # train_v = graph.get_tensor_by_name('v/conv2d/kernel/Adam:0')
        train_pi = tf.get_collection("train_pi")[0]
        train_v = tf.get_collection("train_v")[0]
        # train_pi_optimizer = MpiAdamOptimizer(learning_rate=pi_lr, name='AdamLoad')
        # train_pi = train_pi_optimizer.minimize(pi_loss)
        # train_v_optimizer = MpiAdamOptimizer(learning_rate=vf_lr, name='AdamLoad')
        # train_v = train_v_optimizer.minimize(v_loss)
        # sess.run(tf.variables_initializer(train_pi_optimizer.variables()))
        # sess.run(tf.variables_initializer(train_v_optimizer.variables()))
        # Need all placeholders in *this* order later (to zip with data from buffer)
        all_phs = [x_ph, last_state_a, last_state_v, user_f, a_ph, mask_ph, adv_ph, ret_ph, logp_old_ph]
        # Every step, get: action, value, and logprob
        get_action_ops = [pi, v, logp_pi, prio, next_state_a, next_state_v]

    else:
        a_ph = placeholder_from_space(env.action_space)
        x_ph = tf.placeholder(dtype=tf.float32, shape=[None, MAX_QUEUE_SIZE, JOB_FEATURES + max_nodes])
        # y_ph = placeholder(JOB_SEQUENCE_SIZE*3) # 3 is the number of sequence features
        mask_ph = tf.placeholder(dtype=tf.float32, shape=[None, MAX_QUEUE_SIZE])
        last_state_a = tf.placeholder(dtype=tf.float32, shape=[None, 16])
        last_state_v = tf.placeholder(dtype=tf.float32, shape=[None, 16])
        user_f = tf.placeholder(dtype=tf.float32, shape=[None, max_nodes])
        adv_ph, ret_ph, logp_old_ph = placeholders(None, None, None)

        # Main outputs from computation graph
        pi, logp, logp_pi, v, prio, next_state_a, next_state_v = actor_critic(x_ph, a_ph, mask_ph, user_f, max_nodes, last_state_a, last_state_v)

        # Need all placeholders in *this* order later (to zip with data from buffer)
        all_phs = [x_ph, last_state_a, last_state_v, user_f, a_ph, mask_ph, adv_ph, ret_ph, logp_old_ph]

        # Every step, get: action, value, and logprob
        get_action_ops = [pi, v, logp_pi, prio, next_state_a, next_state_v]

        # Experience buffer

        # Count variables
        var_counts = tuple(count_vars(scope) for scope in ['pi', 'v'])
        logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n' % var_counts)

        # PPO objectives
        ratio = tf.exp(logp - logp_old_ph)  # pi(a|s) / pi_old(a|s)
        min_adv = tf.where(adv_ph > 0, (1 + clip_ratio) * adv_ph, (1 - clip_ratio) * adv_ph)
        # pi_loss = -tf.reduce_mean(tf.minimum(ratio * adv_ph, min_adv))
        pi_loss = -tf.reduce_mean(logp)
        v_loss = tf.reduce_mean((ret_ph - v) ** 2)

        # Info (useful to watch during learning)
        approx_kl = tf.reduce_mean(logp_old_ph - logp)  # a sample estimate for KL-divergence, easy to compute
        approx_ent = tf.reduce_mean(-logp)  # a sample estimate for entropy, also easy to compute
        clipped = tf.logical_or(ratio > (1 + clip_ratio), ratio < (1 - clip_ratio))
        clipfrac = tf.reduce_mean(tf.cast(clipped, tf.float32))

        # Optimizers
        train_pi = tf.train.AdamOptimizer(learning_rate=pi_lr).minimize(pi_loss)
        # train_pi = tf.train.AdamOptimizer(learning_rate=pi_lr)
        # train_pi_grad = train_pi.compute_gradients(pi_loss, tf.trainable_variables())
        # train_pi_apply = train_pi.apply_gradients(train_pi_grad)
        train_v = tf.train.AdamOptimizer(learning_rate=vf_lr).minimize(v_loss)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        tf.add_to_collection("train_pi", train_pi)
        tf.add_to_collection("train_v", train_v)

    # Setup model saving
    # logger.setup_tf_saver(sess, inputs={'x': x_ph}, outputs={'action_probs': action_probs, 'log_picked_action_prob': log_picked_action_prob, 'v': v})
    logger.setup_tf_saver(sess, inputs={'x': x_ph, 'a_last_state': last_state_a, 'v_last_state': last_state_v,
                                        'user': user_f, 'a': a_ph, 'adv': adv_ph, 'mask': mask_ph, 'ret': ret_ph,
                                        'logp_old_ph': logp_old_ph},
                          outputs={'pi': pi, 'v': v, 'pi_loss': pi_loss, 'logp': logp, 'logp_pi': logp_pi,
                                   'v_loss': v_loss, 'approx_ent': approx_ent, 'approx_kl': approx_kl,
                                   'priotity': prio, 'a_next_state': next_state_a, 'v_next_state': next_state_v,
                                   'clipped': clipped, 'clipfrac': clipfrac})

    def update():
        obs_h, st_a_h, st_v_h, a_h, mask_h, adv_h, ret_h, logp_old_h, user_info_h = buf.get()
        # print(np.concatenate(logp_old_h, axis=0))
        # print(max_nodes)
        logp_old_h = np.concatenate(logp_old_h)
        ret_h = np.concatenate(ret_h)
        obs_h = np.concatenate(obs_h).reshape(-1, MAX_QUEUE_SIZE, JOB_FEATURES + max_nodes)
        a_h = np.concatenate(a_h, dtype=np.int32)
        mask_h = np.concatenate(mask_h).reshape(-1, MAX_QUEUE_SIZE)
        st_a_h = np.concatenate(st_a_h).reshape(-1, 16)
        st_v_h = np.concatenate(st_v_h).reshape(-1, 16)
        user_info_h = np.concatenate(user_info_h, dtype=np.float32).reshape(-1, max_nodes)
        pi_l_old, v_l_old, ent = sess.run([pi_loss, v_loss, approx_ent], feed_dict={
            logp_old_ph: logp_old_h,
            adv_ph: adv_h,
            ret_ph: ret_h,
            x_ph: obs_h,
            a_ph: a_h,
            mask_ph: mask_h,
            last_state_a: st_a_h,
            last_state_v: st_v_h,
            user_f: user_info_h
        })
        # print(pi_l_old)

        # Training
        for i in range(train_pi_iters):
            aa, kl, los = sess.run([train_pi, approx_kl, pi_loss], feed_dict={
                logp_old_ph: logp_old_h,
                adv_ph: adv_h,
                ret_ph: ret_h,
                x_ph: obs_h,
                a_ph: a_h,
                mask_ph: mask_h,
                last_state_a: st_a_h,
                last_state_v: st_v_h,
                user_f: user_info_h
            })
            kl = mpi_avg(kl)
            # print(los)
            # print(np.concatenate(mask_h).reshape(-1, MAX_QUEUE_SIZE))
            if abs(kl) >= 1.5 * target_kl:
                logger.log('Early stopping at step %d due to reaching max kl.' % i)
                break
        logger.store(StopIter=i)
        for _ in range(train_v_iters):
            sess.run(train_v, feed_dict={
                logp_old_ph: logp_old_h,
                adv_ph: adv_h,
                ret_ph: ret_h,
                x_ph: obs_h,
                a_ph: a_h,
                mask_ph: mask_h,
                last_state_a: st_a_h,
                last_state_v: st_v_h,
                user_f: user_info_h
            })

        # Log changes from update
        pi_l_new, v_l_new, kl, cf = sess.run([pi_loss, v_loss, approx_kl, clipfrac], feed_dict={
            logp_old_ph: logp_old_h,
            adv_ph: adv_h,
            ret_ph: ret_h,
            x_ph: obs_h,
            a_ph: a_h,
            mask_ph: mask_h,
            last_state_a: st_a_h,
            last_state_v: st_v_h,
            user_f: user_info_h
        })
        logger.store(LossPi=pi_l_new, LossV=v_l_new,
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(pi_l_new - pi_l_old),
                     DeltaLossV=(v_l_new - v_l_old))

    start_time = time.time()
    [o, co, user_index_list], d, ep_ret, ep_len = env.reset(), False, 0, 0
    r = [0.0 for _ in range(len(user_index_list))]

    # Main loop: collect experience in env and update/log each epoch
    start_time = time.time()
    num_total = 0
    for epoch in range(epochs):
        l_states_a = {}
        l_states_v = {}
        t = 0
        while True:
            # lst = []
            # for i in range(0, MAX_QUEUE_SIZE * JOB_FEATURES, JOB_FEATURES):
            #     if all(o[i:i + JOB_FEATURES] == [0] + [1] * (JOB_FEATURES - 2) + [0]):
            #         lst.append(0)
            #     elif all(o[i:i + JOB_FEATURES] == [1] * JOB_FEATURES):
            #         lst.append(0)
            #     else:
            #         lst.append(1)
            users_job = {}
            users_job_index_map = {}
            users_mask = {}
            l_states_a_new = {}
            l_states_v_new = {}
            all_user_info = o[MAX_QUEUE_SIZE * JOB_FEATURES:]
            cluster_info = np.sum(all_user_info.reshape(-1, max_nodes), axis=0)
            for i in range(0, MAX_QUEUE_SIZE * JOB_FEATURES, JOB_FEATURES):

                #     users_job[o[i + 4]].append(o[i:i + JOB_FEATURES])
                #     if all(o[i:i + JOB_FEATURES] == [0] + [1] * (JOB_FEATURES - 2) + [0]):
                #             users_mask[o[i+4]].append(0)
                #         elif all(o[i:i + JOB_FEATURES] == [1] * JOB_FEATURES):
                #             users_mask[o[i+4]].append(0)
                #         else:
                #             users_mask[o[i+4]].append(1)
                # else:
                if o[i + 4] not in users_job.keys():
                    users_job_index_map[o[i + 4]] = []
                    users_job[o[i + 4]] = []
                    users_mask[o[i + 4]] = []
                users_job[o[i + 4]].append(np.concatenate((o[i:i + JOB_FEATURES], cluster_info[:]), axis=0))
                users_job_index_map[o[i + 4]].append(int(i/JOB_FEATURES))
                if all(o[i:i + JOB_FEATURES] == [0] + [1] * (JOB_FEATURES - 2) + [0]):
                    users_mask[o[i+4]].append(0)
                elif all(o[i:i + JOB_FEATURES] == [1] * JOB_FEATURES):
                    users_mask[o[i+4]].append(0)
                else:
                    users_mask[o[i+4]].append(1)

            # 用户编号1.0的任务是用于完成队列补充的，实际并不存在这个用户，因此删除
            if 1.0 in users_mask.keys():
                users_mask.pop(1.0)
                users_job.pop(1.0)
                users_job_index_map.pop(1.0)
            user_info = []
            user_index_tmp = []
            for u in sorted(users_job.keys()):
                i = int(u * len(env.loads.users_id.keys()))
                user_index_tmp.append(i)
                user_f_tmp = all_user_info[i * max_nodes:(i + 1) * max_nodes]
                user_info.append(user_f_tmp)
                while len(users_job[u]) < MAX_QUEUE_SIZE:
                    fea_tmp = np.array([0] + [1] * (JOB_FEATURES - 2) + [0], dtype=np.float32)
                    users_job[u].append(np.concatenate((fea_tmp, cluster_info[:]), axis=0))
                    users_mask[u].append(0)
                if u not in sorted(l_states_a.keys()):
                    l_states_a_new[u] = np.zeros([1,16], dtype=np.float32)
                    l_states_v_new[u] = np.zeros([1, 16], dtype=np.float32)
                else:
                    l_states_a_new[u] = l_states_a[u].reshape(1, 16)
                    l_states_v_new[u] = l_states_v[u].reshape(1, 16)
            # print(sort_dict_by_key(l_states_a_new))
            # print("next")
            job_index_map = sort_dict_by_key(users_job_index_map)
            a, v_t, logp_t, prio_t, l_state_a, l_state_v= sess.run(get_action_ops,
                                            feed_dict={x_ph: np.array(sort_dict_by_key(users_job)).reshape(-1,MAX_QUEUE_SIZE, JOB_FEATURES + max_nodes),
                                                        mask_ph: np.array(sort_dict_by_key(users_mask)).reshape(-1,MAX_QUEUE_SIZE),
                                                        last_state_a: np.array(sort_dict_by_key(l_states_a_new)).reshape(-1, 16),
                                                        last_state_v: np.array(sort_dict_by_key(l_states_v_new)).reshape(-1, 16),
                                                        user_f: np.array(user_info, dtype=np.float32).reshape(-1, max_nodes)})
            # print(ou1)
            # print(ou2)

            change_dict_by_key(l_states_a_new, l_state_a)
            change_dict_by_key(l_states_v_new, l_state_v)
            for key in l_states_a_new.keys():
                l_states_a[key] = l_states_a_new[key]
                l_states_v[key] = l_states_v_new[key]
            # a = np.squeeze(np.array(a))
            # v_t = np.squeeze(np.array(v_t))
            # logp_t = np.squeeze(np.array(logp_t))
            # prio_t = np.squeeze(np.array(prio_t))

            a_real = []
            for l in range(len(a)):
                a_real.append(job_index_map[l][a[l]])
            # print(a_real)
            # print(prio_t)
            if a.shape[0] > 1:
                prio_t = maxminnorm(prio_t)
                a_joint = np.random.choice(a_real, 1, True, prio_t)
            else:
                prio_t = prio_t
                a_joint = a_real

            num_total += 1
            '''
            action = np.random.choice(np.arange(MAX_QUEUE_SIZE), p=action_probs)
            log_action_prob = np.log(action_probs[action])
            '''

            # save and log
            # oo = []
            # st_a = []
            # st_v = []
            # mask = []
            # for k in sorted(users_job.keys()):
            #     for j in users_job[k]:
            #         oo.append(j)
            #     for s_a in l_states_a[k]:
            #         st_a.append(s_a)
            #     for s_v in l_states_v[k]:
            #         st_v.append(s_v)
            #     for ma in users_mask[k]:
            #         mask.append(ma)
            oo = np.array(sort_dict_by_key(users_job)).reshape(-1,MAX_QUEUE_SIZE, JOB_FEATURES + max_nodes)
            st_a = np.array(sort_dict_by_key(l_states_a_new)).reshape(-1, 16)
            st_v = np.array(sort_dict_by_key(l_states_v_new)).reshape(-1, 16)
            user_infomation = np.array(user_info, dtype=np.float32).reshape(-1, max_nodes)
            mask = np.array(sort_dict_by_key(users_mask)).reshape(-1,MAX_QUEUE_SIZE)

            buf.store(oo, st_a, st_v, None, a, mask, r, v_t, logp_t, user_infomation, user_index_tmp)

            o, r, d, r2, sjf_t, f1_t = env.step(a_joint[0])
            r_tmp = []
            for x in user_index_tmp:
                r_tmp.append(r[x])
            r = r_tmp


            ep_ret += np.sum(r)
            ep_len += 1

            if d:
                t += 1
                buf.finish_path(r)
                logger.store(EpRet=ep_ret, EpLen=ep_len)
                [o, co, user_index_list], d, ep_ret, ep_len = env.reset(), False, 0, 0
                r = [0.0 for _ in range(len(user_index_list))]
                if t >= traj_per_epoch:
                    # print ("state:", state, "\nlast action in a traj: action_probs:\n", action_probs, "\naction:", action)
                    break
        # print("Sample time:", (time.time()-start_time)/num_total, num_total)
        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs - 1):
            logger.save_state({'env': env}, None)

        # Perform PPO update!
        # start_time = time.time()
        update()
        # print("Train time:", time.time()-start_time)

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch + 1) * traj_per_epoch * JOB_SEQUENCE_SIZE)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('ClipFrac', average_only=True)
        logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('Time', time.time() - start_time)
        logger.dump_tabular()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--workload', type=str,
                        default='./data/lublin_256.swf')  # RICC-2010-2 lublin_256.swf SDSC-SP2-1998-4.2-cln.swf
    parser.add_argument('--model', type=str, default='./data/lublin_256.schd')
    parser.add_argument('--gamma', type=float, default=1)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--trajs', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--exp_name', type=str, default='ppo')
    parser.add_argument('--pre_trained', type=int, default=0)
    parser.add_argument('--trained_model', type=str, default='./data/logs/ppo_temp/ppo_temp_s0')
    parser.add_argument('--attn', type=int, default=0)
    parser.add_argument('--shuffle', type=int, default=0)
    parser.add_argument('--backfil', type=int, default=0)
    parser.add_argument('--skip', type=int, default=0)
    parser.add_argument('--score_type', type=int, default=0)
    parser.add_argument('--batch_job_slice', type=int, default=0)
    parser.add_argument('--filter', type=int, default=0)
    args = parser.parse_args()

    from src.spinup.spinup.utils.run_utils import setup_logger_kwargs

    # build absolute path for using in hpc_env.
    current_dir = os.getcwd()
    workload_file = os.path.join(current_dir, args.workload)
    log_data_dir = os.path.join(current_dir, './data/logs/')
    logger_kwargs = setup_logger_kwargs(args.exp_name, seed=args.seed, data_dir=log_data_dir)
    if args.pre_trained:
        model_file = os.path.join(current_dir, args.trained_model)
        # get_probs, get_value = load_policy(model_file, 'last')

        multi_ppo(workload_file, args.model, gamma=args.gamma, seed=args.seed, traj_per_epoch=args.trajs, epochs=args.epochs,
            logger_kwargs=logger_kwargs, pre_trained=1, trained_model=os.path.join(model_file, "simple_save"),
            attn=args.attn,
            shuffle=args.shuffle, backfil=args.backfil, skip=args.skip, score_type=args.score_type,
            batch_job_slice=args.batch_job_slice, filter=args.filter)
    else:
        multi_ppo(workload_file, args.model, gamma=args.gamma, seed=args.seed, traj_per_epoch=args.trajs, epochs=args.epochs,
            logger_kwargs=logger_kwargs, pre_trained=0, attn=args.attn, shuffle=args.shuffle, backfil=args.backfil,
            skip=args.skip, score_type=args.score_type, batch_job_slice=args.batch_job_slice, filter=args.filter)
