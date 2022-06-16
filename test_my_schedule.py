import math

import matplotlib.pyplot as plt
import numpy as np
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
from multi_HPCSimPickJobs_gail import *

def load_policy(trained_model):
    sess = tf.Session()
    model = restore_tf_graph(sess, trained_model)
    # Count variables
    var_counts = tuple(count_vars(scope) for scope in ['pi', 'v'])

    x_ph = model['x']
    a_ph = model['a']
    last_state_a = model['a_last_state']
    last_state_v = model['v_last_state']
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

def test_fcfs_fair(workload_file, model_path, ac_kwargs=dict(), seed=0,
        attn=False,trained_model = None,
        shuffle=False,
        backfil=False, skip=False, score_type=0, batch_job_slice=0):

    np.random.seed(seed)

    with open("first_filter_index.txt", "r") as f:
        tmp = f.readline().strip().split(",")
        start_index = [int(x) for x in tmp]

    env = HPCEnv(shuffle=shuffle, backfil=backfil, skip=skip, job_score_type=score_type,
                 batch_job_slice=batch_job_slice, build_sjf=False)
    env.seed(seed)
    env.my_init(workload_file=workload_file, sched_file=model_path)

    # Share information about action space with policy architecture
    ac_kwargs['action_space'] = env.action_space
    ac_kwargs['attn'] = attn

    mean = []
    std = []
    job_wait_ratio_list = []
    user_waits = []
    job_waits = []
    sess = tf.Session()
    model = restore_tf_graph(sess, trained_model)
    # Count variables
    var_counts = tuple(count_vars(scope) for scope in ['pi', 'v'])

    x_ph = model['x']
    a_ph = model['a']
    output_generator_h = model['a_generator']
    output_expert_h = model['a_expert']
    mask_ph = model['mask']
    adv_ph = model['adv']
    ret_ph = model['ret']
    logp_old_ph = model['logp_old_ph']

    pi = model['pi']
    v = model['v']
    fake_logits = model['fake_logits']
    real_logits = model['real_logits']
    get_reward = model['reward']
    out = model['out']
    # logits = model['logits']
    logp = model['logp']
    logp_pi = model['logp_pi']
    pi_loss = model['pi_loss']
    v_loss = model['v_loss']
    reward_loss = model['reward_loss']
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
    train_reward = tf.get_collection("train_reward")[0]
    # train_pi_optimizer = MpiAdamOptimizer(learning_rate=pi_lr, name='AdamLoad')
    # train_pi = train_pi_optimizer.minimize(pi_loss)
    # train_v_optimizer = MpiAdamOptimizer(learning_rate=vf_lr, name='AdamLoad')
    # train_v = train_v_optimizer.minimize(v_loss)
    # sess.run(tf.variables_initializer(train_pi_optimizer.variables()))
    # sess.run(tf.variables_initializer(train_v_optimizer.variables()))
    # Need all placeholders in *this* order later (to zip with data from buffer)
    all_phs = [x_ph, a_ph, mask_ph, adv_ph, ret_ph, logp_old_ph]
    # Every step, get: action, value, and logprob
    get_action_ops = [pi, v, logp_pi, out]

    ii = 0
    start_time = time.time()
    [o, co], d, ep_ret, ep_len, show_ret, sjf, f1 = env.reset_test(
        ii * JOB_SEQUENCE_SIZE), False, 0, 0, 0, 0, 0

    # Main loop: collect experience in env and update/log each epoch
    num_total = 0
    AVEbsld_list = []
    while not ii == 20:
        t = 0
        lst = []
        mask = []
        for i in range(0, MAX_QUEUE_SIZE * JOB_FEATURES, JOB_FEATURES):
            lst.append(o[i:i + JOB_FEATURES])
            # users_job_index_map[o[i + 4]].append(int(i/JOB_FEATURES))
            if all(o[i:i + JOB_FEATURES] == [0] + [1] * (JOB_FEATURES - 2) + [0]):
                mask.append(0)
            elif all(o[i:i + JOB_FEATURES] == [1] * JOB_FEATURES):
                mask.append(0)
            else:
                mask.append(1)

        s_t = np.array(lst).reshape(-1, MAX_QUEUE_SIZE, JOB_FEATURES)
        mask = np.array(mask).reshape(-1, MAX_QUEUE_SIZE)
        a_generator, v_t, logp_t, out = sess.run(get_action_ops, feed_dict={x_ph: s_t, mask_ph: mask})
        a_out = np.argmax(out, axis=1)
        num_total += 1

        o, r, d, r2, sjf_t, f1_t = env.step(a_out[0])
        # print("动作： {}".format(a_out))
        # print(mask)
        # print(out)

        if d:
            t += 1
            bsld = 0
            for i in range(env.start + 1, env.last_job_in_batch):
                wait = env.loads[env.load_index][i].scheduled_time - env.loads[env.load_index][i].submit_time
                bsld += (wait / float(env.loads[env.load_index][i].run_time)) + 1
                # print(bsld)
            AVEbsld = bsld / JOB_SEQUENCE_SIZE
            print("平均{}".format(AVEbsld))
            AVEbsld_list.append(AVEbsld)

            ii += 1
            print(ii)
            [o, co], d, ep_ret, ep_len, show_ret, sjf, f1 = env.reset_test(
                ii * JOB_SEQUENCE_SIZE), False, 0, 0, 0, 0, 0

    AVEbsld_list = np.mean(np.array(AVEbsld_list))
    print("bsld比均值:{}".format(AVEbsld_list))
    # std = np.array(std)
    #
    # mid = np.median(std)
    # mean = np.mean(mid)
    # activate_index = []
    # for l in range(len(std)):
    #     if std[l] >= mid and std[l] <= 2*mean:
    #         activate_index.append(l+1)


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
    parser.add_argument('--epochs', type=int, default=4000)
    parser.add_argument('--exp_name', type=str, default='ppo')
    parser.add_argument('--pre_trained', type=int, default=0)
    parser.add_argument('--trained_model', type=str, default='./data/logs/ppo_temp/ppo_temp_s0')
    parser.add_argument('--attn', type=int, default=0)
    parser.add_argument('--shuffle', type=int, default=0)
    parser.add_argument('--backfil', type=int, default=0)
    parser.add_argument('--skip', type=int, default=0)
    parser.add_argument('--score_type', type=int, default=0)
    parser.add_argument('--batch_job_slice', type=int, default=100000)
    args = parser.parse_args()

    from src.spinup.spinup.utils.run_utils import setup_logger_kwargs

    # build absolute path for using in hpc_env.
    current_dir = os.getcwd()
    workload_file = os.path.join(current_dir, args.workload)
    log_data_dir = os.path.join(current_dir, './data/logs/')
    logger_kwargs = setup_logger_kwargs(args.exp_name, seed=args.seed, data_dir=log_data_dir)
    model_file = os.path.join(current_dir, args.trained_model)

    test_fcfs_fair(workload_file, args.model, trained_model=os.path.join(model_file, "simple_save"), seed=args.seed,  attn=args.attn, shuffle=args.shuffle,
                  backfil=args.backfil,
                  skip=args.skip, score_type=args.score_type, batch_job_slice=args.batch_job_slice)