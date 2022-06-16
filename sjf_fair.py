# import numpy as np
# import tensorflow as tf
# import gym
# import os
# import sys
# import time
# from src.spinup.spinup.utils.logx import EpochLogger
# from src.spinup.spinup.utils.mpi_tf import MpiAdamOptimizer, sync_all_params
# from src.spinup.spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
# from src.spinup.spinup.utils.logx import restore_tf_graph
# import os.path as osp
import matplotlib.pyplot as plt
import numpy as np

from multi_HPCSimPickJobs import *


def test_fcfs_fair(workload_file, model_path, ac_kwargs=dict(), seed=0,
        attn=False,
        shuffle=False,
        backfil=False, skip=False, score_type=0, batch_job_slice=0):

    np.random.seed(seed)

    env = HPCEnv(shuffle=shuffle, backfil=backfil, skip=skip, job_score_type=score_type,
                 batch_job_slice=batch_job_slice, build_sjf=False)
    env.seed(seed)
    env.my_init(workload_file=workload_file, sched_file=model_path)

    # Share information about action space with policy architecture
    ac_kwargs['action_space'] = env.action_space
    ac_kwargs['attn'] = attn

    mean = []
    std = []


    [o, co, user_index_list, done], d, ep_ret, ep_len, show_ret, sjf, f1 = env.reset_for_filter(), False, 0, 0, 0, 0, 0

    num_total = 0
    while not done:
        # print("start index is {}".format(env.start))
        t = 0
        first_job = 0
        for i in range(0, MAX_QUEUE_SIZE * JOB_FEATURES, JOB_FEATURES):
            if all(o[i:i + JOB_FEATURES] == [0] + [1] * (JOB_FEATURES - 2) + [0]):
                continue
            elif all(o[i:i + JOB_FEATURES] == [1] * JOB_FEATURES):
                continue
            else:
                if(o[i+2] < o[first_job*JOB_FEATURES+2]):
                    first_job = int(i/JOB_FEATURES)

        num_total += 1
        o, r, d, r2, sjf_t, f1_t = env.step_for_fair(first_job)




            #
            # print("mean for traj {} is: {}".format(t, m))
            # print("std for traj {} is: {}".format(t, st))


        if d:
            t += 1
            wps = np.array(r)
            m = np.mean(wps)
            st = np.std(wps)

            mean.append(m)
            std.append(st)
            [o, co, user_index_list, done], d, ep_ret, ep_len, show_ret, sjf, f1 = env.reset_for_filter(), False, 0, 0, 0, 0, 0
            r = [0.0 for _ in range(len(user_index_list))]
    std = np.array(std)

    mid = np.median(std)
    mean = np.mean(mid)
    activate_index = []
    for l in range(len(std)):
        if std[l] >= mid and std[l] <= 2*mean:
            activate_index.append(l+1)

    np.save("std_info",std)

    with open("first_filter_index.txt", "a+") as f:
        f.write(str(activate_index[:]))


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

    test_fcfs_fair(workload_file, args.model, seed=args.seed,  attn=args.attn, shuffle=args.shuffle,
                  backfil=args.backfil,
                  skip=args.skip, score_type=args.score_type, batch_job_slice=args.batch_job_slice)