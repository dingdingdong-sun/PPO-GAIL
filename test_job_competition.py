from multi_HPCSimPickJobs_test import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# plt.rcParams['font.sans-serif']=['SimHei']
# plt.rcParams['axes.unicode_minus']=False

sns.set_style("whitegrid")
sns.set_context("paper")

import warnings
warnings.filterwarnings("ignore")


def test_diff_schedule_uiser_fair(workload_file, model_path, ac_kwargs=dict(), seed=0,
        attn=False,
        shuffle=False,
        backfil=False, skip=False, score_type=0, batch_job_slice=0):

    np.random.seed(seed)

    env = HPCEnv(shuffle=shuffle, backfil=backfil, skip=skip, job_score_type=score_type,
                 batch_job_slice=batch_job_slice, build_sjf=True)
    env.seed(seed)
    env.my_init(workload_file=workload_file, sched_file=model_path)

    x = np.array(env.user_fair_score[0])
    y = np.array(env.user_fair_score[1])
    z = np.array(env.user_fair_score[2])
    print(x)
    axes = plt.subplot(1, 3, 1)  # 创建一个1行三列的图片

    sns.distplot(x, ax=axes)
    plt.title("SJF")
    plt.ylabel('number of trajs')
    axes = plt.subplot(1, 3, 2)
    sns.distplot(y, ax=axes)
    plt.title("FCFS")
    plt.xlabel('std user wait ratio distribution of 1000 trajs ')
    axes = plt.subplot(1, 3, 3)
    sns.distplot(z, ax=axes)
    plt.title("F1")
    plt.show()


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
    parser.add_argument('--batch_job_slice', type=int, default=1512)
    args = parser.parse_args()

    from src.spinup.spinup.utils.run_utils import setup_logger_kwargs

    # build absolute path for using in hpc_env.
    current_dir = os.getcwd()
    workload_file = os.path.join(current_dir, args.workload)
    log_data_dir = os.path.join(current_dir, './data/logs/')
    logger_kwargs = setup_logger_kwargs(args.exp_name, seed=args.seed, data_dir=log_data_dir)

    test_diff_schedule_uiser_fair(workload_file, args.model, seed=args.seed,  attn=args.attn, shuffle=args.shuffle,
                  backfil=args.backfil,
                  skip=args.skip, score_type=args.score_type, batch_job_slice=args.batch_job_slice)