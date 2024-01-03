import argparse
import numpy as np
import os
import pickle

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import subprocess
import warnings
import pandas as pd
import csv
import shutil
import copy


def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="JAWSver15", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=50, help="maximum episode length")
    parser.add_argument("--scripted-prey", action="store_true", default=False)
    parser.add_argument("--without-curriculum", action="store_true", default=False)
    parser.add_argument("--num-episodes", type=int, default=100000, help="number of episodes")
    parser.add_argument("--learning-prey", action="store_true", default=False)
    # Core training parameters
    parser.add_argument("--adv-policy", type=str, default="ddpg", help="policy of adversaries")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=128, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default="temp", help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="/tmp/policy/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--load-model-number", type=int, default=0, help='using with --load-dir and --display, specifying the number of model')
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=1000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./results/", help="directory where benchmark data is saved")
    parser.add_argument("--bench-fname", type=str, default="benchmark.csv", help="name of tempraly benchmark file")
    parser.add_argument("--plots-dir", type=str, default="./results/", help="directory where plot data is saved")
    parser.add_argument("--nograph", action="store_true", default=False)
    parser.add_argument("--start-global-counter", type=int, default=0, help="starting global counter")
    parser.add_argument("--end-global-counter", type=int, default=1, help="ending global counter")
    return parser.parse_args()


if __name__ == '__main__':
    arglist = parse_args()
    warnings.filterwarnings('ignore')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    

    # Reset Random seed, we need to set random library's one too
    np.random.seed(0)
    
    global_counter = arglist.start_global_counter
    while(global_counter <= arglist.end_global_counter):
        new_bench_fname = arglist.plots_dir + arglist.exp_name + '/' + arglist.exp_name + '_' + arglist.bench_fname.replace('.csv', '_G'+str(global_counter)+'.csv')
        if not os.path.isdir(arglist.plots_dir + arglist.exp_name):
            os.makedirs(arglist.plots_dir + arglist.exp_name, exist_ok=True)
        with open(new_bench_fname, "w", encoding="utf-8") as f:
            f.write('Global_counter,Episodes,Benchmark socres->,Follower1,Follower2,Leader1,Leader2,SuperLeader,Mutual Collision,Training time course->,mean rew F1, mean rew F2, mean rew L1, mean rew L2, mean rew S, mean rew total,var rew total,min rew total,first quartile rew total,median rew total,third quartile rew total,max rew total, time\n')
        #Training
        train_path = "python -W ignore train_and_eval.py".split(" ")
        train_path.extend(["--scenario", arglist.scenario])
        train_path.extend(["--save-dir", arglist.save_dir])
        train_path.extend(["--load-dir", arglist.save_dir])
        train_path.extend(["--exp-name", arglist.exp_name])
        train_path.extend(["--max-episode-len", str(arglist.max_episode_len)])
        train_path.extend(["--num-episodes", str(arglist.num_episodes)])
        train_path.extend(["--save-rate", str(arglist.save_rate)])
        train_path.extend(["--plots-dir", str(arglist.plots_dir)])
        train_path.extend(["--benchmark-iters", str(arglist.benchmark_iters)])
        train_path.extend(["--bench-fname", str(new_bench_fname)])
        train_path.extend(["--g-counter", str(global_counter)])
        train_path.extend(["--adv-policy", arglist.adv_policy])
        if arglist.learning_prey:
            train_path.extend(["--learning-prey"])
        if arglist.without_curriculum:
            train_path.extend(["--without-curriculum"])
        if arglist.nograph:
            train_path.extend(["--nograph"])
        print ('------------------ Starting meta-training no. ', global_counter, ' ------------------')
        print (' '.join(train_path))
        subprocess.call(train_path)
        
        #Making movie of the best model
        model_path = arglist.save_dir + '_' + arglist.exp_name + str(global_counter)
        mm_path = "python -W ignore making_movie.py".split(" ")
        mm_path.extend(["--scenario", arglist.scenario])
        mm_path.extend(["--load", model_path])
        mm_path.extend(["--exp-name", arglist.exp_name])
        mm_path.extend(["--adv-policy", arglist.adv_policy])
        mm_path.extend(["--movie-fname", str(new_bench_fname.replace('.csv','.mp4'))])
        if arglist.learning_prey:
            mm_path.extend(["--learning-prey"])
        print (mm_path)
        print ('movie encoding....', end="")
        subprocess.call(mm_path)
        print ('done.')
        
        global_counter += 1
