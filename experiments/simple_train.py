import argparse
import gym
import numpy as np
import os
import tensorflow as tf
import time
import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer
import tensorflow.contrib.layers as layers
import random

#Matsunami
import fcntl
import termios
import sys
import csv

import matplotlib
import matplotlib.pyplot as plt

#Matsunami
def getkey():
    fno = sys.stdin.fileno()

    #stdinの端末属性を取得
    attr_old = termios.tcgetattr(fno)

    # stdinのエコー無効、カノニカルモード無効
    attr = termios.tcgetattr(fno)
    attr[3] = attr[3] & ~termios.ECHO & ~termios.ICANON # & ~termios.ISIG
    termios.tcsetattr(fno, termios.TCSADRAIN, attr)

    # stdinをNONBLOCKに設定
    fcntl_old = fcntl.fcntl(fno, fcntl.F_GETFL)
    fcntl.fcntl(fno, fcntl.F_SETFL, fcntl_old | os.O_NONBLOCK)

    chr = 0

    try:
        # キーを取得
        c = sys.stdin.read(1)
        if len(c):
            while len(c):
                chr = (chr << 8) + ord(c)
                c = sys.stdin.read(1)
    finally:
        # stdinを元に戻す
        
        fcntl.fcntl(fno, fcntl.F_SETFL, fcntl_old)
        termios.tcsetattr(fno, termios.TCSANOW, attr_old)
        
    return chr

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=50, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=100000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=4, help="number of adversaries")
    parser.add_argument("--observation-r", type=float, default=1.0, help="radius of observation for each agent")
    parser.add_argument("--good-policy", type=str, default="ddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="ddpg", help="policy of adversaries")
    parser.add_argument("--prey-policy", type=str, default="ddpg", help="policy of adversaries")
    parser.add_argument("--learning-prey", action="store_true", default=False)
    parser.add_argument("--without-curriculum", action="store_true", default=False)
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default=None, help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="/tmp/policy/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")
    parser.add_argument("--log-rate", type=int, default=50, help="save log.csv once every time this many episodes are completed")
    parser.add_argument("--log-name", type=str, default="log.csv", help="name of log file")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--nograph", action="store_true", default=False)
    parser.add_argument("--load-model-number", type=int, default=0, help='using with --load-dir and --display, specifying the number of model')
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=1000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot data is saved")
    parser.add_argument("--bench-name", type=str, default=None)
    parser.add_argument("--leader-manual", action="store_true", default=False)
    parser.add_argument("--bench-fname", type=str, default="benchmark.csv", help="name of tempraly benchmark file")
    parser.add_argument("--g-counter", type=int, default=0, help="Global counter")
    return parser.parse_args()

def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out

def reset_world_for_benchmark( world):
    np.random.seed(0)
    # random properties for agents
    for i, agent in enumerate(world.agents):
        if not agent.adversary:
            agent.color = np.array([0.35, 0.85, 0.35])
        elif agent.advleader:
            agent.color = np.array([0.20, 0.00, 0.20])
        else:
            agent.color = np.array([0.85, 0.35, 0.35])
    # set random initial states
    for i, agent in enumerate(world.agents):# 0, 1, 2: adversary followers # 3: adversary leader # 4: prey
        if i == 0:
            agent.state.p_pos = np.array([-0.5,-0.5])
        elif i == 1:
            agent.state.p_pos = np.array([-0.5,+0.5])
        elif i == 2:
            agent.state.p_pos = np.array([+0.5,-0.5])
        elif i == 3:
            agent.state.p_pos = np.array([+0.5,+0.5])
        elif i == 4:
            agent.state.p_pos = np.array([+0.0,+0.0])
        agent.state.p_vel = np.zeros(world.dim_p)
        agent.state.c = np.zeros(world.dim_c)
        
def make_env(scenario_name, arglist, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario(observation_radius=arglist.observation_r)
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, reset_world_for_benchmark, scenario.reward, scenario.observation, scenario.benchmark_data,observation_radius=arglist.observation_r)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, observation_radius=arglist.observation_r)
    return env

def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    trainers = []
    model = mlp_model
    trainer = MADDPGAgentTrainer
    for i in range(num_adversaries):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.adv_policy=='ddpg')))
    for i in range(num_adversaries, env.n):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.good_policy=='ddpg')))
    return trainers

def good_agent_action(agent,world):
    world_edge = [[-0.9,0.9],[0.9,0.9],[-0.9,-0.9],[0.9,-0.9]]
    other_pos = []
    for other in world.agents:
        if agent == other: continue
        other_pos.append(other.state.p_pos)
    dis = [[0.0] for _ in range(len(world_edge))]
    for i,edge in enumerate(world_edge):
        for pos in other_pos:
            dis[i] += np.linalg.norm(edge - pos)
    idx = dis.index(max(dis))
    target_edge = world_edge[idx]
    epsilon = 0.0
    if random.random() < epsilon:
        target_edge = world_edge[int(random.random()*len(world_edge))]
    if hasattr(agent, 'target_pos'):
        agent.target_pos = target_edge
    vec = target_edge - agent.state.p_pos 
    vec = [vec[0]/2, vec[1]/2]
    act = np.array([0.0 for _ in range(5)])
    # print(agent.state.p_pos)
    # print(target_edge)
    mag = 1
    if vec[0] < 0:
        act[2] = min(abs(vec[0])*mag,1.3)
    else:
        act[1] = min(vec[0]*mag,1.3)
    if vec[1] < 0:
        act[4] = min(abs(vec[1])*mag,1.3)
    else:
        act[3] = min(vec[1]*mag,1.3)
    return act, target_edge

def good_agent_victim_action(agent,world,target):
    vec = target.state.p_pos - agent.state.p_pos
    vec = [vec[0]/2, vec[1]/2]
    act = np.array([0.0 for _ in range(5)])
    mag = 7
    if vec[0] < 0:
        act[2] = min(abs(vec[0])*mag,1.3)
    else:
        act[1] = min(vec[0]*mag,1.3)
    if vec[1] < 0:
        act[4] = min(abs(vec[1])*mag,1.3)
    else:
        act[3] = min(vec[1]*mag,1.3)
    #print ('vec', vec)
    #print ('act', act)
    return act

def train(arglist):
               
    with U.single_threaded_session():
        # Reset Random seed, we need to set random library's one too
        #np.random.seed(1)
        
        # Create environment
        env = make_env(arglist.scenario, arglist, False)
        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
        print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))

    
        # Initialize
        U.initialize()
        
        if arglist.without_curriculum:
            curriculum = False
        else:
            curriculum = True
        tgt_sel = 0
        episode_no = 0

        # Load previous results, if necessary
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir

        episode_rewards = [0.0]  # sum of rewards for all agents
        agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        agent_info = np.zeros(5)  # placeholder for benchmarking info
        saver = tf.train.Saver(max_to_keep=None)
        obs_n = env.reset()
        episode_step = 0
        train_step = 0
        t_start = time.time()
        max_score = 0

        x=[]
        good_agent_reward_list = []
        adv_rewards_list = [[] for i in range(num_adversaries)]

        good_act = np.array([0.0 for _ in range(5)])

        print('Starting iterations...')
        prey_maintain_duration = 5 #10
                
        while True:
            # get action
            action_n = [agent.action(obs) for agent, obs in zip(trainers,obs_n)]
            #[,right,left,up,down]
            if curriculum:
                good_act = good_agent_victim_action(env.world.agents[-1],env.world, env.world.agents[tgt_sel])
                if train_step % 15 == 0:
                    tgt_sel += 1
                    if tgt_sel > 3:
                        tgt_sel = 0
                if episode_no > 5000 or arglist.load_model_number > 5000:
                    curriculum = False
                action_n[-1] = (good_act)
            elif not arglist.learning_prey:
                if train_step % prey_maintain_duration == 0:
                    good_act, _ = good_agent_action(env.world.agents[-1],env.world)
                action_n[-1] = (good_act)
                
            #DEBUG
            #print ('ACTION ',action_n)
            #print ('OBS ',obs_n)
            # environment step
                
            new_obs_n, rew_n, done_n, info_n = env.step(action_n)
            episode_step += 1
            done = all(done_n)
            terminal = (episode_step >= arglist.max_episode_len)
            # collect experience
            for i, agent in enumerate(trainers):
                agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)
            obs_n = new_obs_n

            for i, rew in enumerate(rew_n):
                episode_rewards[-1] += rew
                agent_rewards[i][-1] += rew

            if done or terminal:
                obs_n = env.reset()
                episode_step = 0
                episode_rewards.append(0)
                for a in agent_rewards:
                    a.append(0)
                
            # increment global step counter
            train_step += 1

            episode_no = len(episode_rewards)

            # for displaying learned policies
            key = getkey()
            if key: # temporalily display while training
                env.render()
                time.sleep(0.1)
                
            # update all trainers, if not in display or benchmark mode
            loss = None
            for agent in trainers:
                agent.preupdate()
            for agent in trainers:
                loss = agent.update(trainers, train_step)
                
            if terminal and (len(episode_rewards) % 100 == 0):
                print("G", arglist.g_counter, " ", arglist.exp_name, " at episodes: {}, time: {}".format(
                    len(episode_rewards), round(time.time()-t_start, 3)))

            # save model, display training output
            if terminal and (len(episode_rewards) % arglist.save_rate == 0):
                print('...Finished {} episodes. '.format(episode_no))
                #Start benchmarking
                print ('start benchmarking...')
                bench_step = 0
                agent_info = np.zeros(5) 
                # sotre current arglist.max_episode_len
                max_episode_len_for_restore = arglist.max_episode_len
                arglist.max_episode_len = arglist.benchmark_iters
                
                #Evaluation/Benchmark Loop Start-------------------------------------------------
                env = make_env(arglist.scenario, arglist, True)
                mutual_collide = 0
                #Prepare for plot of trace
                x, y = [np.array([]) for _ in range(len(env.world.agents))], [np.array([]) for _ in range(len(env.world.agents))]
                while True:
                    # get action
                    action_n = [agent.action(obs) for agent, obs in zip(trainers,obs_n)]
                    if bench_step % prey_maintain_duration == 0:
                        good_act, target_edge = good_agent_action(env.world.agents[-1],env.world)
                    action_n[-1] = (good_act)
                    ##TEST
                    '''
                    for agt in env.world.agents:
                        agt.state.p_pos = np.array([random.random()/100., random.random()/100.])
                        agt.state.p_vel = np.zeros(2)
                    action_n = [np.array([1,0,0,0,0], dtype=float), np.array([1,0,0,0,0], dtype=float), np.array([1,0,0,0,0], dtype=float), np.array([1,0,0,0,0,1,2,3,4,5,6,7], dtype=float), np.array([1,0,0,0,0], dtype=float)]
                    #print(action_n,'\n')
                    '''
                    ##TEST
                    new_obs_n, rew_n, done_n, info_n = env.step(action_n)
                    obs_n = new_obs_n
                    bdone = all(done_n)
                    # Add collision counts
                    agent_info += np.array(info_n['n'])
                    #'Follower1, Follower2, Follower3, Leader, Prey
                    if np.array(info_n['n'])[-1] == num_adversaries: #All predators are colliding with prey
                        mutual_collide += 1
                        print('mutual collision at ',bench_step)
                    if (len(episode_rewards) % (arglist.save_rate * 5) == 0):
                        for i, agt in enumerate(env.world.agents):
                            x[i] = np.append(x[i], agt.state.p_pos[0])
                            y[i] = np.append(y[i], agt.state.p_pos[1])
                    #finish benchmark, writing logs
                    if bench_step > arglist.benchmark_iters:# and bdone:
                        writing_info = np.array([arglist.g_counter])
                        writing_info = np.append(writing_info, [len(episode_rewards)]) #as number of finished episodes
                        writing_info = np.append(writing_info, ["Benchmark socres->"])
                        writing_info = np.append(writing_info, agent_info)
                        writing_info = np.append(writing_info, [mutual_collide])
                        writing_info = np.append(writing_info, ["Training time course->"])
                        writing_info = np.append(writing_info, [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards])
                        writing_info = np.append(writing_info, [np.mean(episode_rewards[-arglist.save_rate:])])
                        writing_info = np.append(writing_info, round(time.time()-t_start, 3))
                        print('Finished benchmarking, at episode: ', episode_no, 'Score: ',agent_info[-1].astype(np.int64), '  in detail, ', agent_info[1:6])
                        with open(arglist.bench_fname, "a", encoding="utf-8") as f:
                            writer = csv.writer(f)
                            writer.writerow(writing_info)
                        #Making plot at 5 times of save_rate +++++++++++++++++++++++
                        '''
                        if (len(episode_rewards) % (arglist.save_rate * 5) == 0):
                            fig, ax = plt.subplots(figsize=(5,5))
                            ax.set_xlim(-1.25,1.25)
                            ax.set_ylim(-1.25,1.25)
                            ax.set_aspect(1)
                            #plt.title('trace of model ' + arglist.exp_name[-1])
                            plt.tick_params(bottom=False,
                                        left=False,
                                        right=False,
                                        top=False)
                            plt.tick_params(labelbottom=False,
                                        labelleft=False,
                                        labelright=False,
                                        labeltop=False)
                            scales = [400,400,400,400,166]
                            colors = ['black', 'red', 'purple', 'blue', 'green']
                            #plot
                            #writing border lines
                            ddx = np.array([-1,-1,1,1,-1])
                            ddy = np.array([-1,1,1,-1,-1])
                            ax.plot(ddx,ddy,color='green',alpha=0.5,linewidth = 1 )
                            for i, c, size in zip(range(len(env.world.agents)), colors, scales):
                                plt.scatter(x=x[i], y=y[i], s=size, color=c, alpha=0.1)
                        '''
                        break
                    bench_step += 1
                #Restore max_episode_len
                arglist.max_episode_len = max_episode_len_for_restore 
                env = make_env(arglist.scenario, arglist, False)
                #writing plot
                '''
                if (len(episode_rewards) % (arglist.save_rate * 5) == 0):
                    plt.savefig(arglist.bench_fname.replace('.csv', '_ep'+str(len(episode_rewards))+'_trace.png'))
                    plt.clf()
                '''
                #Finish Evaluation/Becnmark Loop -------------------------------------------------
                
                if True:#max_score <= agent_info[-1]: #always save to keep updating
                    print ('############### Got the best result, saving model!')
                    max_score = agent_info[-1]
                    U.save_state(arglist.save_dir+'_'+arglist.exp_name+str(arglist.g_counter), saver=saver)#, global_step=episode_no)
                #Print statement and update plots
                x.append(episode_no)
                good_agent_reward_list.append(np.mean(agent_rewards[-1][-arglist.save_rate:]))
                for i in range(num_adversaries):
                    adv_rewards_list[i].append(np.mean(agent_rewards[i][-arglist.save_rate:]))
                
                #t_start = time.time()
                # Keep track of final episode reward
                final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
                for rew in agent_rewards:
                    final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))

            if episode_no > arglist.num_episodes:
                break


if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)
