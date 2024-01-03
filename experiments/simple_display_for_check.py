import argparse
import numpy as np
import os
import tensorflow as tf
import time
import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer
import tensorflow.contrib.layers as layers
import random

#import vl_class as vl


def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=100, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=10, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=4, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="ddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="ddpg", help="policy of adversaries")
    parser.add_argument("--curriculum", action="store_true", default=False)
    parser.add_argument("--learning-prey", action="store_true", default=False)
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default=None, help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="/tmp/policy/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes are completed")
    parser.add_argument("--load", type=str, default="", help="file path in which training state and model are loaded")
    parser.add_argument("--log-rate", type=int, default=50, help="save log.csv once every time this many episodes are completed")
    parser.add_argument("--log-name", type=str, default="log.csv", help="name of log file")
    # Evaluation
    return parser.parse_args()

def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out

def make_env(scenario_name, arglist, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario(observation_radius=1)
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data,observation_radius=1)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, observation_radius=1)
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

def good_agent_action(agent,world, idx):
    world_edge = [[-0.9,0.9],[0.9,0.9],[-0.9,-0.9],[0.9,-0.9]]
    '''
    other_pos = []
    for other in world.agents:
        if agent == other: continue
        other_pos.append(other.state.p_pos)
    dis = [[0.0] for _ in range(len(world_edge))]
    for i,edge in enumerate(world_edge):
        for pos in other_pos:
            dis[i] += np.linalg.norm(edge - pos)
    idx = dis.index(max(dis))
    '''
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
    return act

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
    np.random.seed(0)
        
    with U.single_threaded_session():
        # Create environment
        env = make_env(arglist.scenario, arglist, False)
        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
        print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))

        # Initialize
        U.initialize()
        
        curriculum = arglist.curriculum
        tgt_sel = 0
        episode_no = 0

        fp = os.path.dirname(arglist.load)
        if not os.path.exists(fp):
            print('Couldn\'t find the load file, this should not be happend.')
            exit()

        print('\n ------ replaying learnt policy of ', arglist.load)
        U.load_state(arglist.load)

        episode_rewards = [0.0]  # sum of rewards for all agents
        agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
        #final_ep_rewards = []  # sum of rewards for training curve
        #final_ep_ag_rewards = []  # agent rewards for training curve
        agent_info = [[[]]]  # placeholder for benchmarking info
        #saver = tf.train.Saver(max_to_keep=None)
        obs_n = env.reset()
        episode_step = 0
        train_step = 0
        #t_start = time.time()

        #x=[]
        #good_agent_reward_list = []
        #adv_rewards_list = [[] for i in range(num_adversaries)]
        #TEST
        ttt = 0.076 
        env.world.agents[0].state.p_pos = np.array([ttt, ttt])
        env.world.agents[1].state.p_pos = np.array([-ttt, ttt])
        env.world.agents[2].state.p_pos = np.array([ttt, -ttt])
        env.world.agents[3].state.p_pos = np.array([-ttt, -ttt])
        #TEST

        good_act = np.array([0.0 for _ in range(5)])

        prey_maintain_duration = 20
        print('Starting iterations...please push enter...')
        env.render()
        input()
        idx = 0
                
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
                action_n[-1] = (good_act)
            elif not arglist.learning_prey:
                if train_step % prey_maintain_duration == 0:
                    good_act = good_agent_action(env.world.agents[-1],env.world, idx)
                    idx -= 1
                    if idx < -1:
                        idx = 0
                action_n[-1] = (good_act)
            #TEST
            for i in range(3):
                action_n[i] = np.zeros(5)
            action_n[3] = np.array([0,0,0,0,0,   0,0,0,0,0,0,0])
            #TEST
            new_obs_n, rew_n, done_n, info_n = env.step(action_n)
            print ('environment has been stepped with leader action: {}'.format(np.round(action_n[-2],3)))
            input()
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
                agent_info.append([[]])
                
            # increment global step counter
            train_step += 1

            episode_no = len(episode_rewards)
            # for displaying learned policies
            time.sleep(0.1)
            env.render()

                
            # update all trainers, if not in display or benchmark mode
            loss = None
            for agent in trainers:
                agent.preupdate()
            for agent in trainers:
                loss = agent.update(trainers, train_step)

            # saves final episode reward for plotting training curve later
            if episode_no > arglist.num_episodes:
                '''
                rew_file_name = arglist.plots_dir + arglist.exp_name + '_rewards_' + str(episode_no) + '.pkl'
                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_rewards, fp)
                agrew_file_name = arglist.plots_dir + arglist.exp_name + '_agrewards_' + str(episode_no) + '.pkl'
                with open(agrew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_ag_rewards, fp)
                #print('...Finished total of {} episodes.'.format(episode_no))
                '''
                break


if __name__ == '__main__':
    '''
    temp = vl.make_visual_logs("hoge")
    #temp.get()
    temp.set("hage")
    temp.get()
    exit()
    '''
    arglist = parse_args()
    train(arglist)
