#うっすらボロノイ図を書いた上に、それぞれのエージェントのActionに基づく力の方向を図示したムービーを作る。通信ありの場合は、Leader指示に基づく力も図示。
import argparse
import numpy as np
import os
import tensorflow as tf
import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer
import tensorflow.contrib.layers as layers
import random
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import animation
#plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg' # Add the path of ffmpeg here!!
import matplotlib.patches as patches


def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--file", type=str, default="trace.png")
    parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=200, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=1, help="number of episodes")
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
    parser.add_argument("--exp-name", type=str, default="test", help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="/tmp/policy/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes are completed")
    parser.add_argument("--load", type=str, default="", help="file path in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--benchmark-iters", type=int, default=200, help="number of iterations run for benchmarking")
    return parser.parse_args()

def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out

def reset_world_for_benchmark(world):
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
    '''
    for agent in world.agents:
        agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
        agent.state.p_vel = np.zeros(world.dim_p)
        agent.state.c = np.zeros(world.dim_c)
    '''
                
def make_env(scenario_name, arglist, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario(observation_radius=1)
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, reset_world_for_benchmark, scenario.reward, scenario.observation, scenario.benchmark_data,observation_radius=1)
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
    w=h=2
    n=5
    FOLLOWER1 = 0
    FOLLOWER2 = 1
    FOLLOWER3 = 2
    LEADER = 3
    PREY = 4   
    

    
    with U.single_threaded_session():
        bench_step = 0
        agent_info = np.zeros(5)
        # Create environment
        env = make_env(arglist.scenario, arglist, True)
        mutual_collide = 0
        ims = []
        target_edge = [0,0]
        
        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
        print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))

        # Initialize
        U.initialize()
        
        fp = os.path.dirname(arglist.load)
        if not os.path.exists(fp):
            print('Couldn\'t find the load file, this should not be happend.')
            exit()

        print('\n ------ replaying learnt policy of ', arglist.load)
        U.load_state(arglist.load)

        obs_n = env.reset()
        good_act = np.array([0.0 for _ in range(5)])
        prey_maintain_duration = 5
        print('Starting iterations for making movie...')
        
        filename = arglist.load
        filename = filename[filename.rfind('/')+1:]
        tgt_sel = 0
        first_time = True

        #Prepare for plot of trace
        pts = np.array([])
        first_time =True
        
        while True:
            # get action
            action_n = [agent.action(obs) for agent, obs in zip(trainers,obs_n)]
            #[do nothing,right,left,up,down]
            
            if arglist.curriculum:
                good_act = good_agent_victim_action(env.world.agents[-1],env.world, env.world.agents[tgt_sel])
                if bench_step % 15 == 0:
                    tgt_sel += 1
                    if tgt_sel > 3:
                        tgt_sel = 0
                action_n[-1] = (good_act)
            if not arglist.learning_prey:
                if bench_step % prey_maintain_duration == 0:
                    good_act, target_edge = good_agent_action(env.world.agents[-1],env.world)
                    #prey_maintain_duration = 5#random.randint(3,7)
                action_n[-1] = (good_act)
            new_obs_n, rew_n, done_n, info_n = env.step(action_n)
            obs_n = new_obs_n
            # Add collision counts
            agent_info += np.array(info_n['n'])
            #'Follower1, Follower2, Follower3, Leader, Prey
            if agent_info[-1] == num_adversaries: #All predators are colliding with prey
                mutual_collide += 1
                
            #for plot
            tpts = np.array([])
            for a in range(len(obs_n)):
                tpts = np.append(tpts, obs_n[a][2:4])
            if first_time:
                pts = tpts
                first_time = False
            else:
                pts = np.vstack([pts, tpts])
            
            if bench_step > arglist.benchmark_iters:
                break
            bench_step += 1
        
        #Makaing plot
        fig, (ax, ax2) = plt.subplots(nrows=2, figsize=(5,10), gridspec_kw={'height_ratios':[1,1]})
        ax.set_xlim(-1,1)
        ax.set_ylim(-1,1)
        ax.set_aspect(1)
        ax2.set_xlim(-2,2)
        ax2.set_ylim(-2,2)
        ax2.set_aspect(1)
        ticks = [-1, 0, 1]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax2.set_xticks(ticks)
        ax2.set_yticks(ticks)
        '''
        ax.tick_params(bottom=False,
                    left=False,
                    right=False,
                    top=False)
        ax.tick_params(labelbottom=False,
                    labelleft=False,
                    labelright=False,
                    labeltop=False)
        ax2.tick_params(bottom=False,
                    left=False,
                    right=False,
                    top=False)
        ax2.tick_params(labelbottom=False,
                    labelleft=False,
                    labelright=False,
                    labeltop=False)
        '''
        colors = ['black', 'red', 'purple', 'blue']
        predator_size = 400
        prey_size = 166
        al = 0.2
        for l in range(pts.shape[0]):
            for i, c in zip(range(0,9,2),colors):
                ax.scatter(x=pts[l,i], y=pts[l,i+1],s=predator_size, color=c,alpha=al)
                ax.scatter(x=pts[l,8], y=pts[l,9],s=prey_size, color='green',alpha=al)
        
        for l in range(pts.shape[0]):#len(df)+1):
            for i, c in zip(range(0,9,2),colors):
                ax2.scatter(x=(pts[l,i]-pts[l,8]), y=(pts[l,i+1]-pts[l,9]),s=predator_size, color=c,alpha=al)
        ax2.scatter(x=0, y=0,s=prey_size, color='green',alpha=1)
        '''
        #writing border lines
        ddx = np.array([-1,-1,1,1,-1])
        ddy = np.array([-1,1,1,-1,-1])
        ax2.plot(ddx,ddy,color='black',alpha=0.5,linewidth = 1 )
        '''
        plt.savefig(arglist.file)
        
    
if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)
