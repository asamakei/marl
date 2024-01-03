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
    parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=50, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=1000, help="number of episodes")
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
    parser.add_argument("--benchmark-iters", type=int, default=1000, help="number of iterations run for benchmarking")
    parser.add_argument("--movie-fname", type=str, default="movie.mp4", help="name of movie file")
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
    return act, target.state.p_pos

def train(arglist):
    w=h=2
    n=5
    FOLLOWER1 = 0
    FOLLOWER2 = 1
    FOLLOWER3 = 2
    LEADER = 3
    PREY = 4   
    
    def update(action_n, target_edge, agents):
        im = []
        
        scales = [400,400,400,400,166]
        im.append(ax.scatter(x=pts[0:,0], y=pts[0:,1], s=scales, color=np.array(['grey', 'red', 'purple', 'blue', 'green']), alpha=1.0, zorder=100))
        
        #writing border lines
        ddx = np.array([-1,-1,1,1,-1])
        ddy = np.array([-1,1,1,-1,-1])
        im.extend(ax.plot(ddx,ddy,color='green',alpha=0.5,linewidth = 1 ))
        
        '''
        # draw action vectors
        ddx = np.array([])
        ddy = np.array([])
        for i in action_n:
            ddx = np.append(ddx, i[1]-i[2]) # right - left
            ddy = np.append(ddy, i[3]-i[4]) # up - down
        xx = pts[0:5,0]
        yy = pts[0:5,1]
        im.append(ax.quiver(xx,yy,ddx,ddy, scale=7, alpha=0.5))
        '''
        xx = pts[0:5,0]
        yy = pts[0:5,1]
        
        # draw observation radius
        for i in range(3):
            c = patches.Circle(xy=(xx[i],yy[i]), radius=0.2, fill=False)
            im.append(ax.add_patch(c))
            
        # draw prey's target position if we use scripted-prey
        if not arglist.learning_prey:
            px = []
            py = []
            px.append(xx[-1])
            px.append(target_edge[0])
            py.append(yy[-1])
            py.append(target_edge[1])
            im.append(ax.scatter(x=target_edge[0], y=target_edge[1], s=30, color='green'))
            im.extend(ax.plot(px,py,color='green',ls=":"))
        '''
        # draw action vectors
        own_vec = []
        comm_scenario = False
        for i in action_n:
            tv = np.array([i[1]-i[2], i[3]-i[4]])
            #tv /= np.linalg.norm(tv)
            own_vec.append(tv)
        for i, agent in enumerate(agents):
            if hasattr(agent, 'leader_force'):
                comm_scenario = True
                vec =  np.array([agent.leader_force[0], agent.leader_force[1]])
                #vec /= np.linalg.norm(vec)
                #vec *= 1.0
                im.append(ax.quiver(agent.state.p_pos[0], agent.state.p_pos[1], vec[0], vec[1], scale=7, color='blue', alpha=0.7 ))
                # draw synthetic vector
                syn_vec = np.zeros(2)
                syn_vec = vec + own_vec[i]
                #syn_vec /= np.linalg.norm(syn_vec)
                #print(agent.name, 'plot synthetic vector, ', syn_vec)
                im.append(ax.quiver(agent.state.p_pos[0], agent.state.p_pos[1], syn_vec[0], syn_vec[1], scale=7, color='black', alpha=1))
                a = 0.2
            else:
                a = 0.7
            #print(agent.name, 'plot own vector, ', own_vec[i])
            im.append(ax.quiver(agent.state.p_pos[0], agent.state.p_pos[1], own_vec[i][0], own_vec[i][1], scale=7, color='black', alpha=a))
        '''
        # draw leader actions bar
        if comm_scenario:
            original = False
            for agent in agents:
                if agent.advleader:
                    if hasattr(agent, 'org'): #for original implement of first submitted journal
                        original = True
            if original:
                y1 = action_n[3][5:]
                short_w = 0.1
                long_w = 0.2
                cmd = np.argmax(np.array(y1))
                label = ['None', 'Go Left', 'Go Right', 'Go Down', 'Go Up', 'Come on', 'Go Away']
                slabel = ['None', 'Left', 'Right', 'Down', 'Up', 'Come', 'Away']
                x1 = [0.1,0.3,0.5,0.7,0.9,1.1,1.3]
                clr = ['grey' for _ in range(len(label))]
                selclr = ['grey', 'blue', 'blue', 'red', 'red', 'green', 'green']
                clr[cmd] = selclr[cmd]
                im.extend(ax2.bar(x1, y1, short_w, color=clr, alpha=1, edgecolor='black'))
                im.append(ax2.text(1.5, 0.4, label[cmd], size=15, color='black', alpha=1))
                ax2.set_ylim(0,1)
                ax2.set_xlim(0,2)
                ax2.set_title('leader\'s command', fontsize=10)
                ax2.set_xticks(x1)
                ax2.set_xticklabels(slabel, rotation=45, size=8)
            else: # New leader force
                la = action_n[3]
                short_w = 0.1
                long_w = 0.2
                y1 = [la[5], la[6], -la[7], la[8], -la[9], la[10], -la[11]]
                x1 = [0, 0.6, 0.8, 1.6, 1.8, 2.6,2.8]
                clr = ['grey', 'blue', 'blue', 'red', 'red', 'green', 'green']
                im.extend(ax2.bar(x1, y1, short_w, color=clr, alpha=0.3))
                y2 = [y1[1]+y1[2], y1[3]+y1[4], y1[5]+y1[6]]
                x2 = [1, 2, 3]
                clr = ['blue', 'red', 'green']
                im.extend(ax2.bar(x2, y2, long_w, color=clr, alpha=1, edgecolor='black'))
                ax2.set_xticks(x2)
                l=['x', 'y', 'relative']
                ax2.set_ylim(-1,1)
                ax2.set_title('leader\'s command', fontsize=10)
                ax2.set_xticklabels(l)
            

        return im
    
    with U.single_threaded_session():
        bench_step = 0
        agent_info = np.zeros(5)
        # Create environment
        env = make_env(arglist.scenario, arglist, False)
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
        
        curriculum = True
        tgt_sel = 0
        episode_no = 0

        obs_n = env.reset()
        good_act = np.array([0.0 for _ in range(5)])
        prey_maintain_duration = 5
        print('Starting iterations for making movie...')
        
        
        tgt_sel = 0
        first_time = True

        while True:
            # get action
            action_n = [agent.action(obs) for agent, obs in zip(trainers,obs_n)]
            #[do nothing,right,left,up,down]
            
            if True:
                good_act, target_edge = good_agent_victim_action(env.world.agents[-1],env.world, env.world.agents[tgt_sel])
                if bench_step % 15 == 0:
                    tgt_sel += 1
                    if tgt_sel > 3:
                        tgt_sel = 0
                action_n[-1] = (good_act)
            '''
            if not arglist.learning_prey:
                if bench_step % prey_maintain_duration == 0:
                    good_act, target_edge = good_agent_action(env.world.agents[-1],env.world)
                    #prey_maintain_duration = 5#random.randint(3,7)
                action_n[-1] = (good_act)
            '''
            new_obs_n, rew_n, done_n, info_n = env.step(action_n)
            obs_n = new_obs_n

            #plot
            pts = np.array([])
            for a in range(len(obs_n)):
                pts = np.append(pts, [list(obs_n[a][2:4])])
            pts = pts.reshape((5,2))
            #pts = np.append(pts, [[999,999], [-999,999], [999,-999], [-999,-999]], axis = 0)
            
            if first_time:
                comm_scenario = False
                for agent in env.world.agents:
                    if hasattr(agent, 'leader_force'):
                        comm_scenario = True
                if comm_scenario:
                    #prepare for making movie during benchmark
                    fig, (ax, ax2) = plt.subplots(nrows=2, figsize=(5,5), gridspec_kw={'height_ratios':[5,1]})
                    plt.tight_layout()
                    print ('Comm Scenario')
                else:
                    fig, ax = plt.subplots(figsize=(5,5))
                    print ('Nocom Scenario')
                ax.set_xlim(-1.25,1.25)
                ax.set_ylim(-1.25,1.25)
                ax.set_aspect(1)
                first_time = False
        
            im = update(action_n, target_edge, env.world.agents)
            ims.append(im)
            
            if bench_step % arglist.max_episode_len == 0:
                obs_n = env.reset()     
                print (bench_step, arglist.max_episode_len)
            if bench_step > arglist.benchmark_iters:
                print('start making movie of ',arglist.movie_fname)
                ani = animation.ArtistAnimation(fig, ims, interval=0, blit=True)
                ani.save(arglist.movie_fname, fps=10)
                print('making movie done.')
                break
            bench_step += 1
            

if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)
