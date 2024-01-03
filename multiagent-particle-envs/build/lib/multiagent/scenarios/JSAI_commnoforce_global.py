### 2020.11.23 [for response of "SHOKAI"]
### with Leader force but without force, only with comm

import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario

# Scenario explanation
# Leader has force of follower

class Scenario(BaseScenario):
    def __init__(self, observation_radius=1.0):

        self.obs_r = observation_radius
        
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 7
        world.target_sens = 0.0
        num_good_agents = 1
        num_adversaries = 4
        num_agents = num_adversaries + num_good_agents
        num_landmarks = 0
        self.num_good = num_good_agents
        self.num_adv = num_adversaries
        self.num_landmarks = num_landmarks
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            if i < num_adversaries -1: # 0, 1, 2: adversary followers
                agent.adversary = True
                agent.good = False
                agent.silent = True # Listener
                agent.advleader = False
                agent.obs_r = 0.2
                agent.leader_force = np.zeros(2)
            elif i == num_adversaries -1: # 3: adversary leader
                agent.adversary = True
                agent.good = False
                agent.silent = False # Speaker
                agent.advleader = True
                agent.obs_r = 10
                agent.noforce = True # Flag for activate 2020JA8 response mode, case added no.5
                agent.org = True # Flag for activate original submitted journal force
            else: # 4: prey
                agent.adversary = False
                agent.good = True
                agent.silent = True 
                agent.advleader = False
                agent.obs_r = 10
                agent.target_pos = np.zeros(2)
            agent.size = 0.075 if agent.adversary else 0.05
            agent.accel = 3.0 if agent.adversary else 4.0
            #agent.accel = 20.0 if agent.adversary else 25.0
            agent.max_speed = 1.0 if agent.adversary else 1.3
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.2
            landmark.boundary = False
        # make initial conditions
        self.reset_world(world)
        
        #for i, agent in enumerate(world.agents):
        #    print (i, agent.name, agent.adversary, agent.good, agent.silent, agent.advleader)
        return world


    def reset_world(self, world):
        #np.random.seed(0)
        # random properties for agents
        for i, agent in enumerate(world.agents):
            if not agent.adversary:
                agent.color = np.array([0.35, 0.85, 0.35])
            elif agent.advleader:
                agent.color = np.array([0.20, 0.00, 0.20])
            else:
                agent.color = np.array([0.85, 0.35, 0.35])
            # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)


    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        collisions = 0
        if agent.adversary:
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else: #agent is good(prey)
            for a in self.adversaries(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]


    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        main_reward = self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)
        return main_reward

    def agent_reward(self, agent, world):
        # Agents are negatively rewarded if caught by adversaries
        rew = 0
        adversaries = self.adversaries(world)

        # Negatively rewarded when collide with adversary
        if agent.collide:
            for a in adversaries:
                if self.is_collision(a, agent):
                    rew -= 1

        # agents are penalized for exiting the screen, so that they can be caught by the adversaries
        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)
        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)

        return rew

    def adversary_reward(self, agent, world):
        # Adversaries are rewarded for collisions with agents
        rew = 0
        good_agents = self.good_agents(world)

        count = 0
        for a in good_agents:
            for other in world.agents:
                if other is a: continue #both good
                if self.is_collision(a, other) :
                    count += 1
                    rew += 10
            if count == 4:
                rew += 50
                
        # Matsunami added here because advarsaries are partially observable
        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)
        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)
            
        return rew

    def observation(self, agent, world):
        comm = []
        other_pos = []
        other_vel = []
        if not agent.adversary:
            agent.color =np.array([0.0, 1.0, 0.0])
        for other in world.agents:
            if other is agent: continue #myself
            dis = np.sqrt(np.sum(np.square(other.state.p_pos - agent.state.p_pos)))
            if agent.adversary: # I am adversary(predator) agent
                if dis < agent.obs_r: # I can see this other (regardless of Prey or Predator)
                    other_pos.append(other.state.p_pos - agent.state.p_pos)
                    other_vel.append(other.state.p_vel)
                else:
                    other_pos.append(np.array([0,0]))
                    other_vel.append(np.array([0,0]))
                    
                #set colors
                if not other.adversary: # other is good(Prey)
                    if dis < agent.obs_r: # I can see this good(Prey)
                        if agent.advleader:
                            agent.color = np.array([0.00, 0.00, 1.00])
                        elif (agent.name[-1] == '0'):
                            agent.color = np.array([0.0,0.0,0.0])
                            #agent.color[0] = (int(agent.name[-1])+1)/self.num_adv
                        elif (agent.name[-1] == '1'):
                            agent.color = np.array([1.0,0.0,0.0])
                        elif (agent.name[-1] == '2'):
                            agent.color = np.array([1.0,0.0,1.0])
                    else: # good agent is far, cannot observe this other good(Prey)
                        if agent.advleader:
                            agent.color = np.array([0.00, 0.00, 0.70])
                        elif (agent.name[-1] == '0'):
                            agent.color = np.array([0.3,0.3,0.3])
                        elif (agent.name[-1] == '1'):
                            agent.color = np.array([0.7,0.0,0.0])
                        elif (agent.name[-1] == '2'):
                            agent.color = np.array([0.7,0.0,0.7])

            else: # I am good(prey) agent
                other_pos.append(other.state.p_pos)
                other_vel.append(other.state.p_vel)
        if hasattr(agent, 'leader_force'):
            comm.append(agent.leader_force) 
        
        if agent.advleader or agent.good:
            return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + other_pos + other_vel)
        else:
            return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + other_pos + other_vel + comm)





















