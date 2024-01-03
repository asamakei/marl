import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario

# Scenario explanation
# Leader has force of follower

class Scenario(BaseScenario):
    def __init__(self, observation_radius=1.0):

        self.obs_r = observation_radius # ??? どの観測半径?
        
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 7 # ???
        world.target_sens = 0.0 # ???
        num_good_agents = 1 # preyの数(const)
        num_superleader = 1 # super_leaderの数
        num_leader = 2 # leaderの数
        num_follower = 4 # followerの数
        num_adversaries = num_superleader + num_leader + num_follower # predatorの数(const)
        num_agents = num_adversaries + num_good_agents # 合計のエージェント数
        num_landmarks = 0 # ???(const)
        self.num_good = num_good_agents # preyの数(実際に使う)
        self.num_adv = num_adversaries # predatorの数(実際に使う)
        self.num_landmarks = num_landmarks # ???(実際に使う)
        # add agents
        world.agents = [Agent() for i in range(num_agents)] # エージェントのリスト(空エージェント)
        for i, agent in enumerate(world.agents): # 各エージェントについて設定する
            agent.name = 'agent %d' % i # エージェント名
            agent.collide = True # 当たり判定をありにする?
            # follower, leader, preyごとに設定する
            if i < num_follower: # 0, 1, 2, 3: adversary followers
                agent.adversary = True # predatorかどうか
                agent.good = False # preyかどうか
                agent.silent = True # Listener 指示を出さない
                agent.advfollower = True # predatorのFollowerかどうか
                agent.advleader = False # predatorのLeaderかどうか
                agent.advsuperleader = False # predatorのsuperleaderかどうか
                agent.obs_r = 0.2 # 観測可能半径
                agent.leader_force = np.zeros(2) # リーダーからの強制力ベクトル(最初はゼロ)
                agent.force_id = -1 # この値が同じエージェントに指示を出す
                agent.forced_id = 1 # この値が同じエージェントから指示を受ける
                if i < num_follower - 2: agent.forced_id = 2
            elif i < num_follower + num_leader: # 4, 5: adversary leader
                agent.adversary = True
                agent.good = False
                agent.silent = False # Speaker 指示を出す
                agent.advfollower = False
                agent.advleader = True
                agent.advsuperleader = False
                agent.obs_r = 0.4
                agent.leader_force = np.zeros(2) # リーダーからの何らかの指示(とりあえずゼロ)
                agent.force_id = 1 # この値が同じエージェントに指示を出す
                if i < num_follower + num_leader - 1: agent.force_id = 2
                agent.forced_id = 0 # この値が同じエージェントから指示を受ける
            elif i < num_adversaries: # 6: adversary super leader
                agent.adversary = True
                agent.good = False
                agent.silent = False
                agent.advfollower = False
                #agent.advleader = False
                #agent.advsuperleader = True
                agent.advleader =  True
                agent.advsuperleader = True
                agent.obs_r = 0.5
                agent.leader_force = np.zeros(2) # リーダーからの強制力ベクトル(最初はゼロ)
                agent.force_id = 0 # この値が同じエージェントに指示を出す
                agent.forced_id = -1 # この値が同じエージェントから指示を受ける
            else: # 7: prey
                agent.adversary = False
                agent.good = True
                agent.silent = True
                agent.advfollower = False
                agent.advleader = False
                agent.advsuperleader = False
                agent.obs_r = 10
                agent.target_pos = np.zeros(2) # 向かう場所の位置ベクトル(最初はゼロ)
                agent.force_id = -1 # この値が同じエージェントに指示を出す
                agent.forced_id = -1 # この値が同じエージェントから指示を受ける

            agent.size = 0.075 if agent.adversary else 0.05 # 丸のサイズ設定(preyは小さい)
            agent.accel = 3.0 if agent.adversary else 4.0 # 移動の加速度の設定(preyは速い)
            #agent.accel = 20.0 if agent.adversary else 25.0 # 速いモード？
            agent.max_speed = 1.0 if agent.adversary else 1.3 # 最大速度の設定(preyは速い)
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)] # Landmarkのリスト
        for i, landmark in enumerate(world.landmarks): # Landmarkの設定
            landmark.name = 'landmark %d' % i # Landmarkの名前
            landmark.collide = True # 衝突判定
            landmark.movable = False # 動くかどうか
            landmark.size = 0.2 # サイズ
            landmark.boundary = False # 境界???
        # make initial conditions
        self.reset_world(world) # 初期化をする
        
        #for i, agent in enumerate(world.agents):
        #    print (i, agent.name, agent.adversary, agent.good, agent.silent, agent.advleader)
        return world


    def reset_world(self, world):
        #np.random.seed(0)
        # random properties for agents
        for i, agent in enumerate(world.agents): # エージェントの役割ごとに色を設定
            if agent.good: # prey 赤
                agent.color = np.array([1.00, 0.00, 0.00])
            elif agent.advsuperleader: # superleader 紫
                agent.color = np.array([1.00, 0.00, 1.00])
            elif agent.advleader: # leader 緑
                agent.color = np.array([0.00, 1.00, 0.00])
            else: # follower 青
                agent.color = np.array([0.00, 0.00, 1.00])

            # random properties for landmarks
        for i, landmark in enumerate(world.landmarks): # Landmarkの色を設定
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents: # エージェントごとに設定
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p) # 初期位置 次元ごとにランダムに+1と-1を決定する(四つ角？)
            agent.state.p_vel = np.zeros(world.dim_p) # 初期速度ゼロ
            agent.state.c = np.zeros(world.dim_c) # 初期のなんか ???
        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary: # Landmarkごとに初期位置と速度を設定
                landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)


    def benchmark_data(self, agent, world): # 反対の勢力と接触している数を返す(preyとpredator同士)
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

    def is_collision(self, agent1, agent2): # エージェント同士の円が重なっているかどうか
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


    def reward(self, agent, world): # 報酬の設定
        # Agents are rewarded based on minimum agent distance to each landmark
        main_reward = self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)
        return main_reward

    def agent_reward(self, agent, world): # prey側の報酬 predatorの接触数と外に出ている分の和
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

    def adversary_reward(self, agent, world): # predator側の報酬 preyとの接触数と外に出ている分の和
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
            if count >= 4:
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

    def observation(self, agent, world): # あるエージェントが環境を観測する
        comm = []
        other_pos = []
        other_vel = []
        if not agent.adversary: # predatorでなければ色を変える？？真緑
            agent.color =np.array([1.0, 0.0, 0.0])
        for other in world.agents: # 自分以外のエージェントについてそれぞれ以下を行う
            if other is agent: continue #myself
            dis = np.sqrt(np.sum(np.square(other.state.p_pos - agent.state.p_pos))) # 距離を計算
            if agent.adversary: # I am adversary(predator) agent
                # 範囲内のエージェントとの相対位置と速度を取得
                if dis < agent.obs_r: # I can see this other (regardless of Prey or Predator)
                    other_pos.append(other.state.p_pos - agent.state.p_pos)
                    other_vel.append(other.state.p_vel)
                else: # 範囲外ならゼロ
                    other_pos.append(np.array([0,0]))
                    other_vel.append(np.array([0,0]))
                    
                #set colors
                if not other.adversary: # other is good(Prey)
                    if dis < agent.obs_r: # I can see this good(Prey)
                        if agent.advsuperleader: # superleader 紫
                            agent.color = np.array([0.50, 0.00, 0.50])
                        elif agent.advleader: # leader 緑
                            agent.color = np.array([0.00, 0.50, 0.00])
                        else: # follower 青
                            agent.color = np.array([0.00, 0.00, 0.50])
                    else: # good agent is far, cannot observe this other good(Prey)
                        if agent.advsuperleader: # superleader 紫
                            agent.color = np.array([1.00, 0.00, 1.00])
                        elif agent.advleader: # leader 緑
                            agent.color = np.array([0.00, 1.00, 0.00])
                        else: # follower 青
                            agent.color = np.array([0.00, 0.00, 1.00])

            else: # I am good(prey) agent preyなら全ての位置と速度を把握する
                other_pos.append(other.state.p_pos)
                other_vel.append(other.state.p_vel)
        if hasattr(agent, 'leader_force'): # リーダーからの強制力を観測
            comm.append(agent.leader_force) 
        
        if agent.advleader or agent.good: # 観測情報をリストで返す
            return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + other_pos + other_vel)
        else:
            return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + other_pos + other_vel + comm)





















