import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario

# Scenario explanation
# Leader has force of follower
# 敵は逃げない 1体 Leaderの報酬はFollowerの報酬の和とする
class Scenario(BaseScenario):
    def __init__(self, observation_radius=1.0):
        self.obs_r = observation_radius # ??? どの観測半径?
        
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 7 # 命令の次元数
        world.target_sens = 0.0
        num_good_agents = 0 # 動く標的の数
        num_superleader = 0 # superleaderの数
        num_leader = 1 # leaderの数
        num_follower = 1 # followerの数
        num_adversaries = num_superleader + num_leader + num_follower # 捜索エージェントの総数
        num_agents = num_adversaries + num_good_agents # 合計のエージェント数
        num_landmarks = 10 # 動かない標的の数
        self.num_good = num_good_agents
        self.num_adv = num_adversaries
        self.num_landmarks = num_landmarks
        zoom_rate = 0.5 # 観測範囲とエージェントサイズの拡大率(フィールドを広くする際に調整)

        # add agents
        world.agents = [Agent() for i in range(num_agents)] # エージェントのリスト(空エージェント)
        for i, agent in enumerate(world.agents): # 各エージェントについて設定する
            agent.name = 'agent %d' % i # エージェント名
            agent.collide = True # 当たり判定をありにする

            # follower, leader, preyごとに設定する
            if i < num_follower: # Followers
                agent.adversary = True # 捜索側かどうか
                agent.good = False # 標的かどうか
                agent.silent = True # 指示を出さないかどうか
                agent.advfollower = True # Followerかどうか
                agent.advleader = False # Leaderかどうか
                agent.advsuperleader = False # SuperLeaderかどうか
                agent.obs_r = 0.2*zoom_rate # 観測可能半径
                agent.leader_force = np.zeros(2) # リーダーからの強制力ベクトル(最初はゼロ)
                agent.force_id = -1 # この値が同じエージェントに指示を出す(-1で無し)
                agent.forced_id = 1 # この値が同じエージェントから指示を受ける(-1で無し)

            elif i < num_follower + num_leader: # Leaders
                agent.adversary = True
                agent.good = False
                agent.silent = False
                agent.advfollower = False
                agent.advleader = True
                agent.advsuperleader = False
                agent.obs_r = 0.64*zoom_rate
                agent.leader_force = np.zeros(2)
                agent.force_id = 1
                agent.forced_id = -1

            elif i < num_adversaries: # SuperLeader
                agent.adversary = True
                agent.good = False
                agent.silent = False
                agent.advfollower = False
                agent.advleader =  True # もろもろの都合上Trueにしておく
                agent.advsuperleader = True
                agent.obs_r = 0.5*zoom_rate
                agent.leader_force = np.zeros(2)
                agent.force_id = -1
                agent.forced_id = -1

            else: # 7: 動くエージェントの設定(使わない)
                agent.adversary = False
                agent.good = True
                agent.silent = True
                agent.advfollower = False
                agent.advleader = False
                agent.advsuperleader = False
                agent.obs_r = 10/2
                agent.target_pos = np.zeros(2) # 向かう場所の位置ベクトル(最初はゼロ)
                agent.force_id = -1 # この値が同じエージェントに指示を出す
                agent.forced_id = -1 # この値が同じエージェントから指示を受ける

            agent.size = 0.075 * zoom_rate # エージェントのサイズ設定
            agent.accel = 3.0 # 移動の際の加速度の設定
            agent.max_speed = 1.0 # 最大速度の設定

        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)] # Landmark(動かない標的)のリスト
        for i, landmark in enumerate(world.landmarks): # 標的の設定
            landmark.name = 'landmark %d' % i # 名前
            landmark.collide = True # 衝突判定あり
            landmark.movable = False # 動くかどうか
            landmark.size = 0.05 * zoom_rate # サイズ
            landmark.boundary = False
            landmark.adversary = False
            landmark.good = True
            landmark.silent = True
            landmark.advfollower = False
            landmark.advleader = False
            landmark.advsuperleader = False

        # make initial conditions
        self.reset_world(world) # 初期化をする
        
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
            landmark.color = np.array([1.00, 0.00, 0.00])
        # set random initial states
        for agent in world.agents: # エージェントごとに設定
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p) # 初期位置
            agent.state.p_vel = np.zeros(world.dim_p) # 初期速度ゼロ
            agent.state.c = np.zeros(world.dim_c) # 指示内容は空っぽに初期化
        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary: # Landmarkごとに初期位置と速度を設定
                landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
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
        return self.adversary_reward(agent, world)

    def adversary_reward(self, agent, world): # predator側の報酬 preyとの接触数と外に出ている分の和
        # Adversaries are rewarded for collisions with agents
        rew = 0
        landmarks = world.landmarks
        adversary_agents = world.agents
        
        for a in landmarks:
            if not agent.advleader:
                if self.is_collision(a,agent):rew+=10
            elif not agent.advsuperleader:
                for other in adversary_agents:
                    if other.forced_id == agent.force_id and self.is_collision(a,other):rew+=10
            else:
                for other in adversary_agents:
                    if (not other.advleader) and self.is_collision(a,other):rew+=10

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
        target_dis = []
        target_pos = []
        target_zeros = []
        sq_rad = agent.obs_r * agent.obs_r

        # 他エージェントの位置と速度
        for other in world.agents:
            if other is agent: continue
            sq_dis = np.sum(np.square(other.state.p_pos - agent.state.p_pos)) # 二乗距離を計算
            # 他エージェントとの相対位置と速度を取得
            if (sq_dis < sq_rad  #範囲内
                or (agent.force_id >= 0 and agent.force_id == other.forced_id) # 指示を出す対象
                or (agent.forced_id >= 0 and agent.forced_id == other.force_id) # 指示を受ける対象
                ):
                other_pos.append(other.state.p_pos - agent.state.p_pos)
                other_vel.append(other.state.p_vel)
            else: # 範囲外ならゼロ
                other_pos.append(np.array([0,0]))
                other_vel.append(np.array([0,0]))

        # 標的の観測と色セット(visualizer用)
        for other in world.landmarks:
            sq_dis = np.sum(np.square(other.state.p_pos - agent.state.p_pos)) # 二乗距離を計算
            #set colors
            if sq_dis < sq_rad: # 標的が範囲内
                if agent.advsuperleader: # superleader 紫
                    agent.color = np.array([0.50, 0.00, 0.50])
                elif agent.advleader: # leader 緑
                    agent.color = np.array([0.00, 0.50, 0.00])
                else: # follower 青
                    agent.color = np.array([0.00, 0.00, 0.50])
                target_dis.append((sq_dis,other.state.p_pos - agent.state.p_pos))

            else: # 標的が範囲外
                if agent.advsuperleader: # superleader 紫
                    agent.color = np.array([1.00, 0.00, 1.00])
                elif agent.advleader: # leader 緑
                    agent.color = np.array([0.00, 1.00, 0.00])
                else: # follower 青
                    agent.color = np.array([0.00, 0.00, 1.00])
                target_zeros.append(np.array([0,0]))

        # 距離の近い順にソート
        target_dis.sort(key = lambda x: x[0])
        target_pos = [p for (_,p) in target_dis]
        target_pos = target_pos + target_zeros

        if agent.forced_id >= 0: # リーダーからの強制力があれば観測
            comm.append(agent.leader_force) 
        
        if agent.forced_id < 0: # 観測情報をリストで返す
            return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + other_pos + other_vel + target_pos[0:3])
        else:
            return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + other_pos + other_vel + comm + target_pos[0:3])





















