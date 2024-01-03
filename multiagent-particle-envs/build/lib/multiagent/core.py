import numpy as np

# physical/external base state of all entites
class EntityState(object):
    def __init__(self):
        # physical position
        self.p_pos = None
        # physical velocity
        self.p_vel = None

# state of agents (including communication and internal/mental state)
class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()
        # communication utterance
        self.c = None

# action of the agent
class Action(object):
    def __init__(self):
        # physical action
        self.u = None
        # communication action
        self.c = None

# properties and state of physical world entity
class Entity(object):
    def __init__(self):
        # name 
        self.name = ''
        # properties:
        self.size = 0.050
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = True
        # material density (affects mass)
        self.density = 25.0
        # color
        self.color = None
        # max speed and accel
        self.max_speed = None
        self.accel = None
        # state
        self.state = EntityState()
        # mass
        self.initial_mass = 1.0
        # observation radius
        self.obs_r = 0.0

    @property
    def mass(self):
        return self.initial_mass

# properties of landmark entities
class Landmark(Entity):
     def __init__(self):
        super(Landmark, self).__init__()

# properties of agent entities
class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()
        # agents are movable by default
        self.movable = True
        # cannot send communication signals
        self.silent = False
        # cannot observe the world
        self.blind = False
        # can only observe the world partically
        self.pblind = True
        # physical motor noise amount
        self.u_noise = None
        # communication noise amount
        self.c_noise = None
        # control range
        self.u_range = 1.0
        # state
        self.state = AgentState()
        # action
        self.action = Action()
        # script behavior to execute
        self.action_callback = None
        # command position for speaker
        self.command_pos = np.zeros(2)

# multi-agent world
class World(object):
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.landmarks = []
        # communication channel dimensionality
        self.dim_c = 0
        # position dimensionality
        self.dim_p = 2
        # color dimensionality
        self.dim_color = 3
        # simulation timestep
        self.dt = 0.1
        # physical damping
        self.damping = 0.25
        # contact response parameters
        self.contact_force = 1e+2
        self.contact_margin = 1e-3
        self.target_sens = 1.0

    # return all entities in the world
    @property
    def entities(self):
        return self.agents + self.landmarks

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    # update state of the world
    def step(self):
        # set actions for scripted agents 
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)
        # gather forces applied to entities
        p_force = [None] * len(self.entities)
        # apply agent physical controls
        p_force = self.apply_action_force(p_force)
        # apply leader's force onto followers --Matsunami
        p_force = self.apply_leader_force(p_force)
        # apply environment forces
        p_force = self.apply_environment_force(p_force)
        # integrate physical state
        self.integrate_state(p_force)
        # update agent state
        for agent in self.agents:
            self.update_agent_state(agent)

    # gather agent action forces
    def apply_action_force(self, p_force):
        # set applied forces
        for i,agent in enumerate(self.agents):
            if agent.movable:
                noise = np.random.randn(*agent.action.u.shape) * agent.u_noise if agent.u_noise else 0.0
                p_force[i] = agent.action.u + noise                
        return p_force
    
    #Matsunami
    def apply_leader_force(self, p_force):
        for agent in self.agents:
            if agent.advleader:
                leader = agent
        if not leader.silent:
            if hasattr(leader, 'pow'): #for pow1 scenario
                mag = leader.pow
            else:
                if hasattr(leader, 'noforce'): #for only comm withouth force, 2020 JA8 response
                    mag = 0.0
                elif hasattr(leader, 'org'): #for original implement of first submitted journal and there is no pow attribute
                    if hasattr(leader, 'half'):
                        mag = 0.5
                    else:
                        mag = 1.0
                else:
                    mag = 0.5 # for altered environment called like JSAI_comm_local, not org/5PMD/pow1
            if hasattr(leader, 'org'): #for original implement of first submitted journal
                if (leader.org == True):
                    for i,agent in enumerate(self.agents):
                        if agent.adversary and not agent.advleader:
                            cmd = np.argmax(np.array(leader.action.c))
                            #print (np.round(leader.action.c,2), 'command is ', cmd)
                            if cmd == 0:
                                agent.leader_force = np.zeros(2)
                            elif cmd == 1:
                                agent.leader_force = np.array([-mag,0])
                            elif cmd == 2:
                                agent.leader_force = np.array([mag,0])
                            elif cmd == 3:
                                agent.leader_force = np.array([0,-mag])
                            elif cmd == 4:
                                agent.leader_force = np.array([0,mag])
                            elif cmd == 5:
                                vector = (agent.state.p_pos - leader.state.p_pos)
                                agent.leader_force = -vector
                            elif cmd == 6:
                                vector = (agent.state.p_pos - leader.state.p_pos)
                                #Keep Follower inside of the area
                                outbound = False
                                for p in range(2):
                                    if abs(agent.state.p_pos[p]) > 0.9:
                                        outbound = True
                                if not outbound:
                                    agent.leader_force = vector
                                else:
                                    agent.leader_force = np.zeros(2)
                            p_force[i][0] = (np.array(p_force[i][0]) + np.array(agent.leader_force[0]) * mag).tolist()
                            p_force[i][1] = (np.array(p_force[i][1]) + np.array(agent.leader_force[1]) * mag).tolist()
                            np.set_printoptions(precision=1, floatmode='fixed')
                            #print (agent.name, ': leader_force is ', np.round(agent.leader_force,2), 'and p_force is ', np.round(p_force[i],2), ' mag:',mag)
            else: # for pow1, same, not org scenario
                for i,agent in enumerate(self.agents):
                    if agent.adversary and not agent.advleader:
                        vector = (agent.state.p_pos - leader.state.p_pos)
                        vector /= np.linalg.norm(np.array(vector))
                        agent.leader_force[0] = leader.action.c[1] - leader.action.c[2] + vector[0] * (leader.action.c[5] - leader.action.c[6])
                        agent.leader_force[1] = leader.action.c[3] - leader.action.c[4] + vector[1] * (leader.action.c[5] - leader.action.c[6]) 
                        #print(agent.name, 'own action force:',np.round(p_force[i],2),' leader force:',np.round(agent.leader_force * mag))
                        p_force[i][0] = (np.array(p_force[i][0]) + np.array(agent.leader_force[0]) * mag).tolist()
                        p_force[i][1] = (np.array(p_force[i][1]) + np.array(agent.leader_force[1]) * mag).tolist()
                        np.set_printoptions(precision=2, floatmode='fixed')
                        #print('at sum', np.round(p_force[i],2))
        return p_force

    # gather physical forces acting on entities
    def apply_environment_force(self, p_force):
        # simple (but inefficient) collision response
        for a,entity_a in enumerate(self.entities):
            for b,entity_b in enumerate(self.entities):
                if(b <= a): continue
                [f_a, f_b] = self.get_collision_force(entity_a, entity_b)
                if(f_a is not None):
                    if(p_force[a] is None): p_force[a] = 0.0
                    p_force[a] = f_a + p_force[a] 
                if(f_b is not None):
                    if(p_force[b] is None): p_force[b] = 0.0
        return p_force

    # integrate physical state
    def integrate_state(self, p_force):
        for i,entity in enumerate(self.entities):
            if not entity.movable: continue
            entity.state.p_vel = entity.state.p_vel * (1 - self.damping)
            if (p_force[i] is not None):
                entity.state.p_vel += (p_force[i] / entity.mass) * self.dt
            if entity.max_speed is not None:
                speed = np.sqrt(np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1]))
                if speed > entity.max_speed:
                    entity.state.p_vel = entity.state.p_vel / np.sqrt(np.square(entity.state.p_vel[0]) +
                                                                  np.square(entity.state.p_vel[1])) * entity.max_speed
            entity.state.p_pos += entity.state.p_vel * self.dt
   
    def update_agent_state(self, agent):
        # set communication state (directly for now)
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            noise = np.random.randn(*agent.action.c.shape) * agent.c_noise if agent.c_noise else 0.0
            agent.state.c = agent.action.c + noise      

    # get collision forces for any contact between two entities
    def get_collision_force(self, entity_a, entity_b):
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None] # not a collider
        if (entity_a is entity_b):
            return [None, None] # don't collide against itself
        # compute actual distance between entities
        delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        # minimum allowable distance
        dist_min = entity_a.size + entity_b.size
        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min)/k)*k
        force = self.contact_force * delta_pos / dist * penetration
        force_a = +force if entity_a.movable else None
        force_b = -force if entity_b.movable else None
        return [force_a, force_b]