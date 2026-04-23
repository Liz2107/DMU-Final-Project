from MDP import MDP
import numpy as np
class DeterministicMDP(MDP):
    def __init__(self, discount=0.99):
        super().__init__(discount)
        self.mu = 0.5 #this should be figured out and perhaps adjusted so it can be set by the problem later. I'm just lazy
        self.dt = .1
        self.thrust_mag = .5
        self.max_vel = 0.5
    
    @property
    def states(self):
        #need to update to be some grid that has the positions and velocities of each
        # perhaps itertools can be useful here, thinking format of [x, y, z, dx, dy, dz, xs, ys, zs, dxs, dys, dzs]
        xs, ys, zs, dxs, dys, dzs = np.linspace(0, 10, 10), np.linspace(0, 10, 10), np.linspace(0, 10, 10), np.linspace(-.5, .5, 10), np.linspace(-.5, .5, 10), np.linspace(-.5, .5, 10)
        num_states = (len(xs) * len(ys) * len(zs) * len(dxs) * len(dys) * len(dzs))**2
        s = [[] for _ in range(num_states)]
        idx = 0
        for x in xs:
            for y in ys: 
                for z in zs:
                    for dx in dxs: 
                        for dy in dys:
                            for dz in dzs:
                                for x2 in xs:
                                    for y2 in ys: 
                                        for z2 in zs:
                                            for dx2 in dxs: 
                                                for dy2 in dys:
                                                    for dz2 in dzs:
                                                        s[idx] = [x,y,z,dx,dy,dz,dz, x2, y2, z2, dx2, dy2, dz2]
        return s
    
    @property
    def actions(self):
        # can thrust along any axis or do nothing
        a = [(0,0,self.thrust_mag), (0,0,-self.thrust_mag), (0,self.thrust_mag,0), (0, -self.thrust_mag, 0), (self.thrust_mag, 0, 0), (-self.thrust_mag, 0, 0), (0, 0, 0)]
        return a

    def is_terminal(self, state):
        collision_radius = 1e-3
        primary_x = state[0]
        primary_y = state[1]
        primary_z = state[2]
        secondary_x = state[6]
        secondary_y = state[7]
        secondary_z = state[8]
        distance = np.sqrt((primary_x - secondary_x)**2 + (primary_y - secondary_y)**2 + (primary_z - secondary_z)**2)
        return collision_radius > distance
    
    def cr3bp_derivatives(self, state): 
        r = state[0:3]
        v = state[3:6]
            
        x = r[0]
        y = r[1]
        z = r[2]

        vx = v[0]
        vy = v[1]

        r_mu = np.sqrt((x + self.mu) ** 2 + y ** 2 + z ** 2)
        r_ommu = np.sqrt((x - 1 + self.mu) ** 2 + y ** 2 + z ** 2)
        
        dVdx = x - (1 - self.mu) / r_mu ** 3 * (x + self.mu) - self.mu / r_ommu ** 3 * (x - 1 + self.mu)
        dVdy = y - (1 - self.mu) / r_mu ** 3 * y - self.mu / r_ommu ** 3 * y
        dVdz = -(1 - self.mu) / r_mu ** 3 * z - self.mu / r_ommu ** 3 * z

        ax = 2 * vy + dVdx
        ay = -2 * vx + dVdy
        az = dVdz

        drdt = v
        dvdt = np.array([ax, ay, az])

        return np.concat((drdt, dvdt))
    
    def transition(self, state, action):
        # update state with action
        state[3:6] += action
        if abs(state[3]) > self.max_vel or abs(state[4]) > self.max_vel or abs(state[5]) > self.max_vel: # clip so impose max velocity to remain in state space
            state -= action

        pstate_new = state[0:6] + self.dt * self.cr3bp_derivatives(state[0:6])
        sstate_new = state[6:12] + self.dt * self.cr3bp_derivatives(state[6:12])
        return np.concat((pstate_new, sstate_new))
    
    def reward(self, state, action, state_p):
        if self.is_terminal(state): #collision, at least thus far
            return -100
        elif action != (0,0,0):
            return -1 * self.thrust_mag
        return 0
        
m = DeterministicMDP()
print(m.value_iteration())
