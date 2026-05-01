from util.mdp import MDP
import numpy as np

from util.dynamicsmodels import eom_cr3bp
from util.numericalsolvers import ivp
from util.consts import DU, VU

class DeterminsticCislunarMDP(MDP):
    def __init__(self, mu, u_mag, hbr, dt, discount=0.99, integrator='RK45'):
        super().__init__(discount)
        self.mu = mu
        self.u_mag = u_mag
        self.hbr = hbr
        self.dt = dt
        self.discount = discount
        self.integrator = integrator
    
    @property
    def states(self):
        # Continuous state space
        return []
    
    @property
    def actions(self):
        # can thrust along any axis or do nothing
        u_valid = np.array([0.25, 0.5, 0.75, 1.0])
        axes = np.eye(3)
        sign = [-1, 1]

        a =  [(0, 0, 0)] + [tuple(u * ax * s) for u in u_valid for ax in axes for s in sign]
        return a

    def is_terminal(self, state):
        distance = np.linalg.norm(state[0:3] - state[6:9])
        return self.hbr > distance
    
    def transition(self, state, action):
        state = np.array(state, dtype=float).copy()
        state[3:6] += action
        # enforce velocity constraint

        # propagate both separately
        p = state[0:6]
        s = state[6:12]
        pi = state[12:18]

        pp = ivp(eom_cr3bp, p, [0, self.dt], self.integrator, mu=self.mu)
        sp = ivp(eom_cr3bp, s, [0, self.dt], self.integrator, mu=self.mu)
        pip = ivp(eom_cr3bp, pi, [0, self.dt], self.integrator, mu=self.mu)

        next_state = np.concatenate((pp.y[:, -1], sp.y[:, -1], pip.y[:, -1]))

        return [(1.0, next_state)] # deterministic so prob = 1
    
    def reward(self, state, action, state_p):
        ps = state[0:6]
        ss = state[6:12]
        pis = state[12:18]
        
        psp = state_p[0:6]
        pisp = state_p[12:18]

        def phi(state, k, c):
            p = state[0:6]
            pi = state[12:18]

            r = p[0:3]
            ri = pi[0:3]
            v = p[3:6]
            vi = pi[3:6]

            return -k * ((np.linalg.norm(r - ri) * DU) ** 2 + c * (np.linalg.norm(v - vi) * VU) ** 2)
        
        d = np.linalg.norm(ps[0:3] - ss[0:3]) * DU
        kd = 10
        wd = 10 if d > kd * self.hbr * DU else 0.5

        wr = 0.1 * wd # Distance penalty weight
        wv = 2.0 * wd # Velocity penalty weight
        wc = 40 # Control penalty weight

        k = 0.1 # Shaping distance gain
        c = 0.1 # Shaping velocity gain
        gamma = 0.99 # Shaping discount factor

        k_p = 1 # Stepwise tracking error gain

        i = -1000 # Collision penalty

        r_tracking = -wr * ((np.linalg.norm(ps[0:3] - pis[0:3]) * DU) ** 2) - wv * ((np.linalg.norm(ps[3:6] - pis[3:6]) * VU) ** 2)
        r_control = -wc * (np.linalg.norm(action) * self.u_mag * VU) ** 2
        r_shaping = gamma * phi(state_p, k, c) - phi(state, k, c)
        r_stepwise_tracking = -k_p * ((np.linalg.norm(ps[0:3] - pis[0:3]) * DU) - (np.linalg.norm(psp[0:3] - pisp[0:3]) * DU))
        r_on_track = 0.1 if (np.linalg.norm(psp[0:3] - pisp[0:3]) * DU < 2) and (action == (0, 0, 0)) else 0

        r = r_tracking + r_control + r_shaping + r_stepwise_tracking + r_on_track

        # min_r_dist = 0 # [km]
        # r_dist = (np.linalg.norm(ps[0:3] - pis[0:3]) * DU) ** 2 # [km^2]
        # r_dist_p = (np.linalg.norm(psp[0:3] - pisp[0:3]) * DU) ** 2 # [km^2]
        # r_dist_weight = -10
        # r_dist_penalty = r_dist_weight * r_dist_p if r_dist_p > min_r_dist else 0

        # min_v_dist = 0 # [km / s]
        # v_dist = (np.linalg.norm(ps[3:6] - pis[3:6]) * VU) ** 2 # [km^2 / s^2]
        # v_dist_p = (np.linalg.norm(psp[3:6] - pisp[3:6]) * VU) ** 2 # [km^2 / s^2]
        # v_dist_weight = -1
        # v_dist_penalty = v_dist_weight * v_dist_p if v_dist_p > min_v_dist else 0

        # dist_penalty = r_dist_penalty + v_dist_penalty

        # shaping_weight = 0.1
        # shaping = shaping_weight * (r_dist - r_dist_p + v_dist - v_dist_p)

        # # if dist_penalty:
        # #     print(dist_penalty)

        if self.is_terminal(state_p): #collision, at least thus far
            return i
        else:
            return r
