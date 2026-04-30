from util.mdp import MDP
import numpy as np

from util.dynamicsmodels import eom_cr3bp
from util.numericalsolvers import ivp

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
        a = [(0, 0, self.u_mag), (0, 0, -self.u_mag), (0, self.u_mag, 0), (0, -self.u_mag, 0), (self.u_mag, 0, 0), (-self.u_mag, 0, 0), (0, 0, 0)]
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

        pp = ivp(eom_cr3bp, p, [0, self.dt], self.integrator, mu=self.mu)
        sp = ivp(eom_cr3bp, s, [0, self.dt], self.integrator, mu=self.mu)

        next_state = np.concatenate((pp.y[:, -1], sp.y[:, -1]))

        return [(1.0, next_state)] # deterministic so prob = 1
    
    def reward(self, state, action, state_p):
        if self.is_terminal(state): #collision, at least thus far
            return -1000
        elif action != (0,0,0):
            return -1 * self.u_mag
        return 0
