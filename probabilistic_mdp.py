from util.mdp import MDP
import numpy as np

from util.dynamicsmodels import eom_cr3bp
from util.numericalsolvers import ivp

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

from util.deterministicmdp import DeterminsticCislunarMDP
from util.mdpsolvers import simulate_mcts_policy
from util.consts import mu, DU, TU, VU

class GaussianMDP(MDP):
    def __init__(self, mu, u_mag, hbr, dt, discount=0.99, integrator='RK45'):
        super().__init__(discount)
        self.mu = mu
        self.u_mag = u_mag
        self.hbr = hbr
        self.dt = dt
        self.discount = discount
        self.integrator = integrator
        self.Q = np.eye(12) * 1e-6  # or tune this
    
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
        # print(state[0])
        distance = np.linalg.norm(state[0][0:3] - state[0][6:9])
        return self.hbr > distance

    def cr3bp_derivatives(self, state):
        r = state[0:3]
        v = state[3:6]

        x, y, z = r
        vx, vy, vz = v

        r_mu = np.sqrt((x + self.mu)**2 + y**2 + z**2)
        r_ommu = np.sqrt((x - 1 + self.mu)**2 + y**2 + z**2)

        dVdx = x - (1 - self.mu)/r_mu**3 * (x + self.mu) - self.mu/r_ommu**3 * (x - 1 + self.mu)
        dVdy = y - (1 - self.mu)/r_mu**3 * y - self.mu/r_ommu**3 * y
        dVdz = -(1 - self.mu)/r_mu**3 * z - self.mu/r_ommu**3 * z

        ax = 2*vy + dVdx
        ay = -2*vx + dVdy
        az = dVdz

        return np.concatenate((v, [ax, ay, az]))

    def numerical_jacobian(self, f, x, eps=1e-5):
        n = len(x)
        J = np.zeros((n, n))
        fx = f(x)

        for i in range(n):
            dx = np.zeros(n)
            dx[i] = eps
            J[:, i] = (f(x + dx) - fx) / eps

        return J

    def transition(self, state, action):
        mean, cov = state
        mean = mean.copy()

        mean[3:6] += action

        # if np.any(np.abs(mean[3:6]) > self.max_vel):
        #     mean[3:6] -= action

        def dynamics(state): # still not doing scipy rk4
            p = state[0:6]
            s = state[6:12]

            pp = ivp(eom_cr3bp, p, [0, self.dt], self.integrator, mu=self.mu)
            sp = ivp(eom_cr3bp, s, [0, self.dt], self.integrator, mu=self.mu)

            return np.concatenate((pp.y[:, -1], sp.y[:, -1]))
            
        mean_next = dynamics(mean)

        # linearize
        F = self.numerical_jacobian(dynamics, mean)

        cov_next = F @ cov @ F.T + self.Q

        return [(1.0, (mean_next, cov_next))]

    def reward(self, state, action, state_p):
        if self.is_terminal(state): #collision, at least thus far
            return -1000
        elif action != (0,0,0):
            return -1 * self.u_mag
        return 0
    

def simulate_gaussian_mcts(mdp, init_mean, init_cov, steps=10):
    state = (init_mean, init_cov)

    trajectory = [init_mean.copy()]
    covariances = [init_cov.copy()]

    for i in range(steps):
        print(i)
        if mdp.is_terminal(state):
            break

        action, _ = mdp.MCTS(state, n_simulations=200)

        (_, (next_mean, next_cov)) = mdp.transition(state, action)[0]

        state = (next_mean, next_cov)

        trajectory.append(next_mean.copy())
        covariances.append(next_cov.copy())

    return np.array(trajectory), covariances

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_trajectory_with_uncertainty(traj, covs, sigma_scale=2.0):
    # sigma_scale: 1 = 68%, 2 = 95%, 3 = 99.7%
    p = traj[:, 0:3]

    stds = np.array([
        np.sqrt(np.diag(cov[0:3, 0:3])) for cov in covs
    ]) 

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.plot(p[:,0], p[:,1], p[:,2], color='blue', label='Mean Trajectory')

    for i in range(3):
        upper = p[:, i] + sigma_scale * stds[:, i]
        lower = p[:, i] - sigma_scale * stds[:, i]

        if i == 0:
            ax.plot(upper, p[:,1], p[:,2], color='cyan', alpha=0.3)
            ax.plot(lower, p[:,1], p[:,2], color='cyan', alpha=0.3)
        elif i == 1:
            ax.plot(p[:,0], upper, p[:,2], color='green', alpha=0.3)
            ax.plot(p[:,0], lower, p[:,2], color='green', alpha=0.3)
        else:
            ax.plot(p[:,0], p[:,1], upper, color='purple', alpha=0.3)
            ax.plot(p[:,0], p[:,1], lower, color='purple', alpha=0.3)

    ax.scatter(p[0,0], p[0,1], p[0,2], c='green', label='Start')

    ax.legend()
    plt.title(f"Trajectory with uncertainty")
    plt.show()

def plot_trajectory(traj):
    p = traj[:, 0:3] # primary
    s = traj[:, 6:9] # secondary

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.plot(p[:,0], p[:,1], p[:,2], label="Primary")
    ax.plot(s[:,0], s[:,1], s[:,2], label="Secondary")

    ax.scatter(p[0,0], p[0,1], p[0,2], c='green', label="Start")
    ax.scatter(s[0,0], s[0,1], s[0,2], c='green')

    ax.legend()
    plt.show()


u_mag = 0.1 / VU # Maximum control magnitude (0.1 km / s)
hbr = 1 / DU # Hard-body radius (0.1 km)
dt = 0.5 / TU # Time step (1 min)

L4_x = 0.5 - mu
L4_y = np.sqrt(3) / 2
L4_z = 0

L4_x_offset = 0.005
L4_y_offset = -0.005
L4_z_offset = 0
dx = 5 / DU
dy = -5 / DU
dz = 0

dr = np.array([dx, dy, dz])
dr_hat = dr / np.linalg.norm(dr)

pr_0 = np.array([L4_x + L4_x_offset, L4_y + L4_y_offset, L4_z + L4_z_offset])
pv_0 = 1 / VU * dr_hat
sr_0 = pr_0 + dr
sv_0 = -pv_0

m = GaussianMDP(mu, u_mag, hbr, dt)

init_state = np.concat((pr_0, pv_0, sr_0, sv_0))
init_cov = np.eye(12) * 10.0 

traj, covs = simulate_gaussian_mcts(m, init_state, init_cov)
print(traj, covs)
plot_trajectory_with_uncertainty(traj, covs)

plot_trajectory(traj) # not working :(
