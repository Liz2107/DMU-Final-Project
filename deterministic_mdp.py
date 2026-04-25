from MDP import MDP
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

class DeterministicMDP(MDP):
    def __init__(self, discount=0.99):
        super().__init__(discount)
        self.mu = 0.00121505
        self.dt = .1
        self.thrust_mag = .5
        self.max_vel = 0.5
    
    @property
    def states(self):
        #need to update to be some grid that has the positions and velocities of each
        # perhaps itertools can be useful here, thinking format of [x, y, z, dx, dy, dz, xs, ys, zs, dxs, dys, dzs]
        xs, ys, zs, dxs, dys, dzs = np.linspace(0, 10, 10), np.linspace(0, 10, 10), np.linspace(0, 10, 10), np.linspace(-.5, .5, 10), np.linspace(-.5, .5, 10), np.linspace(-.5, .5, 10)
        num_states = (len(xs) * len(ys) * len(zs) * len(dxs) * len(dys) * len(dzs))**2
        # s = [[] for _ in range(num_states)]
        return []
    
    @property
    def actions(self):
        # can thrust along any axis or do nothing
        a = [(0,0,self.thrust_mag), (0,0,-self.thrust_mag), (0,self.thrust_mag,0), (0, -self.thrust_mag, 0), (self.thrust_mag, 0, 0), (-self.thrust_mag, 0, 0), (0, 0, 0)]
        return a

    def is_terminal(self, state):
        collision_radius = 1
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

    def rk4_step(self, f, t, y, dt, mu): # TO-DO: update this to use the cr3bp_derivatives 
        # just taken from google AI
        k1 = f(t, y, mu)
        k2 = f(t + dt/2, y + dt/2 * k1)
        k3 = f(t + dt/2, y + dt/2 * k2)
        k4 = f(t + dt,   y + dt * k3)

        return y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    def transition(self, state, action):
        state = np.array(state, dtype=float).copy()
        state[3:6] += action
        # enforce velocity constraint
        if np.any(np.abs(state[3:6]) > self.max_vel):
            state[3:6] -= action

        # propagate both separately
        p = state[0:6]
        s = state[6:12]

        # need to switch to rk4
        p_new = p + self.dt * self.cr3bp_derivatives(p)
        s_new = s + self.dt * self.cr3bp_derivatives(s)

        next_state = np.concatenate((p_new, s_new))

        return [(1.0, next_state)] # deterministic so prob = 1
    
    def reward(self, state, action, state_p):
        if self.is_terminal(state): #collision, at least thus far
            return -100
        elif action != (0,0,0):
            return -1 * self.thrust_mag
        return 0
import matplotlib.pyplot as plt

def simulate_mcts_policy(mdp, init_state, steps=200):
    state = np.array(init_state, dtype=float)
    trajectory = [state.copy()]

    for _ in range(steps):
        if mdp.is_terminal(state):
            break

        action, _ = mdp.MCTS(state, n_simulations=200)
        transitions = mdp.transition(state, action)
        _, next_state = transitions[0] # deterministic

        state = next_state
        trajectory.append(state.copy())

    return np.array(trajectory)

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


import matplotlib.animation as animation

def animate_trajectory(traj): # kinda lame kinda fun
    p = traj[:, 0:3]
    s = traj[:, 6:9]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    line1, = ax.plot([], [], [], 'b-')
    line2, = ax.plot([], [], [], 'r-')

    def update(i):
        line1.set_data(p[:i,0], p[:i,1])
        line1.set_3d_properties(p[:i,2])

        line2.set_data(s[:i,0], s[:i,1])
        line2.set_3d_properties(s[:i,2])

        return line1, line2

    ani = animation.FuncAnimation(fig, update, frames=len(traj), interval=50)
    plt.show()

def plot_distance(traj):
    p = traj[:, 0:3]
    s = traj[:, 6:9]

    d = np.linalg.norm(p - s, axis=1)

    plt.plot(d)
    plt.title("Distance between objects")
    plt.xlabel("time step")
    plt.ylabel("distance")
    plt.show()


def simulate_no_action(mdp, init_state, steps=200):
    state = np.array(init_state, dtype=float)
    trajectory = [state.copy()]

    zero_action = (0.0, 0.0, 0.0)

    for _ in range(steps):
        if mdp.is_terminal(state):
            break

        transitions = mdp.transition(state, zero_action)
        _, next_state = transitions[0]

        state = next_state
        trajectory.append(state.copy())

    return np.array(trajectory)


def plot_comparison(traj_mcts, traj_no_action):
    p_mcts = traj_mcts[:, 0:3]
    p_free = traj_no_action[:, 0:3]

    s_mcts = traj_mcts[:, 6:9] # secondary (same in both if no thrust applied to it)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # MCTS
    ax.plot(p_mcts[:,0], p_mcts[:,1], p_mcts[:,2], label="Primary (MCTS)", color='blue')
    # no action
    ax.plot(p_free[:,0], p_free[:,1], p_free[:,2], linestyle='dashed', label="Primary (No Action)", color='cyan')
    # secondary
    ax.plot(s_mcts[:,0], s_mcts[:,1], s_mcts[:,2], label="Secondary", color='red')
    ax.scatter(p_mcts[0,0], p_mcts[0,1], p_mcts[0,2], c='green', label="Start")

    ax.legend()
    plt.show()


def simulate_mcts_policy_with_rewards(mdp, init_state, steps=200):
    state = np.array(init_state, dtype=float)
    trajectory = [state.copy()]
    rewards = []
    took_action = []

    for _ in range(steps):
        if mdp.is_terminal(state):
            break

        action, _ = mdp.MCTS(state, n_simulations=200)
        if action == (0,0,0):
            took_action.append(0)
        else:
            took_action.append(1)
        transitions = mdp.transition(state, action)
        _, next_state = transitions[0]

        r = mdp.reward(state, action, next_state)

        rewards.append(r)
        state = next_state
        trajectory.append(state.copy())

    return np.array(trajectory), np.array(rewards), np.array(took_action)


def plot_colored_trajectory(traj, rewards):
    p = traj[:, 0:3]

    norm = Normalize(vmin=np.min(rewards), vmax=np.max(rewards))
    cmap = cm.viridis

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for i in range(len(rewards)):
        color = cmap(norm(rewards[i]))
        ax.plot(p[i:i+2, 0], p[i:i+2, 1], p[i:i+2, 2], color=color)

    ax.scatter(p[0,0], p[0,1], p[0,2], c='green', label='Start')

    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(rewards)
    fig.colorbar(mappable, ax=ax, label="Reward/Action Taken")

    ax.legend()
    plt.show()

m = DeterministicMDP()
init_state = [100000,100000,100000,
              0.1,0.1,0.1,
              100000,100000,99997,
              0.1,0.1,1.0] #objects near opposite corners, moving kinda towards each other

# print(m.MCTS(init_state)) # should spit out a mess bc dictionary indexing is bad

traj = simulate_mcts_policy(m, init_state, steps=100)
# plot_trajectory(traj)
# plot_distance(traj)
# animate_trajectory(traj)

traj_free = simulate_no_action(m, init_state, steps=100)
plot_comparison(traj, traj_free)
traj, rewards, took_action = simulate_mcts_policy_with_rewards(m, init_state, steps=100)

plot_colored_trajectory(traj, rewards)
plot_colored_trajectory(traj, took_action)

# useful to see for small examples with < 100 steps
print(took_action)
print(rewards)