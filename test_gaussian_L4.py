import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag

from util.gaussianmdp import GaussianCislunarMDP
from util.consts import mu, DU, TU, VU

def simulate_gaussian_mcts(mdp, init_mean, init_cov, steps=20, n_simulations=200):
    state = (init_mean.copy(), init_cov.copy())
    traj = [init_mean.copy()]
    covs = [init_cov.copy()]
    rewards = []
    took_action = []

    for step in range(steps):
        print(step+1)
        if mdp.is_terminal(state):
            print("kersplode")
            break

        action, _ = mdp.MCTS(state, n_simulations=n_simulations)
        _, (next_mean, next_cov) = mdp.transition(state, action)[0]

        r   = mdp.reward(state, action, (next_mean, next_cov))
        thrust = int(np.linalg.norm(action) > 0)

        state = (next_mean, next_cov)
        traj.append(next_mean.copy())
        covs.append(next_cov.copy())
        rewards.append(r)
        took_action.append(thrust)

    return np.array(traj), covs, rewards, took_action


def simulate_gaussian_coast(mdp, init_mean, init_cov, steps=20):
    state = (init_mean.copy(), init_cov.copy())
    traj = [init_mean.copy()]
    covs = [init_cov.copy()]
    zero = np.zeros(3)

    for _ in range(steps):
        if mdp.is_terminal(state):
            break
        _, (nm, nc) = mdp.transition(state, zero)[0]
        state = (nm, nc)
        traj.append(nm.copy())
        covs.append(nc.copy())

    return np.array(traj), covs


def plot_trajectory_with_uncertainty(traj, covs, sigma_scale=2.0, title="Mean trajectory +/- uncertainty"):
    p    = traj[:, 0:3]
    stds = np.array([np.sqrt(np.diag(c)[0:3]) for c in covs])

    fig = plt.figure(figsize=(10, 7))
    ax  = fig.add_subplot(projection="3d")

    ax.plot(*p.T, color="blue", linewidth=2, label="Mean trajectory (primary)")

    colors = ["cyan", "limegreen", "orchid"]
    labels = ["±σ x", "±σ y", "±σ z"]
    for i, (col, lbl) in enumerate(zip(colors, labels)):
        upper = p.copy(); lower = p.copy()
        upper[:, i] += sigma_scale * stds[:, i]
        lower[:, i] -= sigma_scale * stds[:, i]
        ax.plot(*upper.T, color=col, alpha=0.5, linewidth=1.5, label=f"{lbl} upper")
        ax.plot(*lower.T, color=col, alpha=0.5, linewidth=1.5, label=f"{lbl} lower")

    s = traj[:, 6:9]
    ax.plot(*s.T, color="red", linewidth=1.5, label="Secondary (mean)")

    ax.scatter(*p[0],  c="green",      s=60, label="Start")
    ax.scatter(*p[-1], c="tab:purple", s=60, label="End")

    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title(f"{title}")
    plt.tight_layout()
    plt.show()


def plot_distance(traj, hbr, covs=None, sigma_scale=2.0, title="Mean separation over time"):
    p = traj[:, 0:3]
    s = traj[:, 6:9]
    d = np.linalg.norm(p - s, axis=1)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axhline(hbr, color="red", linestyle="--", label=f"HBR")

    if covs is not None:
        stds_rel = np.array([
            np.sqrt(np.trace(c[0:3, 0:3]) + np.trace(c[6:9, 6:9])) / np.sqrt(3)
            for c in covs
        ])
        ax.fill_between(
            range(len(d)),
            d - sigma_scale * stds_rel,
            d + sigma_scale * stds_rel,
            alpha=0.2, color="blue", label=f"±{sigma_scale}σ band"
        )
        

    ax.set_xlabel("Time step")
    ax.set_ylabel("Distance")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_comparison(traj_mcts, traj_coast, hbr, title="MCTS vs No Action (mean trajectories)"):
    p_mcts  = traj_mcts[:, 0:3]
    p_coast = traj_coast[:, 0:3]
    s       = traj_mcts[:, 6:9]

    fig = plt.figure(figsize=(10, 7))
    ax  = fig.add_subplot(projection="3d")

    ax.plot(*p_mcts.T,  color="blue",  linewidth=2,   label="Primary - MCTS")
    ax.plot(*p_coast.T, color="cyan",  linewidth=1.5, linestyle="dashed", label="Primary - coast")
    ax.plot(*s.T,       color="red",   linewidth=1.5, label="Secondary")

    ax.scatter(*p_mcts[0],  c="green",      s=60)
    ax.scatter(*p_mcts[-1], c="tab:purple", s=60)

    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.legend(fontsize=8)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_rewards(rewards, took_action, title="Per-step reward and thrust"):
    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    plt.plot(rewards, color="steelblue", marker="o", markersize=3)
    # plt.set_ylabel("Reward"); ax1.set_title(title)
    # ax2.bar(range(len(took_action)), took_action)
    # ax2.set_ylabel("Thrust (1=yes)"); ax2.set_xlabel("Time step")
    # plt.tight_layout()
    # plt.show()


u_mag = 0.1 / VU
hbr   = 1.0 / DU
dt    = 0.5 / TU

L4_x = 0.5 - mu
L4_y = np.sqrt(3) / 2
L4_z = 0.0

pr_0 = np.array([L4_x + 0.005, L4_y - 0.005, L4_z])

dx, dy, dz = 5.0 / DU, -5.0 / DU, 0.0
dr     = np.array([dx, dy, dz])
dr_hat = dr / np.linalg.norm(dr)

pv_0 = (1.0 / VU) * dr_hat
sr_0 = pr_0 + dr
sv_0 = -pv_0

init_mean = np.concatenate([pr_0, pv_0, sr_0, sv_0])

init_cov = block_diag(
    np.eye(3) * (0.5 / DU) ** 2, # primary position  (0.5 km std)
    np.eye(3) * (0.01 / VU) ** 2, # primary velocity  (0.01 km/s std)
    np.eye(3) * (0.5 / DU) ** 2, # secondary position
    np.eye(3) * (0.01 / VU) ** 2, # secondary velocity
)

mdp = GaussianCislunarMDP(
    mu=mu,
    u_mag=u_mag,
    hbr=hbr,
    dt=dt,
    discount=0.99,
    process_noise_pos=(1e-4 / DU) ** 2,
    process_noise_vel=(1e-5 / VU) ** 2,
    shape_radius=5 * hbr,
    shape_weight=50.0,
    fuel_weight=1.0,
    safe_bonus=0.1,
    uncertainty_weight=20.0,
)


SIM_STEPS = 10

print("Running Gaussian MCTS …")
traj, covs, rewards, took_action = simulate_gaussian_mcts(mdp, init_mean, init_cov, steps=SIM_STEPS, n_simulations=10)

print("Running coast/no action trajectory …")
traj_coast, covs_coast = simulate_gaussian_coast(mdp, init_mean, init_cov, steps=SIM_STEPS)


print("rewards: ", np.round(rewards, 4))
print("took action", took_action)

final_dist   = np.linalg.norm(traj[-1, 0:3] - traj[-1, 6:9])
initial_dist = np.linalg.norm(traj[0,  0:3] - traj[0,  6:9])
print(f"\nInitial mean separation : {initial_dist * DU:.2f} km")
print(f"Final mean separation : {final_dist * DU:.2f} km")
print(f"HBR: {hbr * DU:.2f} km")


plot_trajectory_with_uncertainty(traj, covs, sigma_scale=2.0)
plot_distance(traj, hbr, covs=covs, sigma_scale=2.0)

plot_comparison(traj, traj_coast, hbr)
plot_rewards(rewards, took_action)