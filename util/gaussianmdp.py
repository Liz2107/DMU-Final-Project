from util.mdp import MDP
import numpy as np
from scipy.stats import chi2

from util.dynamicsmodels import eom_cr3bp
from util.numericalsolvers import ivp


class GaussianCislunarMDP(MDP):
    def __init__(
        self,
        mu,
        u_mag,
        hbr,
        dt,
        discount=0.99,
        integrator="RK45",
        process_noise_pos=1e-8, # position process-noise variance (per axis)
        process_noise_vel=1e-8, # velocity process-noise variance (per axis)
        shape_radius=None,
        shape_weight=50.0,
        fuel_weight=1.0,
        uncertainty_weight=20.0, # penalty scaling for collision-probability estimate
    ):
        super().__init__(discount)
        self.mu = mu
        self.u_mag = u_mag
        self.hbr = hbr
        self.dt = dt
        self.integrator = integrator
        self.shape_radius = shape_radius if shape_radius is not None else 10 * hbr
        self.shape_weight = shape_weight
        self.fuel_weight = fuel_weight
        self.uncertainty_weight = uncertainty_weight

        # noise matrix 
        q_diag = np.array(
            [process_noise_pos] * 3 # primary pos
            + [process_noise_vel] * 3 # primary vel
            + [process_noise_pos] * 3 # secondary pos
            + [process_noise_vel] * 3 # secondary vel
        )
        self.Q = np.diag(q_diag)

    @property
    def states(self):
        return [] 

    @property
    def actions(self):
        u = self.u_mag
        return [
            ( u,  0,  0),
            (-u,  0,  0),
            ( 0,  u,  0),
            ( 0, -u,  0),
            ( 0,  0,  u),
            ( 0,  0, -u),
            ( 0,  0,  0),
        ]

    def is_terminal(self, state): # terminal based on mean, so MCTS can continue
        mean, _ = state
        dist = np.linalg.norm(mean[0:3] - mean[6:9])
        return dist < self.hbr

    def _cr3bp_step(self, state_12):
        p_next = ivp(eom_cr3bp, state_12[0:6],  [0, self.dt], self.integrator, mu=self.mu)
        s_next = ivp(eom_cr3bp, state_12[6:12], [0, self.dt], self.integrator, mu=self.mu)
        return np.concatenate((p_next.y[:, -1], s_next.y[:, -1]))

    def _numerical_jacobian(self, f, x, eps=1e-6):
        n = len(x)
        # fx = f(x)
        J = np.zeros((n, n))
        for i in range(n):
            dx = np.zeros(n)
            dx[i] = eps
            J[:, i] = (f(x + dx) - f(x - dx)) / (2 * eps)
        return J

    def transition(self, state, action):
        mean, cov = state
        mean = np.asarray(mean, dtype=float).copy()
        cov  = np.asarray(cov,  dtype=float).copy()
        action = np.asarray(action, dtype=float)

        mean[3:6] += action

        # propagate
        mean_next = self._cr3bp_step(mean)

        # linearise around current mean 
        F = self._numerical_jacobian(self._cr3bp_step, mean)

        # propagate covariance
        cov_next = F @ cov @ F.T + self.Q

        # make symmetrical
        cov_next = 0.5 * (cov_next + cov_next.T)

        return [(1.0, (mean_next, cov_next))]


    def _collision_probability_approx(self, mean, cov): # 
        mu_rel  = mean[0:3] - mean[6:9] # relative distance
        cov_rel = cov[0:3, 0:3] + cov[6:9, 6:9] # uncertainty adds

        try: #somehow I got this error ;-;
            cov_inv = np.linalg.inv(cov_rel) 
        except np.linalg.LinAlgError:
            return 0.0 # idk what to actually do please just work

        # Mahalanobis distance
        dist_mean = np.linalg.norm(mu_rel)
        if dist_mean < 1e-12:
            # print("returning")
            return 1.0 # perhaps this is always returning?

        direction = mu_rel / dist_mean
        maha_sq = float(direction @ cov_inv @ direction) * (dist_mean - self.hbr) ** 2

        # P(inside hbr) is approx 1 - chi2.cdf(maha_sq, df=3)
        return float(1.0 - chi2.cdf(max(maha_sq, 0.0), df=3))

    def reward(self, state, action, state_p):
        mean_p, cov_p = state_p
        action = np.asarray(action, dtype=float)

        # terminal collision
        if self.is_terminal(state_p):
            return -100000000.0

        dist_p = np.linalg.norm(mean_p[0:3] - mean_p[6:9])

        if dist_p < self.shape_radius:
            shape = -self.shape_weight / max(dist_p, 1e-9)
        else:
            shape = 0.0

        p_col = self._collision_probability_approx(mean_p, cov_p)
        uncertainty_penalty = -self.uncertainty_weight * p_col

        dv = np.linalg.norm(action)
        fuel = -self.fuel_weight * dv if dv > 0 else 0.0

        return uncertainty_penalty + fuel + shape