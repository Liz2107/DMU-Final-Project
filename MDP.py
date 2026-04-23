import numpy as np
from abc import ABC, abstractmethod
from typing import Any


class MDP(ABC):
    def __init__(self, discount = 0.99):
        if not (0.0 < discount <= 1.0):
            raise ValueError("Discount factor must be in (0, 1].")
        self.discount = discount

    @property
    @abstractmethod
    def states(self):
        """Return S"""

    @property
    @abstractmethod
    def actions(self):
        """Return A"""

    @abstractmethod
    def transition(self, state, action):
        """Return a list of (probability, next_state) tuples, probs must sum to 1"""

    @abstractmethod
    def reward(self, state, action, state_p):
        """R(s, a, s')"""

    @abstractmethod
    def is_terminal(self, state):
        """Return True if terminal"""

    def value_iteration(self, epsilon = 1e-6, max_iter = 10_000, verbose = False):
        states = self.states
        actions = self.actions
        gamma = self.discount

        # initial v is all zeros
        V = {str(s): 0.0 for s in states}
        print(V)

        for iteration in range(max_iter):
            print(f"iteration #{iteration}")
            delta = 0.0
            Vp = {}

            for s in states:
                if self.is_terminal(s):
                    Vp[s] = 0.0
                    continue

                best = -np.inf
                for a in actions:
                    q = sum(p * (self.reward(s, a, sp) + gamma * V[sp]) for p, sp in self.transition(s, a))
                    if q > best:
                        best = q

                Vp[s] = best
                delta = abs(Vp[s] - V[s])

            V = Vp

            if verbose and iteration % 100 == 0: # optional progress tracking, might make the 100 a param later if we run big sims
                print(f"Value iteration {iteration}  with difference delta={delta}")

            if delta < epsilon:
                if verbose:
                    print(f"Value iteration converged on iteration {iteration} (delta={delta})")
                break
        else:
            print(f"Value iteration did not converge!")

        # policy
        policy = self._extract_policy(V)
        return V, policy

    def policy_iteration(self, epsilon = 1e-6, eval_max_iter = 5_000, max_policy_iter = 500, verbose = False):
        states = self.states
        actions = self.actions
        gamma = self.discount

        # same random policy
        policy = {s: actions[0] for s in states}
        V = {s: 0.0 for s in states}

        for pi_iter in range(max_policy_iter):
            for _ in range(eval_max_iter):
                delta = 0.0
                Vp = {}
                for s in states:
                    if self.is_terminal(s):
                        Vp[s] = 0.0
                        continue
                    a = policy[s]
                    v = sum(p * (self.reward(s, a, sp) + gamma * V[sp]) for p, sp in self.transition(s, a))
                    Vp[s] = v
                    delta = max(delta, abs(v - V[s]))
                V = Vp
                if delta < epsilon:
                    break

            # improvement
            policy_stable = True
            for s in states:
                if self.is_terminal(s):
                    continue
                old_action = policy[s]
                best_a, best_q = old_action, -np.inf
                for a in actions:
                    q = sum(p * (self.reward(s, a, s_next) + gamma * V[s_next]) for p, s_next in self.transition(s, a))
                    if q > best_q:
                        best_q = q
                        best_a = a
                policy[s] = best_a
                if best_a != old_action:
                    policy_stable = False

            if verbose:
                print(f"Policy iteration {pi_iter}")

            if policy_stable:
                if verbose:
                    print(f"Converged at iteration {pi_iter}")
                break
        else:
            print(f"Policy did not converge!")

        return V, policy


    def _extract_policy(self, V):
        gamma = self.discount
        policy = {}
        for s in self.states:
            if self.is_terminal(s):
                policy[s] = None
                continue
            best_a, best_q = None, -np.inf
            for a in self.actions:
                q = sum(
                    p * (self.reward(s, a, s_next) + gamma * V[s_next])
                    for p, s_next in self.transition(s, a)
                )
                if q > best_q:
                    best_q = q
                    best_a = a
            policy[s] = best_a
        return policy

    # def q_value(self, V, state, action):
    #     gamma = self.discount
    #     return sum(p * (self.reward(state, action, sp) + gamma * V[s_next]) for p, sp in self.transition(state, action))