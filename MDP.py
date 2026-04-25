import numpy as np
from abc import ABC, abstractmethod
import math
from collections import defaultdict


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
    
    def MCTS(self, root_state, n_simulations=1000, c=1.4, max_depth=100):
        N = defaultdict(int)
        Nsa = defaultdict(int)
        Q = defaultdict(float)

        def sample_transition(state, action):
            transitions = self.transition(state, action)
            probs, next_states = zip(*transitions)
            idx = np.random.choice(len(next_states), p=probs)
            return next_states[idx]

        def select_action(state):
            state_key = tuple(state)

            best_score = -np.inf
            best_action = None

            for a in self.actions:
                if Nsa[(state_key, a)] == 0:
                    return a

                uct = Q[(state_key, a)] + c * math.sqrt(
                    math.log(N[state_key]) / Nsa[(state_key, a)]
                )

                if uct > best_score:
                    best_score = uct
                    best_action = a

            return best_action

        for _ in range(n_simulations):
            state = np.array(root_state, dtype=float).copy()
            path = []
            depth = 0

            while True:
                state_key = tuple(state)

                if self.is_terminal(state) or depth >= max_depth:
                    break

                a = select_action(state)
                next_state = sample_transition(state, a)
                r = self.reward(state, a, next_state)

                path.append((state_key, a, r))

                if Nsa[(state_key, a)] == 0:
                    state = next_state
                    break

                state = next_state
                depth += 1

            G = 0.0

            for s_key, a, r in reversed(path):
                G = r + self.discount * G

                N[s_key] += 1
                Nsa[(s_key, a)] += 1
                Q[(s_key, a)] += (G - Q[(s_key, a)]) / Nsa[(s_key, a)]

        root_key = tuple(root_state)

        best_action = None
        best_value = -np.inf

        for a in self.actions:
            if Q[(root_key, a)] > best_value:
                best_value = Q[(root_key, a)]
                best_action = a

        return best_action, Q

    # def q_value(self, V, state, action):
    #     gamma = self.discount
    #     return sum(p * (self.reward(state, action, sp) + gamma * V[s_next]) for p, sp in self.transition(state, action))