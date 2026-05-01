import numpy as np

def simulate_mcts_policy(mdp, init_state, steps=200, n_simulations=50, max_depth=10):
    state = np.array(init_state, dtype=float)
    trajectory = [state.copy()]
    rewards = []
    took_action = []

    for i in range(steps):
        print(i)
        if mdp.is_terminal(state):
            break

        action, _ = mdp.MCTS(state, n_simulations=n_simulations, max_depth=max_depth)
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
