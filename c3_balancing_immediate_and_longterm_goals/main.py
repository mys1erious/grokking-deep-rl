import numpy as np


def policy_evaluation(pi, P, gamma=1.0, theta=1e-10):
    # theta - checks for convergence
    # gamma - discount factor
    prev_V = np.zeros(len(P))

    while True:
        all_states_len = len(P)
        V = np.zeros(all_states_len)
        for s in range(all_states_len):
            for prob, next_state, reward, done in P[s][pi(s)]:
                V[s] += prob * (reward + gamma * prev_V[next_state] * (not done))
        if np.max(np.abs(prev_V - V)) < theta:
            break
        prev_V = V.copy()
    return V


def policy_improvement(V, P, gamma=1.0):
    states_len = len(P)

    Q = np.zeros((states_len, len(P[0])), dtype=np.float64)

    for s in range(states_len):
        actions_len = len(P[s])
        for a in range(actions_len):
            for prob, next_state, reward, done in P[s][a]:
                # calculate Q func
                Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))

    # new greedy policy
    new_policy = lambda s: {s: a for s, a in enumerate(np.argmax(Q, axis=1))}[s]  # noqa: E731
    return new_policy


def policy_iteration(P, gamma=1.0, theta=1e-10):
    random_actions = np.random.choice(tuple(P[0].keys()), len(P))
    # first step: create a randomly generated policy
    pi = lambda s: {s: a for s, a in enumerate(random_actions)}[s]  # noqa: E731

    while True:
        old_pi = {s: pi(s) for s in range(len(P))}
        V = policy_evaluation(pi, P, gamma, theta)
        pi = policy_improvement(V, P, gamma)
        if old_pi == {s: pi(s) for s in range(len(P))}:
            break

    return V, pi


def value_iteration(P, gamma=1.0, theta=1e-10):
    V = np.zeros(len(P), dtype=np.float64)

    while True:
        Q = np.zeros((len(P), len(P[0])), dtype=np.float64)

        for s in range(len(P)):
            for a in range(len(P[s])):
                for prob, next_state, reward, done in P[s][a]:
                    Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))

        if np.max(np.abs(V - np.max(Q, axis=1))) < theta:
            break
        V = np.max(Q, axis=1)

    pi = lambda s: {s: a for s, a in enumerate(np.argmax(Q, axis=1))}[s]  # noqa: E731
    return V, pi
