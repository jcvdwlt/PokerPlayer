import matplotlib.pyplot as plt
import numpy as np


def plot_outcomes(outcomes, ax):
    n_games, n_players, n_runs = outcomes.shape
    outcomes = np.cumsum(outcomes, axis=0)
    means = np.mean(outcomes, axis=2)
    mins = np.percentile(outcomes, q=90, axis=2)
    maxes = np.percentile(outcomes, q=10, axis=2)

    # std = np.std(outcomes, axis=2)
    # mins = means - std/2
    # maxes = means + std/2

    cols = ['C0', 'C1', 'C2']

    xx = range(n_games)

    for i in range(n_players):
        ax.plot(xx, means[:, i], c=cols[i])
        ax.fill_between(xx, mins[:, i], maxes[:, i], color=cols[i], alpha=0.3)

    ax.set_xlabel('Round', fontsize=14)
    ax.set_ylabel('Cumulative outcome', fontsize=14)


outcomes = np.load('mlp_rb_rand_outcomes.npy')
f, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 10))
plot_outcomes(outcomes, ax)
ax.legend(['MLP value function', 'Probability rule based', 'Random play'], fontsize=14)
plt.show()
