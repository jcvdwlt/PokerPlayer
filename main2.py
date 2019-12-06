from core import Player, Game, Policy, MLPValueFunction, RationalPolicy, RandomPolicy
import numpy as np
import matplotlib.pyplot as plt


N_RUNS = 8
N_GAMES = 1200
N_PLAYERS = 3
outcomes = np.zeros(shape=[N_GAMES, N_PLAYERS, N_RUNS])

for run in range(N_RUNS):
    vf1 = MLPValueFunction(file_name='p1.joblib')

    for seed in range(N_GAMES):
        # np.random.seed(seed)
        print('GAME: {}'.format(seed))

        greed = max([1 - 2*seed / N_GAMES, 0.05])  # linearly increase greed
        # greed = 0.05

        p1 = Player(policy=Policy(value_function=vf1, greedy=greed), name='JC', learn='incremental')
        p2 = Player(policy=RationalPolicy(), name='Lee', learn=False)
        p3 = Player(policy=RandomPolicy(), name='Jack', learn=False)

        dealer = seed % N_PLAYERS
        g = Game([p1, p2, p3], dealer=dealer)
        g.play_round()
        outcomes[seed, :, run] = g.outcomes

print(outcomes.sum(axis=0))

vf1.save_model()
np.save('outcomes.npy', outcomes)


# TODO: make interface to play with human
#       make better probabilistic player
#       improve explore vs exploit play in RL


