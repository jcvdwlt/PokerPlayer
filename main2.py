from core import Player, Game, Policy, MLPValueFunction, BlindPolicy
import numpy as np
import matplotlib.pyplot as plt


vf1 = MLPValueFunction(file_name='p1.joblib')
vf1.load_model()

N_RUNS = 10
N_GAMES = 50
N_PLAYERS = 3
outcomes = np.zeros(shape=[N_GAMES, N_PLAYERS, N_RUNS])

for run in range(N_RUNS):
    for seed in range(N_GAMES):
        # np.random.seed(seed)
        print('GAME: {}'.format(seed))

        # learning player
        p1 = Player(policy=Policy(value_function=vf1, greedy=True, random=False), name='JC', learn=True)

        # fixed rational strategy based on hand, better than random
        p2 = Player(policy=BlindPolicy(), name='Jack', learn=False)
        p3 = Player(policy=Policy(value_function=vf1, greedy=False, random=True), name='Lee', learn=True)

        dealer = seed % N_PLAYERS
        g = Game([p1, p2, p3], dealer=dealer)
        g.play_round()
        outcomes[seed, :, run] = g.outcomes


print(outcomes.sum(axis=0))
vf1.save_model()
np.save('mlp_rb_rand_outcomes.npy', outcomes)


# TODO: make interface to play with human
#       make better probabilistic player
#       improve explore vs exploit play in RL

# let's look at the cumulative outcomes
# outcomes = np.cumsum(outcomes, axis=0)
# f, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 10))
# ax.plot(np.arange(N_GAMES), outcomes[:, 0], label='MLP')
# ax.plot(np.arange(N_GAMES), outcomes[:, 1], label='Rule based')
# ax.plot(np.arange(N_GAMES), outcomes[:, 2], label='Random')
# ax.set_ylim([-1000, 1000])
# ax.legend()
# ax.set_title('All')
# plt.show()



