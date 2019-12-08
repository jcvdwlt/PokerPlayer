from lib.core import Player, Game, Policy, MLPValueFunction, RationalPolicy, RandomPolicy
import numpy as np


runs = 500
state_histories = np.zeros(shape=[0, 27])

for _ in range(runs):
    p1 = Player(policy=RandomPolicy(), name='p1', learn=False)
    p2 = Player(policy=RandomPolicy(), name='p2', learn=False)
    p3 = Player(policy=RandomPolicy(), name='p3', learn=False)

    g = Game(players=[p1, p2, p3])

    g.play_round()

    state_histories = np.vstack([state_histories, p1.state_history, p2.state_history, p3.state_history])


np.save('dumps/init_mlp.npy', state_histories)

