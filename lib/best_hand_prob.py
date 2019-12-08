from joblib import dump
from collections import defaultdict, Counter

from lib.core import Evaluate, Deck, Winner
from lib.const import HANDS


def get_flop_river_turn_r(hand, table):
    pf = Evaluate(hand[:, 1:], table[0:0, 1:]).eval_vec()
    f = Evaluate(hand[:, 1:], table[0:3, 1:]).eval_vec()
    t = Evaluate(hand[:, 1:], table[0:4, 1:]).eval_vec()
    r = Evaluate(hand[:, 1:], table[0:5, 1:]).eval_vec()

    r_l = [pf[0], f[0], t[0], r[0]]

    return r_l, r


hd = defaultdict(Counter)
wd = defaultdict(Counter)

# 3 player rational blind player
for _ in range(int(1e4)):
    d = Deck()
    d.shuffle()
    table = d.deal(n_cards=5)
    p1 = d.deal(n_cards=2)
    p2 = d.deal(n_cards=2)
    p3 = d.deal(n_cards=2)

    frt1, s1 = get_flop_river_turn_r(p1, table)
    frt2, s2 = get_flop_river_turn_r(p2, table)
    frt3, s3 = get_flop_river_turn_r(p3, table)

    w = Winner([frt1, frt2, frt3], [s1, s2, s3]).find_winners()

    for i in range(4):
        # occurances
        hd[frt1[i]][i] += 1
        hd[frt2[i]][i] += 1
        hd[frt3[i]][i] += 1

        # wins
        for winner in w:
            wd[winner[i]][i] += 1

# make prob_win_dict
for i, v in wd.items():
    for si, sv in v.items():
        wd[i][si] = sv / hd[i][si]


print('Best hand probability')
print('{0:20} {1:>10} {2:>10} {3:>10}  {4:>10}'.format('HANDS', 'pre-flop', 'flop', 'river', 'turn'))
for i in range(1,10):
    print('{0:20} {1:10.3f} {2:10.3f} {3:10.3f} {4:10.3f}'.format(HANDS[i], wd[i][0], wd[i][1], wd[i][2], wd[i][3]))

dump(wd, 'dumps/3_player_best_hand_dict.joblib')



