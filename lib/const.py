import numpy as np

SUIT = {0: 'Spades',
        1: 'Diamonds',
        2: 'Clubs',
        3: 'Hearts'
        }

NUMBER = dict(zip(range(9), (np.arange(2, 11)).astype('str')))
NUMBER.update(zip(range(9, 12), ['Jack', 'Queen', 'King']))
NUMBER.update([(12, 'Ace')])

HANDS = {
    1: 'High card',
    2: 'Pair',
    3: 'Two pair',
    4: '3 of a kind',
    5: 'Straight',
    6: 'Flush',
    7: 'Full house',
    8: '4 of a kind',
    9: 'Straight flush'
         }


class Actions:
    fold = 0
    call_bet = 1
    raise_bet = 2
    small_blind = 3
    big_blind = 4


ACTIONS = {
    0: 'fold',
    1: 'call',
    2: 'raise',
    3: 'small blind',
    4: 'big blind'
}


def print_outcome(score_vec):
    if score_vec[0] in [9, 6, 5]:
        return HANDS[score_vec[0]] + ' with highest card ' + NUMBER[score_vec[1]]

    elif score_vec[0] in [8, 4, 2]:
        return HANDS[score_vec[0]] + ' of ' + NUMBER[score_vec[1]] + ' with kicker ' + NUMBER[score_vec[2]]

    elif score_vec[0] in [7]:
        return HANDS[score_vec[0]] + ' with ' + NUMBER[score_vec[1]] + ' full of ' + NUMBER[score_vec[2]]

    elif score_vec[0] in [3]:
        return HANDS[score_vec[0]] + ' of ' + NUMBER[score_vec[1]] + ' and ' + NUMBER[score_vec[2]] + ' with kicker ' + NUMBER[score_vec[3]]

    elif score_vec[0] in [1]:
        return HANDS[score_vec[0]] + ' ' + NUMBER[score_vec[1]]


def print_card(_, number, suit):
    print('{0:5} of {1}'.format(NUMBER[number], SUIT[suit]))


def print_cards(card_array):
    print('-'*20)
    for i in range(card_array.shape[0]):
        print_card(*card_array[i, :])
    print('-' * 20)
