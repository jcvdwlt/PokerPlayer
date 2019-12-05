import copy
from joblib import dump, load

import numpy as np
from collections import Counter
from sklearn.neural_network import MLPRegressor

from const import print_cards, print_outcome, ACTIONS, Actions


BIG_BLIND = 4
SMALL_BLIND = 2
POT = 100

EVAL_VEC_LEN = 28


def to_one_hot(vector, n_states):
    try:
        n = len(vector)
        vector = vector.astype(int)
    except TypeError:
        n = 1

    out_v = np.zeros([n, n_states])

    if vector:
        vector = int(vector)
        out_v[:, vector] = 1

    return out_v


class Deck:
    def __init__(self, n_decks=1):
        self.n_decks = n_decks
        self.cards = np.hstack([
            np.arange(52 * n_decks).reshape([52 * n_decks, 1]),
            np.tile(np.arange(13), [1, 4 * n_decks]).reshape(52 * n_decks, 1),
            np.tile(np.arange(4), [13 * n_decks, 1]).transpose().reshape([52 * n_decks, 1])]
        )
        self.dealt_ind = 0

    def show_all(self):
        print_cards(self.cards)

    def shuffle(self):
        self.cards = self.cards[np.random.permutation(np.arange(52 * self.n_decks))]

    def deal(self, n_cards):
        deal_cards = self.cards[self.dealt_ind:self.dealt_ind+n_cards, :]
        self.dealt_ind += n_cards
        return deal_cards


class Winner:
    def __init__(self, players, scores):
        assert len(players) == len(scores)

        self.scores = np.zeros([len(players), 7], dtype=int)
        for i in range(len(players)):
            self.scores[i, 0:len(scores[i])] = np.array(scores[i])

        self.players = np.array(players)

    def find_winners(self):
        w_players = np.ones(len(self.players), dtype=bool)
        for i in range(7):
            mx = self.scores[:, i][w_players].max()
            w_players = (self.scores[:, i] == mx) & w_players
            if w_players.sum() == 1:
                return list(self.players[w_players])

        return list(self.players[w_players])


class Evaluate:

    def __init__(self, hand, table):
        self.hand = hand
        self.table = table
        self.cards = np.vstack([hand, table])
        self.numbers = self.cards[:, 0]
        self.suits = self.cards[:, 1]
        self.out = None

    def eval_vec(self):
        out_vec = np.zeros(7)
        r, s = self.evaluate()
        out = [r]
        out.extend(s)
        out_vec[0:len(out)] = np.array(out)
        return out_vec

    def evaluate(self):
        if self.find_straight_flush():
            return 9, self.out
        elif self.find_4_of_a_kind():
            return 8, self.out
        elif self.find_full_house():
            return 7, self.out
        elif self.find_flush():
            return 6, self.out
        elif self.find_straight():
            return 5, self.out
        elif self.find_3_of_a_kind():
            return 4, self.out
        elif self.find_two_pair():
            return 3, self.out
        elif self.find_2_of_a_kind():
            return 2, self.out
        else:
            self.find_high_card()
            return 1, self.out

    def find_high_card(self):
        self.out = sorted(self.numbers)[-5:][::-1]
        return self.out

    def find_2_of_a_kind(self):
        pair = self.find_n_of_a_kind(n=2)
        if pair:
            kickers = sorted(self.numbers[self.numbers != pair[0]])[-3:][::-1]
            self.out = [pair[0]]
            self.out.extend(kickers)
            return True

    def find_3_of_a_kind(self):
        num = self.find_n_of_a_kind(n=3)
        if num is not None:
            kickers = sorted(self.numbers[self.numbers != num[0]])[-2:][::-1]
            self.out = [num[0]]
            self.out.extend(kickers)
            return True

    def find_4_of_a_kind(self):
        num = self.find_n_of_a_kind(n=4)
        kicker = sorted(self.numbers[self.numbers != num])[-1]
        if num:
            self.out = [num[0], kicker]
            return num

    def find_two_pair(self):
        pairs = self.find_n_of_a_kind(n=2)
        if pairs is not None:
            if len(pairs) > 1:
                two_pairs = sorted(pairs)[-2:][::-1]
                kicker = sorted(self.numbers[(self.numbers != two_pairs[0]) & (self.numbers != two_pairs[1])])[-1]
                self.out = two_pairs
                self.out.append(kicker)
                return True

    def find_full_house(self):
        three = self.find_n_of_a_kind(n=3)
        two = self.find_n_of_a_kind(n=2)
        if two is not None and three is not None:
            self.out = [three[0], two[0]]
            return self.out
        elif three is not None:
            if len(three) == 2:
                self.out = list(three)
                return self.out

    def find_straight(self):
        self.out = self.find_run(self.numbers)
        return self.out

    @staticmethod
    def find_run(num_list):
        numbers = sorted(list(set(num_list)))
        cnt = 1
        highest_str = []
        for i in range(len(numbers) - 1):
            if numbers[i + 1] == numbers[i] + 1:
                cnt += 1
            else:
                cnt = 1

            if cnt >= 5:
                highest_str.append(numbers[i + 1])

        if len(highest_str) > 0:
            return [highest_str[-1]]

    def find_flush(self):
        s = self.find_same_suit()
        if s is not None:
            self.out = sorted(self.numbers[self.suits == s])[-5:][::-1]
            return self.out

    def find_same_suit(self):
        suits = Counter(self.suits)
        for suit, cnt in suits.items():
            if cnt >= 5:
                return suit

    def find_straight_flush(self):
        s = self.find_same_suit()
        if s:
            self.out = self.find_run(self.numbers[self.suits == s])
            return self.out

    def find_n_of_a_kind(self, n):
        numbers = Counter(self.numbers)
        pairs = list()
        for num, cnt in numbers.items():
            if cnt == n:
                pairs.append(num)

        if len(pairs) == 0:
            pairs = None
        else:
            pairs = np.sort(pairs)[::-1]

        return pairs


class Game:
    def __init__(self, players, dealer=0):
        self.players = players
        self.round = 0  # 0: pre-flop, 1: flop, 2: turn, 3: river, 4: showdown
        self.n_players = len(players)
        self.current_player = dealer
        self.dealer = dealer
        self.deck = Deck()
        self.table = None
        self.visible_table = None
        self.outcomes = None

    def play_round(self):
        self.setup_bets()
        self.deal()
        print('Pre-flop:')
        self.betting_round()

        if self.game_active() and self.round == 1:
            self.flop()
            self.betting_round()

        if self.game_active() and self.round == 2:
            self.turn()
            self.betting_round()

        if self.game_active() and self.round == 3:
            self.river()
            self.betting_round()

        if self.game_active() and self.round == 4:
            active_players = [player for player in self.players if player.active]
            outcomes = [player.evaluate(self.visible_table) for player in active_players]
            winners = Winner(active_players, outcomes).find_winners()

            print('Showdown:')
            for player in active_players:
                player.print_eval(self.visible_table)
                if player in winners:
                    player.win(len(winners))
                else:
                    player.fold()

        # everyone but one player folded.
        else:
            self.next_player()  # find the active player
            self.active_player().win()

        for player in self.players:
            player.final_update()  # learn from the final outcome

        self.outcomes = np.array([player.outcome for player in self.players])

    def deal(self):
        self.deck.shuffle()
        for player in self.players:
            player.set_hand(hand=self.deck.deal(n_cards=2))

        self.table = self.deck.deal(n_cards=5)
        self.visible_table = self.table[0:0, :]

    def flop(self):
        self.visible_table = self.table[0:3, :]
        print('Flop:')
        print_cards(self.visible_table)

    def turn(self):
        self.visible_table = self.table[0:4, :]
        print('Turn:')
        print_cards(self.visible_table[3:4, :])

    def river(self):
        self.visible_table = self.table[0:5, :]
        print('River:')
        print_cards(self.visible_table[4:5, :])

    def bets_equal(self):
        bets = np.array(self.get_active_bets())
        return all((bets == min(bets)))

    def all_acted(self):
        return all([player.acted for player in self.players if player.active])

    def get_active_bets(self):
        return [player.bets for player in self.players if player.active]

    def get_bets(self):
        return [player.bets for player in self.players]

    def max_bet(self):
        return max(self.get_bets())

    def get_statusses(self):
        return [player.active for player in self.players]

    def get_player_opponent_bets(self, i):
        b = self.get_bets()
        s = self.get_statusses()
        return b[0:i] + b[i+1:], s[0:i] + s[i+1:]

    def update_players_opponent_bets(self):
        for i in range(self.n_players):
            self.players[i].opponent_bets, self.players[i].active_opponents = self.get_player_opponent_bets(i)

    def next_player(self):
        self.current_player = (self.current_player + 1) % self.n_players
        while not self.active_player().active:
            self.current_player = (self.current_player + 1) % self.n_players

        self.update_players_opponent_bets()
        return self.current_player

    def betting_round(self):
        self.set_all_not_acted()
        if self.round == 0:
            self.current_player = (self.dealer + 2) % self.n_players
        else:
            self.current_player = self.dealer
        self.next_player()

        while (not self.all_acted() or not self.bets_equal()) and self.game_active():
            self.active_player().make_move(self.visible_table, self.round, self.max_bet())
            self.next_player()

        self.round += 1

    def active_player(self):
        return self.players[self.current_player]

    def setup_bets(self):
        self.next_player()
        self.active_player().set_small_blind()
        self.next_player()
        self.active_player().set_big_blind()
        self.next_player()

    def set_all_not_acted(self):
        for player in self.players:
            if player.active:
                player.acted = False

    def game_active(self):
        active_players = 0
        for player in self.players:
            if player.active:
                active_players += 1

        return active_players > 1


class Player:
    def __init__(self, policy, name, learn=True):
        self.name = name
        self.bets = 0
        self.pot = POT
        self.hand = None
        self.outcome = None
        self.last_act = None

        self.acted = False
        self.active = True

        self.small_blind = False
        self.big_blind = False

        self.policy = policy

        self.active_opponents = []
        self.opponent_bets = []

        self.previous_state = None
        self.current_state = None

        self.learn = learn

    def set_hand(self, hand):
        print(self.name)
        print_cards(hand)
        self.hand = hand
        self.current_state = self.get_state(np.zeros(shape=[5, 3])[0:0, :], 0)

    def set_small_blind(self):
        self.small_blind = True
        self._place_bet(SMALL_BLIND)
        self.last_act = 3
        print('Small blind: ' + self.name + ' puts in ' + str(SMALL_BLIND))

    def set_big_blind(self):
        self.big_blind = True
        self._place_bet(BIG_BLIND)
        self.last_act = 4
        print('Big blind: ' + self.name + ' puts in ' + str(BIG_BLIND))

    def _place_bet(self, amount):
        self.bets += amount
        self.pot -= amount

    def raise_bet(self, max_previous_bet):
        target = max_previous_bet + BIG_BLIND
        bet = target - self.bets
        self.acted = True
        self.last_act = 0
        return self._place_bet(bet)

    def call_bet(self, max_previous_bet):  # or check
        bet = max_previous_bet - self.bets
        self.acted = True
        self.last_act = 1
        return self._place_bet(bet)

    def fold(self):
        self.acted = True
        self.active = False
        self.outcome = -self.bets
        self.last_act = 2

    def win(self, n_winners=1):
        self.active = False
        self.outcome = sum(self.opponent_bets) / n_winners
        print(self.name + ' wins ' + str(self.outcome))

    def get_state(self, table, rnd):
        r = Evaluate(self.hand[:, 1:], table[:, 1:]).eval_vec()
        ob = np.array(self.opponent_bets, dtype=float)
        ao = np.array(self.active_opponents, dtype=bool)
        rnd = np.array(rnd, dtype=int)
        mb = np.array(self.bets, dtype=float)
        p = np.array(self.pot, dtype=float)
        # last_act = np.array(self.last_act, dtype=int)
        # 9, 6, n-1, n-1, 1, 1, 1, 4
        out = np.hstack([to_one_hot(np.array(r[0]-1), n_states=9).flatten(), r[1:], ob, ao, rnd, mb, p, to_one_hot(self.last_act, n_states=5).flatten()])
        # print(out[np.newaxis, :])
        return out[np.newaxis, :]

    def evaluate(self, table):
        return Evaluate(self.hand[:, 1:], table[:, 1:]).eval_vec()

    def make_move(self, table, rnd, max_bet):

        self.previous_state = self.get_state(table, rnd)
        act = self.policy.choose_action(self, table, rnd, max_bet)

        if act == 0.:
            self.fold()
            self.last_act = 0
            print(self.name + ' folds at ' + str(self.bets))
        elif act == 1:
            self.call_bet(max_bet)
            self.last_act = 1
            print(self.name + ' calls to ' + str(self.bets))
        elif act == 2:
            self.raise_bet(max_bet)
            self.last_act = 2
            print(self.name + ' raises to ' + str(self.bets))
        else:
            print(act)
            raise Exception('Invalid move!')

        self.current_state = self.get_state(table, rnd)
        self.update_value_function()

        self.policy.print_action_values()

    def update_value_function(self):
        if self.learn:
            nv = self.policy.value_function.get_value(self.current_state)
            self.policy.update_value_function(self.previous_state, nv)

    def final_update(self):
        self.policy.update_value_function(self.current_state, np.array([self.outcome]))

    def print_hand(self):
        print(self.name)
        print_cards(self.hand)

    def print_eval(self, table):
        v = self.evaluate(table)
        print(self.name + ' has ' + print_outcome(v))


class BlindPolicy:
    def __init__(self, random=False):
        self.lookup = load('3_player_best_hand_dict.joblib')
        self.prob_best_hand = None
        self.random = random

    def choose_action(self, player, table, rnd, max_bet):
        r, _ = Evaluate(player.hand[:, 1:], table[:, 1:]).evaluate()
        self.prob_best_hand = self.lookup[r][rnd]

        if rnd == 0:
            return Actions.call_bet
        elif self.prob_best_hand < 0.3:
            return Actions.fold
        elif self.prob_best_hand < 0.6:
            return Actions.call_bet
        else:
            if player.acted:
                return Actions.call_bet
            else:
                return Actions.raise_bet

    def update_value_function(self, state, value):
        pass

    def print_action_values(self):
        print('Prob best hand: {0:2.3f}'.format(self.prob_best_hand))


class Policy:
    def __init__(self, value_function, greedy=False, random=False):
        self.value_function = value_function
        self.current_state = None
        self.pv0 = None
        self.pv = None
        self.greedy = greedy
        self.random = random

    @staticmethod
    def possible_states(player, table, rnd, max_bet):
        raise_p = copy.deepcopy(player)
        call_p = copy.deepcopy(player)
        fold_p = copy.deepcopy(player)
        raise_p.raise_bet(max_bet)
        call_p.call_bet(max_bet)
        fold_p.fold()
        if player.acted:
            out = (call_p.get_state(table, rnd),
                   fold_p.get_state(table, rnd))
        else:
            out = (raise_p.get_state(table, rnd),
                   call_p.get_state(table, rnd),
                   fold_p.get_state(table, rnd))

        return out

    def possible_values(self, player, table, rnd, max_bet):
        out = np.array([self.value_function.get_value(state) for state in self.possible_states(player, table, rnd, max_bet)])
        return out

    def choose_action(self, player, table, rnd, max_bet):
        self.pv0 = self.possible_values(player, table, rnd, max_bet)

        self.pv = self.pv0 / (max(self.pv0) - min(self.pv0))  # make the range from 0 to 1 (roughly)
        if min(self.pv) < 0:
            self.pv = self.pv - np.min(self.pv) + 0.2
        self.pv = self.pv / sum(self.pv)

        if self.greedy:
            return np.argmax(self.pv == max(self.pv))
        elif self.random:
            if player.acted:
                return np.random.choice(np.array([0, 1]))
            else:
                return np.random.choice(np.array([0, 1, 2]))
        else:
            p = 0
            rnd = np.random.uniform()

            for i in range(len(self.pv)):
                p += self.pv[i]

                if p > rnd:
                    return i

            # IF THIS PRINTS, THERE IS AN ERROR
            print(self.possible_states(player, table, rnd, max_bet))
            print(self.pv0)

    # always call after an action, with the value of th new state, and once more once the game ends with the outcome
    def update_value_function(self, state, value):
        self.value_function.update(state, value)

    def print_action_values(self):
        for i in range(len(self.pv0)):
            print('{0:10} {1:5.3f}  {2:5.3f}'.format(ACTIONS[i], self.pv0[i, 0], self.pv[i, 0]))


class MLPValueFunction:
    def __init__(self, file_name, hidden_layers=(100,)):
        self.estimator = MLPRegressor(hidden_layers)
        # TODO: the initialization is a little tricky, need some start data to train on.
        # self.estimator.coefs_ = [np.random.normal(size=(27, 100)), np.random.normal(size=(100, 1))]
        # self.estimator.intercepts_ = [np.random.normal(size=(100,)), np.random.normal(size=(1,))]

        rnd = np.random.normal(size=[3, EVAL_VEC_LEN])
        self.estimator.fit(X=rnd[:, :-1], y=rnd[:, -1])
        self.filename = file_name

    def update(self, X, y):
        self.estimator.partial_fit(X, y)

    def get_value(self, X):
        out = self.estimator.predict(X)
        if np.isnan(out):
            print('nan out!')
            print(X)

        return out

    def save_model(self):
        dump(self.estimator, self.filename)

    def load_model(self):
        self.estimator = load(self.filename)

    def set_coefs(self, coefs, interceps):
        self.estimator.coefs_ = coefs
        self.estimator.intercepts_ = interceps




