import numpy as np
import unittest
from core import Evaluate, Winner


class TestEvaluate(unittest.TestCase):

    def test_straight_flush(self):
        hand = np.array([[1, 1, 3],
                         [2, 2, 3]])

        table = np.array([[1, 3, 3],
                          [2, 4, 3],
                          [3, 5, 3],
                          [3, 2, 2]])
        # test using evaluate method, instead of the individual methods
        r, s = Evaluate(hand, table).evaluate()
        self.assertEqual(r, 9)
        self.assertEqual(s, [5])

    def test_4_of_a_kind(self):
        hand = np.array([[1, 1, 2],
                         [2, 1, 3]])

        table = np.array([[1, 1, 3],
                          [2, 1, 3],
                          [3, 5, 3],
                          [3, 2, 2]])

        r, s = Evaluate(hand, table).evaluate()
        self.assertEqual(r, 8)
        self.assertEqual(s, [1, 5])

    def test_full_house(self):
        #  TODO: what if we have 2x 3-of-a-kind in 6/7 cards?
        hand = np.array([[1, 1, 2],
                         [2, 1, 3]])

        table = np.array([[1, 2, 3],
                          [2, 2, 2],
                          [3, 2, 3],
                          [3, 3, 2]])

        r, s = Evaluate(hand, table).evaluate()
        self.assertEqual(r, 7)
        self.assertEqual(s, [2, 1])

    def test_flush(self):
        hand = np.array([[1, 1, 2],
                         [2, 2, 2]])

        table = np.array([[1, 2, 2],
                          [2, 3, 2],
                          [3, 2, 2],
                          [3, 5, 2]
                          ])

        r, s = Evaluate(hand, table).evaluate()
        self.assertEqual(r, 6)
        self.assertEqual(s, [5, 3, 2, 2, 2])

    def test_straight(self):

        hand = np.array([[1, 0, 1],
                         [2, 1, 2]])

        table = np.array([[1, 2, 0],
                          [2, 3, 2],
                          [3, 4, 2],
                          [3, 7, 2]
                          ])

        r, s = Evaluate(hand, table).evaluate()
        self.assertEqual(r, 5)
        self.assertEqual(s, [4])

    def test_3_of_a_kind(self):

        hand = np.array([[1, 1, 1],
                         [2, 2, 2]])

        table = np.array([[1, 2, 0],
                          [2, 2, 2],
                          [3, 4, 2],
                          [3, 5, 2]
                          ])

        r, s = Evaluate(hand, table).evaluate()
        self.assertEqual(r, 4)
        self.assertEqual(s, [2, 5, 4])

    def test_two_pair(self):

        hand = np.array([[1, 1, 1],
                         [2, 1, 2]])

        table = np.array([[1, 2, 0],
                          [2, 2, 2],
                          [3, 4, 2],
                          [3, 5, 2]
                          ])

        r, s = Evaluate(hand, table).evaluate()
        self.assertEqual(r, 3)
        self.assertEqual(s, [2, 1, 5])

    def test_2_of_a_kind(self):
        hand = np.array([[1, 1, 1],
                         [2, 2, 2]])

        table = np.array([[1, 1, 0],
                          [2, 3, 2],
                          [3, 4, 2],
                          [3, 6, 2]
                          ])

        r, s = Evaluate(hand, table).evaluate()
        self.assertEqual(r, 2)
        self.assertEqual(s, [1, 6, 4, 3])

    def test_high_card(self):

        hand = np.array([[1, 1, 1],
                         [2, 2, 2]])

        table = np.array([[1, 7, 0],
                          [2, 3, 2],
                          [3, 4, 2],
                          [3, 6, 2]
                          ])

        r, s = Evaluate(hand, table).evaluate()
        self.assertEqual(r, 1)
        self.assertEqual(s, [7, 6, 4, 3, 2])


class TestWinner(unittest.TestCase):
    def test_winner(self):
        w = Winner(['Will', 'John', 'Alice'], [[5, 5, 3], [5, 5], [4, 8, 9, 10, 11]])
        self.assertEqual(w.find_winners(), ['Will'])

        w = Winner(['Will', 'John', 'Alice'], [[5, 5, 3], [5, 5, 3], [4, 8, 9, 10, 11]])
        self.assertListEqual(w.find_winners(), ['Will', 'John'])


