# PokerPlayer

A Reinforement Learning Texas Holdem poker player.

Here is a 3 person example tournament, with a RL player, a very simple rational player (which only considers hand type, but not individual cards), and a random player.  
<img src="https://github.com/jcvdwlt/PokerPlayer/blob/master/outcomes.png">
As expeced, the rational player initially outperforms its opponents.  The RL player's randomness is reduced linearly over the first half of the tournament, after which it is fixed at 5%.
