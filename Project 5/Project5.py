import numpy as np
import math
import matplotlib.pyplot as plt
import random

def exponential_weights(v, epsilon, h):
    """
    It gives suggestions to players for probabilities to pick actions given an array of previous payoffs 
    of those actions.

    Args:
      v: an array of payoffs for each action at a given turn
      epsilon: the learning rate
      h: the range of the payoffs
      
    Returns:
      the optimal probabilities for picking each action at a given turn
    """
    V = np.cumsum(v, axis=1)
    weights = np.power((1 + epsilon), V[:, :-1]/h)
    # because we're referencing the previous column, the first
    # column of ones gets removed so we need to add it back
    weights = np.insert(weights, 0, np.ones(v.shape[0]), axis=1)
    pi = np.divide(weights, np.sum(weights, axis=0))
    return pi

#part 1
def make_bids_and_qualities(bidders, rounds):
    qualities = np.random.rand(bidders, rounds)
    bids = np.zeros((bidders, rounds))
    possible_bids = [i/100 for i in range(100)]

    bidder_payoffs = [np.zeros((len(possible_bids), rounds)) for _ in range(bidders)]        
    for round in range(rounds):
        for bidder in range(bidders):
            # generate payoffs for possible bids for the previous round, given the other bidders bids
            for bid_i, bid in enumerate(possible_bids):
                this_round_bids = bids[:, round]
                this_round_bids[bidder] = bid
                winner = np.argmax(np.multiply(qualities[:, round], this_round_bids))
                
                # assign the payoff as the utility from the bid
                bidder_payoffs[bidder][bid_i][round] = 1 - bid if winner == bidder else 0
            # run EW on payoffs to get bid for next round
            pi = exponential_weights(bidder_payoffs[bidder], 1, 1)
            
            # bid the expected value of the learning algorithm (sum of bid * pr[pick bid] according to the learning algorithm)
            bid = np.sum(np.multiply(pi[:, round], possible_bids))
            bids[bidder][round] = bid
            
    return bids, qualities

bids, qualities = make_bids_and_qualities(2, 5)
print(bids, qualities)