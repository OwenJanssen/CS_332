import numpy as np
import math

def exponential_weights(v, epsilon, h):
    """
    It gives suggestions to players for probabilities to pick actions given an array of previous payoffs 
    of those actions.

    Args:
      v: an array of payoffs for each action at a given turn
      epsilon: the learning rate
      h: the range of the payoffs
      
    Returns:
      weights: weights for each action at a given turn
      pi: optimal probabilities for picking each action at a given turn
    """
    V = np.cumsum(v, axis=1)
    weights = np.power((1 + epsilon), V[:, :-1]/h)
    # because we're referencing the previous column, the first
    # column of ones gets removed so we need to add it back
    weights = np.insert(weights, 0, np.ones(v.shape[0]), axis=1)
    # print(f"SUM:{np.sum(weights, axis=0)}" )
    pi = np.divide(weights, np.sum(weights, axis=0))
    return weights, pi


def make_reserve_price_payoffs(bidders, inverse_distribution, reserve_prices, rounds, items=1):
    """
    We simulation rounds rounds of auctions and record the revenue derived based on reserve prices between 0 and 1 with a step of 0.01
    
    Args:
      bidders: the number of bidders
      inverse_distribution: a function that takes a random number between 0 and 1 and returns a value
    from the distribution
      items: number of items to be sold. Defaults to 1
    
    Returns:
      a payoff matrix with actions for each reserve price and 100 rounds
    """
    v = np.zeros((len(reserve_prices), rounds))
    
    for round in range(v.shape[1]):
        bidder_values = [inverse_distribution(np.random.uniform()) for _ in range(bidders)]
        bidder_values.sort()
        
        revenues = [0 for rp in reserve_prices]
        for i, reserve_price in enumerate(reserve_prices):
            # fix later for multiple items
            revenues[i] = 0 if bidder_values[0] < reserve_price else max(bidder_values[1], reserve_price)
        v[:, round] = revenues
    return v
        
def expected_reserve_price_from_EW(bidders, inverse_distribution, items=1):
    reserve_prices = np.divide(list(range(0, 10)), 10)
    rounds = 100
    optimal_epsilon = np.sqrt(np.log(len(reserve_prices))/rounds)
    v = make_reserve_price_payoffs(bidders, inverse_distribution, reserve_prices, rounds, items)
    _, probabilities = exponential_weights(v, optimal_epsilon, 1)
    
    expected_reserve_price = np.sum([probabilities[i][rounds-1]*reserve_prices[i] for i in range(len(reserve_prices))])
    return probabilities[:, rounds-1]
    
def F_1(z):
    return z

def F_1_inverse(z):
    return z

def F_1_density(z):
    return 1

def F_2(z):
    return z ** 2

def F_2_inverse(z):
    return math.sqrt(z)

def F_2_density(z):
    return 2 * z


def optimal_values(reserve, distribution, bids, items, density ):
    # expected virtual welfare = expected revenue
    # inverse virtual value = optimal reserve price 
    
   # expected_revenue = np.sum(v - ((1 - distribution(v)) / density(v)) for v in bids)
    return 
    
    
if __name__ == '__main__':
    print(expected_reserve_price_from_EW(2, F_1_inverse, items=1))