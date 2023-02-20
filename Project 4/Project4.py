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
    We simulation rounds of auctions and record the revenue derived based on reserve_prices
    
    Args:
      bidders: the number of bidders
      inverse_distribution: a function that takes a random number between 0 and 1 and returns a value
    from the distribution
      items: number of items to be sold. Defaults to 1
    
    Returns:
      a payoff matrix with actions for each reserve price and round
    """
    v = np.zeros((len(reserve_prices), rounds))
    
    for round in range(v.shape[1]):
        bidder_values = [inverse_distribution(np.random.uniform()) for _ in range(bidders)]
        bidder_values.sort()
        
        revenues = [0 if bidder_values[bidders-1] < rp else max(bidder_values[bidders-2], rp) for rp in reserve_prices]
        v[:, round] = revenues    
    return v

def expected_reserve_price_from_EW(bidders, inverse_distribution, items=1):
    """
    It takes a list of bidders, an inverse distribution, and the number of items, and returns the
    expected reserve price
    
    Args:
        bidders: the number of bidders
        inverse_distribution: the inverse distribution of the bidders' values
        items: number of items to be sold, defaults to 1 (optional)
    
    Returns:
        the expected reserve price from exponential weights
    """
    reserve_prices = np.divide(list(range(0, 100)), 100)
    rounds = 100000
    optimal_epsilon = np.sqrt(np.log(len(reserve_prices))/rounds)
    v = make_reserve_price_payoffs(bidders, inverse_distribution, reserve_prices, rounds, items)
    _, probabilities = exponential_weights(v, optimal_epsilon, 1)
    
    expected_reserve_price = np.sum([prob[rounds-1]*rp for prob, rp in zip(probabilities, reserve_prices)])
    return expected_reserve_price
    
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

def part1():
    for inverse_distribution in [F_1_inverse, F_2_inverse]:
        for bidders in range(2, 20):
            items = 1 # add iterating over items for items in range(bidders+1, 21)
            exp_rp = expected_reserve_price_from_EW(bidders, inverse_distribution, items)
            opt_rp = 0
            # store difference in exp_rp and opt_rp to plot

if __name__ == '__main__':
    print(expected_reserve_price_from_EW(2, F_1_inverse, items=1))