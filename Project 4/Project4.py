import numpy as np
import math
import matplotlib.pyplot as plt

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
    pi = np.divide(weights, np.sum(weights, axis=0))
    return weights, pi


def make_reserve_price_payoffs(bidders, inverse_distribution, reserve_prices, rounds, items):
    """
    We simulation rounds of auctions and record the revenue derived based on reserve_prices
    
    Args:
      bidders: the number of bidders
      inverse_distribution: a function that takes a random number between 0 and 1 and returns a value
    from the distribution
      items: number of items to be sold
    
    Returns:
      a payoff matrix with actions for each reserve price and round
    """
    v = np.zeros((len(reserve_prices), rounds))
    
    for round in range(v.shape[1]):
        bidder_values = [inverse_distribution(np.random.uniform()) for _ in range(bidders)]
        bidder_values.sort()
        
        if bidders == 1:
            revenues = [0 if bidder_values[0] < rp else rp for rp in reserve_prices]
        else:  
            # fix, consider some items gets sold but not all
            revenues = [0 if bidder_values[bidders-1] < rp else max(bidder_values[bidders-1-items], rp) for rp in reserve_prices]
        v[:, round] = revenues    
    return v

def expected_reserve_price_from_EW(bidders, inverse_distribution, items=1, rounds = 1000):
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
    reserve_prices = np.divide(list(range(0, 10)), 10)
    optimal_epsilon = np.sqrt(np.log(len(reserve_prices))/rounds)
    v = make_reserve_price_payoffs(bidders, inverse_distribution, reserve_prices, rounds, items)
    _, probabilities = exponential_weights(v, optimal_epsilon, 1)
    
    expected_reserve_price = np.sum([prob[rounds-1]*rp for prob, rp in zip(probabilities, reserve_prices)])
    return expected_reserve_price
    
def F_1(v):
    return v

def F_1_inverse(v):
    return v

def F_1_density(v):
    return 1

def F_1_E(low, high, bidders, i):
    return F_1_inverse(((high-low)/(bidders+1))*(bidders-i+1)+low)

distributions_1 = {
    "F": F_1,
    "F inverse": F_1_inverse,
    "f": F_1_density,
    "Optimal RP": 1/2,
    "E": F_1_E
}

def F_2(v):
    return v**2

def F_2_inverse(v):
    return math.sqrt(v)

def F_2_density(v):
    return 2 * v

def F_2_E(low, high, bidders, i):
    return F_2_inverse(((high-low)/(bidders+1))*(bidders-i+1)+low)

distributions_2 = {
    "F": F_2,
    "F inverse": F_2_inverse,
    "f": F_2_density,
    "Optimal RP": (1/3)**(1/2),
    "E": F_2_E
}

def F_3(v):
    return v / 4

def F_3_inverse(v):
    return v*4

def F_3_density(v):
    return 4

def F_3_E(low, high, bidders, i):
    return F_3_inverse(((high-low)/(bidders+1))*(bidders-i+1)+low)

distributions_3 = {
    "F": F_3,
    "F inverse": F_3_inverse,
    "f": F_3_density,
    "Optimal RP": 4/17,
    "E": F_3_E
}

def optimal_values(reserve, distribution, bids, items, density ):
    # expected virtual welfare = expected revenue
    # inverse virtual value = optimal reserve price 
    
   # expected_revenue = np.sum(v - ((1 - distribution(v)) / density(v)) for v in bids)
    return 0

def expected_revenue(distribution, r, bidders, items):
    pr = distribution["F"](r)
    # case 1, no bidders over
    case1 = (pr ** bidders) * 0

    # case 2, only one bidder over
    case2 = (pr**(bidders-1) * (1-pr) * bidders) * r

    # case 3, more than one bidder are over
    case3 = 0
    for i in range(2, bidders+1):
        case3 += (pr**(bidders-i) * (1-pr)**i) * distribution["E"](1, r, i, items)

    return case1 + case2 + case3

def part1():
    distributions = [distributions_1, distributions_3, distributions_2]
    # MAX_BIDDERS = 100
    # x = [list(range(MAX_BIDDERS)) for _ in distributions]
    # y = [[0 for _ in x[0]] for _ in x]
    # for i, dist in enumerate(distributions):
    #     # print("NEW DISTRIBUTION")
    #     for bidders in range(1, MAX_BIDDERS):
    #         exp_rp = expected_reserve_price_from_EW(bidders, dist["F inverse"], 1, 1000)
    #         opt_rp = dist["Optimal RP"]
    #         y[i][bidders]=abs(expected_revenue(dist, exp_rp, bidders, 1)-expected_revenue(dist, opt_rp, bidders, 1))
    #         # print(y[i][bidders])
    # plt.plot(x[0], y[0], label='F(z) = z')
    # plt.plot(x[1], y[1], label='F(z) = z/4')
    # plt.plot(x[2], y[2], label='F(z) = z^2')
    # plt.xlabel("Bidders")
    # plt.ylabel("|Online Learning Revenue - Optimal Revenue|")
    # plt.legend()
    # plt.show()

    MAX_ROUNDS = 10000
    x = [np.multiply(list(range(int(MAX_ROUNDS/100))), 100) for _ in distributions]
    z = [[0 for _ in x[0]] for _ in x]
    for i, dist in enumerate(distributions): 
        for rounds in range(1, MAX_ROUNDS, 100):
            exp_rp = expected_reserve_price_from_EW(2, dist["F inverse"], 1, rounds)
            opt_rp = dist["Optimal RP"]
            z[i][int(rounds/100)]=abs(expected_revenue(dist, exp_rp, 2, 1)-expected_revenue(dist, opt_rp, 2, 1))
            # print(y[i][bidders])
    plt.plot(x[0], z[0], label='F(z) = z')
    plt.plot(x[1], z[1], label='F(z) = z/4')
    plt.plot(x[2], z[2], label='F(z) = z^2')
    plt.xlabel("Rounds")
    plt.ylabel("|Online Learning Revenue - Optimal Revenue|")
    plt.legend()
    plt.show()



# Part 2: Introductions 
# for 2 players: introduce if 2v1 - 1 + 2v2 - 1 > 0
    
# generalized: for i in range bidders: expected_revenue += 2(v_i) + h 
# U[0,h] 




def introductions_actual(employee_distributions, employer_distributions, employee_values, employer_values):
    # distributions is an array that looks like [0, h] for every person
    # values is a list of value for each person
    introductions = np.zeros(len(employee_values), len(employer_values))
    expected_revenues = np.zeros(len(employee_values), len(employer_values))
    for i in range(len(employee_values)):
        for j in range(len(employer_values)):
            expected_revenues[i, j] = 2 * employee_values[i] - employee_distributions[i][1] + 2 * employer_values[j] -  employer_distributions[j][1]
            introductions[i][j] = True if expected_revenues[i][j] > 0 else False
            
            
   
   


if __name__ == '__main__':
    # print(expected_revenue(distributions_1, 0.5, 2, 1), 5/12)
    part1()
