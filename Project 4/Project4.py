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

def revenue(bidder_values, items, rp):
    """
    Returns the revenue for a 2nd price (items+1 price) auction with reserve

    Args:
      bidder_values: values of bidders participating in the auction
      items: number of items being sold
      rp: reserve price of auction
      
    Returns:
      the revenue of the auction
    """
    values_over_rp = [v for v in bidder_values if v > rp]
    bidders_over_rp = len(values_over_rp)
    if items < bidders_over_rp:
        return bidder_values[len(bidder_values)-1-items] * items
    else:
        return rp * bidders_over_rp 

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
    
    for round in range(rounds):
        bidder_values = [inverse_distribution(np.random.uniform()) for _ in range(bidders)]
        bidder_values.sort()
        
        revenues = [revenue(bidder_values, items, rp) for rp in reserve_prices]
        v[:, round] = revenues    
    return v

def expected_reserve_price_from_EW(bidders, distributions, items=1, rounds=1000):
    """
    It takes a list of bidders, an inverse distribution, and the number of items, and returns the
    expected reserve price
    
    Args:
        bidders: the number of bidders
        distributions: the distribution functions of the bidders' values
        items: number of items to be sold
        rounds: number of rounds to use for exponential weights
    
    Returns:
        the expected reserve price from exponential weights
    """
    reserve_prices = np.divide(list(range(0, 100)), 100/distributions["F inverse"](1))
    optimal_epsilon = np.sqrt(np.log(len(reserve_prices))/rounds)/1000
    v = make_reserve_price_payoffs(bidders, distributions["F inverse"], reserve_prices, rounds, items)
    _, probabilities = exponential_weights(v, optimal_epsilon, distributions["F inverse"](1))
    
    expected_reserve_price = np.sum([prob*rp for prob, rp in zip(probabilities[:, rounds-1], reserve_prices)])
    return expected_reserve_price

def expected_revenue(distribution, r, bidders, items):
    """
    The expected revenue is the probability that no one bids over the reserve price times the reserve
    price plus the probability that one person bids over the reserve price times the reserve price plus
    the probability that more than one person bids over the reserve price times the expected value of
    the bids of the people who bid over the reserve price
    
    Args:
      distribution: the distribution of the bidders' values
      r: the reserve price
      bidders: number of bidders
      items: number of items
    
    Returns:
      The expected revenue of the auction.
    """
    pr = distribution["F"](r)

    if (bidders == 1):
        return r * (1-pr)
    
    pr_sum = 0
    # case 1, no bidders over
    case1 = (pr ** bidders) * 0
    pr_sum += (pr ** bidders)

    # case 2, only one bidder over
    case2 = (pr**(bidders-1) * (1-pr) * bidders) * r
    pr_sum += (pr**(bidders-1) * (1-pr) * bidders)

    # case 3, more than one bidder are over
    case3 = 0
    for i in range(2, bidders+1):
        case3 += (pr**(bidders-i) * (1-pr)**i * math.comb(bidders, i)) * distribution["E"](distribution["F"](r), 1, i, min(items+1, i)) * min(items, i)
        pr_sum += (pr**(bidders-i) * (1-pr)**i * math.comb(bidders, i))
        
    return case1 + case2 + case3

# Functions for different CDFs
def F_1(v):
    return v

def F_1_inverse(v):
    return v

def F_1_density(v):
    return 1

def F_1_E(low, high, bidders, i):
    uniform_sample = ((high-low)/(bidders+1))*(bidders-i+1)+low
    return F_1_inverse(uniform_sample)

def vv_1(v):
    return 2*v - 1

distributions_1 = {
    "F": F_1,
    "F inverse": F_1_inverse,
    "f": F_1_density,
    "Optimal RP": 1/2,
    "E": F_1_E,
    "virtual value": vv_1
}

def F_2(v):
    return v**2

def F_2_inverse(v):
    return math.sqrt(v)

def F_2_density(v):
    return 2 * v

def F_2_E(low, high, bidders, i):
    uniform_sample = ((high-low)/(bidders+1))*(bidders-i+1)+low
    return F_2_inverse(uniform_sample)

def vv_2(v):
    return v - ((1 - v**2) / 2*v)

distributions_2 = {
    "F": F_2,
    "F inverse": F_2_inverse,
    "f": F_2_density,
    "Optimal RP": (1/3)**(1/2),
    "E": F_2_E,
    "virtual value": vv_2
}

def F_3(v):
    return v / 4

def F_3_inverse(v):
    return v*4

def F_3_density(v):
    return 1 / 4

def F_3_E(low, high, bidders, i):
    uniform_sample = ((high-low)/(bidders+1))*(bidders-i+1)+low
    return F_3_inverse(uniform_sample)

distributions_3 = {
    "F": F_3,
    "F inverse": F_3_inverse,
    "f": F_3_density,
    "Optimal RP": 2,
    "E": F_3_E
}

def F_4(v):
    return math.sqrt(v)

def F_4_inverse(v):
    return v**2

def f_4(v):
    return 1 / (2 * math.sqrt(v))

def F_4_E(low, high, bidders, i):
    uniform_sample = ((high-low)/(bidders+1))*(bidders-i+1)+low
    return F_4_inverse(uniform_sample)

def vv_4(v):
    return 3 * v - 2 * math.sqrt(v)

distributions_4 = {
    "F": F_4,
    "F inverse": F_4_inverse,
    "f": f_4,
    "Optimal RP": 0,
    "E": F_4_E,
    "virtual value": vv_4
}

# Part 1: 
def part1_bidders():
    """
    It plots the difference in revenue between the optimal reserve price and the expected reserve price
    for each distribution, for each number of bidders
    """
    distributions = [distributions_1, distributions_2, distributions_3]
    MAX_BIDDERS = 250
    bidders = [list(range(1, MAX_BIDDERS)) for _ in distributions]
    revenue_diff = [[0 for _ in bidders[0]] for _ in bidders]
    for i, dist in enumerate(distributions):
        for b in bidders[i]:
            exp_rp = expected_reserve_price_from_EW(b, dist, 1, 1000)
            opt_rp = dist["Optimal RP"]
            revenue_diff[i][b-1]=abs(expected_revenue(dist, opt_rp, b, 1)-expected_revenue(dist, exp_rp, b, 1))
    plt.plot(bidders[0], revenue_diff[0], label='F(z) = z', color="Red")
    plt.plot(bidders[1], revenue_diff[1], label='F(z) = z^2', color="Blue")
    plt.plot(bidders[2], np.divide(revenue_diff[2], 4), label='F(z) = z/4 (scaled to [0, 1])', color="Green")
    plt.xlabel("Bidders")
    plt.ylabel("$\Delta$Revenue")
    plt.title(f"Bidders vs $\Delta$Revenue for 1 item")
    plt.legend()
    plt.show()

def part1_items():   
    """
    It plots the difference in revenue between the optimal reserve price and the expected reserve price
    for different distributions and different numbers of items
    """
    distributions = [distributions_1, distributions_2, distributions_3]
    bidders=25
    items_arr = [list(range(1, bidders)) for _ in distributions]
    revenue_diff = [[0 for _ in items_arr[0]] for _ in items_arr]
    for i, dist in enumerate(distributions):
        for items in items_arr[i]:
            exp_rp = expected_reserve_price_from_EW(bidders, dist, items, 1000)
            opt_rp = dist["Optimal RP"]
            revenue_diff[i][items-1]=expected_revenue(dist, opt_rp, bidders, items) - expected_revenue(dist, exp_rp, bidders, items)
    plt.plot(items_arr[0], revenue_diff[0], label='F(z) = z', color="Red")
    plt.plot(items_arr[1], revenue_diff[1], label='F(z) = z^2', color="Blue")
    plt.plot(items_arr[2], np.divide(revenue_diff[2], 4), label='F(z) = z/4 (scaled to [0, 1])', color="Green")
    plt.xlabel("Items")
    plt.ylabel("$\Delta$Revenue")
    plt.title(f"Items vs $\Delta$Revenue for {bidders} bidders")
    plt.legend()
    plt.show()

def part1_rounds():
    """
    It takes in a distribution, a number of rounds, and a number of bidders, and returns the expected
    revenue difference between the optimal reserve price and the expected reserve price
    """
    distributions = [distributions_1, distributions_2, distributions_3]
    rounds_arr = [[i*100 for i in range(1, 100)] for _ in distributions]
    revenue_diff = [[0 for _ in rounds_arr[0]] for _ in rounds_arr]
    for i, dist in enumerate(distributions): 
        for j, rounds in enumerate(rounds_arr[i]):
            sum = 0
            for _ in range(10):
                exp_rp = expected_reserve_price_from_EW(2, dist, 1, rounds)
                opt_rp = dist["Optimal RP"]
                sum+=abs(expected_revenue(dist, exp_rp, 2, 1)-expected_revenue(dist, opt_rp, 2, 1))
            revenue_diff[i][j]=sum/100
    plt.plot(rounds_arr[0], revenue_diff[0], label='F(z) = z')
    plt.plot(rounds_arr[1], revenue_diff[1], label='F(z) = z/4')
    plt.plot(rounds_arr[2], np.divide(revenue_diff[2], 4), label='F(z) = z^2')
    plt.xlabel("Rounds")
    plt.ylabel("$\Delta$Revenue")
    plt.title(f"Rounds vs $\Delta$Revenue for 2 bidders and 1 item")
    plt.legend()
    plt.show()



# Part 2: Introductions 
# for 2 players: introduce if 2v1 - 1 + 2v2 - 1 > 0
    
# generalized: for i in range bidders: expected_revenue += 2(v_i) + h

# virtual value formulas for each distribution ~ U[0, 1] 
# F(z) = z, F_2(z) = z^2, F_3(z) = sqrt(z) --> computer vv formula
# truthful mechanism = introduce them if the sum of their virtual values is greater than 0
# payoffs: 1 or 0
# 

def make_introduction_payoffs(distributions, rounds):
    v = np.zeros((len(distributions), rounds))
    
    for round in range(rounds):
        payoffs = [0 for _ in distributions]
        values = [dist["virtual value"](np.random.uniform()) for dist in distributions]
        should_introduce = False
        for i in range(len(values)-1):
            # values[0] is the employee
            if values[i+1] + values[0] > 0:
                should_introduce = True
                payoffs[i+1] = 1
        if not should_introduce:
            payoffs[0] = 1
        v[:, round] = payoffs
    return v

def introductions_optimal(employee_distributions, employers_distributions):
    # distributions is an array that looks like [0, h] for every person
    # values is a list of value for each person

    # get expected values for employee and employers for their distributions between 0 and 1
    employee_value = employee_distributions["E"](0, 1, 1, 1)
    employer_values = [dist["E"](0, 1, 1, 1) for dist in employers_distributions]
    
    introductions = np.zeros(1+len(employers_distributions))
    should_meet = False
    for i in range(len(employers_distributions)):
        expected_revenue = employee_distributions["virtual value"](employee_value) + employers_distributions[i]["virtual value"](employer_values[i])
        if expected_revenue > 0:
            introductions[i+1] = 1
            should_meet = True
    if not should_meet:
        introductions[0] = 1

    return introductions

def introductions_EW(distributions, rounds=1000):
    optimal_epsilon = np.sqrt(np.log(len(distributions))/rounds)
    v = make_introduction_payoffs(distributions, rounds)
    _, probabilities = exponential_weights(v, optimal_epsilon, 1)

    return probabilities[:, -1]

def part2():
    rounds = [i*10 for i in range(1, 100)]
    intro_diffs = [0 for _ in rounds]
    
    N = 1000
    for i in range(len(rounds)):
        for _ in range(N):
            distrib = [distributions_1, distributions_2, distributions_4]
            employee_distribution = random.choice(distrib)
            distrib.remove(employee_distribution)
            employer_distributions = [random.choice(distrib)]

            opt_intro = introductions_optimal(employee_distribution, employer_distributions)
            ew_intro = introductions_EW(np.concatenate([[employee_distribution], employer_distributions]))
            intro_diffs[i] += np.sum([abs(o-ew) for o, ew in zip(opt_intro, ew_intro)])/2
        intro_diffs[i] /= N

    plt.plot(rounds, intro_diffs)
    plt.xlabel("Rounds")
    plt.ylabel("$\Delta$Introduction Probabilities")
    plt.title(f"Rounds vs $\Delta$Introduction Probabilities")
    plt.show()
    

if __name__ == '__main__':
    print("TEST PASSED" if abs(expected_revenue(distributions_1, 0.5, 2, 1) - 5/12) < 0.001 else "TEST FAILED")
    #part1_rounds()
    part2()