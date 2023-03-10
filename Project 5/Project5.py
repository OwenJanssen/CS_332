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


def random_pick(probability_array):
    """
    It generates a random number between 0 and 1, and then returns the index of the first element in the
    probability array that is greater than the random number

    Args:
      probability_array: an array of probabilities, where the sum of all the probabilities is 1.0

    Returns:
      The index of the element in the array that is greater than the random number.
    """
    random_number = np.random.random()
    cumulative_probability = 0.0
    for index, probability in enumerate(probability_array):
        cumulative_probability += probability
        if random_number < cumulative_probability:
            return index
    return len(probability_array) - 1

# part 1


def make_bids_and_qualities(values, rounds):
    bidders = len(values)
    qualities = np.random.rand(bidders, rounds)
    bids = np.zeros((bidders, rounds))
    possible_bids = [[i*v/100 for i in range(100)] for v in values]

    bidder_payoffs = [np.zeros((len(possible_bids[0]), rounds))
                      for _ in range(bidders)]
    for round in range(rounds):
        for bidder in range(bidders):
            # generate payoffs for possible bids given, given the other bidders bids
            for bid_i, bid in enumerate(possible_bids[bidder]):
                this_rounds_bids = np.copy(bids[:, round])
                this_rounds_bids[bidder] = bid
                winner = np.argmax(np.multiply(
                    qualities[:, round], this_rounds_bids))

                # assign the payoff as the utility from the bid
                bidder_payoffs[bidder][bid_i][round] = 1 - \
                    bid if winner == bidder else 0
            # run EW on payoffs to get bid for next round
            pi = exponential_weights(bidder_payoffs[bidder], 0.1, 1)

            # randomly pick a bid according to the probabilities generated by exponential weights
            bid = possible_bids[bidder][random_pick(pi[:, round])]
            bids[bidder][round] = bid

    return bids, qualities


def delta_x_and_delta_p(bids, qualities, j):
    """
    It takes in the bids and qualities of all the bidders, and the index of the bidder we're interested
    in, and returns the change in the number of times the bidder wins and the change in the bidder's
    payment as a function of the bidder's bid

    Args:
      bids: a matrix of bids, where each row is a bidder and each column is a round
      qualities: a matrix of size (num_agents, num_rounds)
      j: the bidder we're looking at

    Returns:
      delta_x and delta_p are being returned.
    """
    num_z = 100
    delta_x = np.zeros(num_z)
    delta_p = np.zeros(num_z)
    max_z = 5
    n = len(bids[0])
    for i in range(num_z):
        z = max_z * i / num_z

        for round in range(n):
            this_rounds_bids = np.copy(bids[:, round])
            this_rounds_bids[j] = z
            actual_winner = np.argmax(np.multiply(
                qualities[:, round], bids[:, round]))
            z_winner = np.argmax(np.multiply(
                qualities[:, round], this_rounds_bids))

            delta_x[i] += (1 if actual_winner == j else 0) - \
                (1 if z_winner == j else 0)
            delta_p[i] += (bids[j, round] if actual_winner ==
                           j else 0) - (z if z_winner == j else 0)

        delta_x[i] /= n
        delta_p[i] /= n

    return delta_x, delta_p


def infer_value(bids, qualities, j):
    """
    It takes in the bids and qualities of all the other bidders, and the index of the bidder whose value
    we want to infer. It then calculates the delta_x and delta_p values for that bidder, and then
    iterates through all possible values and epsilons to see which value is rationalizable for the
    lowest possible epsilon

    Args:
      bids: a list of the bids of all the bidders
      qualities: a list of the qualities of all the bids for all the bidders
      j: the bidder we're trying to infer the value for

    Returns:
      The highest rationalizable value for the lowest possible epsilon.
    """
    delta_x, delta_p = delta_x_and_delta_p(bids, qualities, j)
    values = [i/100 for i in range(500)]
    epsilons = [i/100 for i in range(500)]
    rationalizable = [[False for _ in range(500)] for _ in range(500)]
    for ii, e in enumerate(epsilons):
        for jj, v in enumerate(values):
            not_rationalizable_z = [True for r in np.subtract(np.multiply(
                v, delta_x), delta_p) if r < -1 * e]
            rationalizable[ii][jj] = len(not_rationalizable_z) == 0
            # pick the highest rationalizable value for the lowest possible epsilon
            if (~rationalizable[ii][jj] & rationalizable[ii][jj-1]):
                return values[jj-1]

    return 5


def part_1():
    N = 5
    rounds_arr = [(i+1)*10 for i in range(10)]
    error_arr = [0 for i in range(10)]
    for rounds_i, rounds in enumerate(rounds_arr):
        for _ in range(N):
            values = [1, 1.5, 2, 2.5]
            bids, qualities = make_bids_and_qualities(values, rounds)
            error = 0
            for i, v in enumerate(values):
                infered_value = round(infer_value(bids, qualities, i), 3)
                print(f"V: {v}, IF: {infered_value}")
                error += (v - infered_value) ** 2
            error /= len(values)
            error_arr[rounds_i] += error
        error_arr[rounds_i] /= N
    plt.plot(rounds_arr, error_arr)
    plt.xlabel("Rounds")
    plt.ylabel("E[(v_i - v'_i)^2]")
    plt.show()


#part_1()
# values = [1, 1.5, 2, 2.5]
# bids, qualities = make_bids_and_qualities(values, 5)
# infer_value(bids, qualities, 0, 2)
# print(bids)

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
        return np.sum(values_over_rp[0:items])
    else:
        return np.sum(values_over_rp)

def multi_unit_auction(values, rounds, rp, k):
    bidders = len(values)
    qualities = np.random.rand(bidders, rounds)
    bids = np.zeros((bidders, rounds))
    possible_bids = [[i*v/100 for i in range(100)] for v in values]

    bidder_payoffs = [np.zeros((len(possible_bids[0]), rounds))
                      for _ in range(bidders)]
    for round in range(rounds):
        for bidder in range(bidders):
            # generate payoffs for possible bids given, given the other bidders bids
            for bid_i, bid in enumerate(possible_bids[bidder]):
                this_rounds_bids = np.copy(bids[:, round])
                this_rounds_bids[bidder] = bid
                # changes start here
                winning_bids = np.argpartition(np.multiply(
                    qualities[:, round], this_rounds_bids), -k)[-k:]
                winners = []
                for i in range(len(winning_bids)):
                    if winning_bids[i] >= rp:
                        winners.append(winning_bids[i])

                # assign the payoff as the utility from the bid
                bidder_payoffs[bidder][bid_i][round] = 1 - \
                    bid if bidder in winners else 0 
            # run EW on payoffs to get bid for next round
            pi = exponential_weights(bidder_payoffs[bidder], 0.1, 1)

            # randomly pick a bid according to the probabilities generated by exponential weights
            bid = possible_bids[bidder][random_pick(pi[:, round])]
            bids[bidder][round] = bid

    return bids, qualities

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

def expected_reserve_price_from_EW(bidders, items=1, rounds=1000):
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
    reserve_prices = np.divide(list(range(0, 5)), 5/F_1_inverse(1))
    optimal_epsilon = np.sqrt(np.log(len(reserve_prices))/rounds)
    v = make_reserve_price_payoffs(bidders, F_1_inverse, reserve_prices, rounds, items)
    probabilities = exponential_weights(v, optimal_epsilon, F_1_inverse(1))
    
    expected_reserve_price = np.sum([prob*rp for prob, rp in zip(probabilities[:, rounds-1], reserve_prices)])
    return expected_reserve_price

def F_1_inverse(v):
    return 2.5 * v

def part_2():
    N = 25
    rounds_arr = [(i+1)*10 for i in range(10)]
    ITEMS = 2
    step_4_revenue = [0 for i in range(10)]
    optimal_endowed_revenue = revenue([1, 1.5, 2, 2.5], ITEMS, 0)
    step_5_revenue = [0 for i in range(10)]

    for rounds_i, rounds in enumerate(rounds_arr):
        for _ in range(N):
            values = [1, 1.5, 2, 2.5] # step 1
            inferred_vals = [0, 0, 0, 0]
            bids, qualities = multi_unit_auction(values, rounds, 0, ITEMS) #step 2, for now using k = 1, can change later
            error = 0
            for i, v in enumerate(values):
                infered_value = round(infer_value(bids, qualities, i), 3) #step 3
                inferred_vals[i] = infered_value
            # print(rounds, inferred_vals)
            step_4_revenue[rounds_i] += revenue(inferred_vals, ITEMS, 0.5) #step 4, optimal reserve price for F(v) = v auction = 0.5
        step_4_revenue[rounds_i] /= N
          
    # step 5: with tuned reserve price
    for rounds_i, rounds in enumerate(rounds_arr):
        for _ in range(N):
            values = [1, 1.5, 2, 2.5] 
            inferred_vals = [0, 0, 0, 0]
            bids, qualities = multi_unit_auction(values, rounds, expected_reserve_price_from_EW(len(values), ITEMS, 1000), ITEMS)
            for i, v in enumerate(values):
                infered_value = round(infer_value(bids, qualities, i), 3) #step 3
                inferred_vals[i] = infered_value
            # print(rounds, inferred_vals)
            step_5_revenue[rounds_i] += revenue(inferred_vals, ITEMS, 0.5)
        step_5_revenue[rounds_i] /= N

    plt.plot(rounds_arr, step_4_revenue, label='optimized parameters max revenue with inferred vals', color="Red")
    plt.plot(rounds_arr, step_5_revenue, label='new mechanism max revenue', color="Blue" )
    plt.plot(rounds_arr, [optimal_endowed_revenue for _ in rounds_arr], label='optimal endowed revenue', color="Green")
    plt.xlabel("Rounds")
    plt.ylabel("Revenue")
    plt.legend()
    plt.show()
                
part_2()