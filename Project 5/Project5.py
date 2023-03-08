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


def regret_for_z(bids, qualities, j, v, z):
    n = len(bids[0])
    x_sum = 0
    p_sum = 0
    for round in range(n):
        this_rounds_bids = np.copy(bids[:, round])
        this_rounds_bids[j] = z
        actual_winner = np.argmax(np.multiply(
            qualities[:, round], bids[:, round]))
        z_winner = np.argmax(np.multiply(
            qualities[:, round], this_rounds_bids))

        x_sum += (1 if actual_winner == j else 0) - (1 if z_winner == j else 0)
        p_sum += (bids[j, round] if actual_winner ==
                  j else 0) - (z if z_winner == j else 0)
    x_sum /= n
    p_sum /= n
    regret = v * x_sum - p_sum
    print(
        f"Regret: {regret}, z: {z}, Rationalizable: {regret >= -1 * 0.1}, x: {x_sum}, p: {p_sum}")
    # print(x_sum, p_sum)
    return regret


def infer_value(bids, qualities, j, epsilon=0.1):
    value = 0.01

    while value < 5:
        rationalizable = True
        for z in [z*value/10 for z in range(10)]:
            regret = regret_for_z(bids, qualities, j, value, z)

            if regret < -1 * epsilon:
                rationalizable = False

        if not rationalizable:
            return value - 0.01

        value += 0.01

    return value


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


part_1()
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
        return bidder_values[len(bidder_values)-1-items] * items
    else:
        return rp * bidders_over_rp 
    
def multi_unit_auction(values, rounds, rp, k):
    return 

def part_2():
    N = 5
    rounds_arr = [(i+1)*10 for i in range(10)]
    k_arr = [i for i in range(5)]
    step_4_revenue = [0 for i in range(10)]
    optimal_endowed_revenue = revenue([1, 1.5, 2, 2,5], 1, 0)
    step_5_revenue = [0 for i in range(10)]

    for rounds_i, rounds in enumerate(rounds_arr):
        for _ in range(N):
          values = [1, 1.5, 2, 2.5] # step 1
          inferred_vals = [0, 0, 0, 0]
          bids, qualities = multi_unit_auction(values, rounds, 0, 1) #step 2, for now using k = 1, can change later
          error = 0
          for i, v in enumerate(values):
              infered_value = round(infer_value(bids, qualities, i), 3) #step 3
              inferred_vals[i] = infered_value
          step_4_revenue[rounds_i] = revenue(inferred_vals, 1, 0.5) #step 4, optimal reserve price for F(v) = v auction = 0.5
    # step 5:
    for rounds_i, rounds in enumerate(rounds_arr):
        for _ in range(N):
          values = [1, 1.5, 2, 2.5] 
          inferred_vals = [0, 0, 0, 0]
          bids, qualities = multi_unit_auction(values, rounds, 0.5, 1)
          for i, v in enumerate(values):
              infered_value = round(infer_value(bids, qualities, i), 3) #step 3
              inferred_vals[i] = infered_value
          step_5_revenue[rounds_i] = revenue(inferred_vals, 1, 0.5)

    plt.plot(rounds_arr, step_4_revenue, label='optimized parameters max revenue with inferred vals', color="Red")
    plt.plot(rounds_arr, step_5_revenue, label='new mechanism max revenue', color="Blue" )
    plt.plot(rounds_arr, optimal_endowed_revenue, label='optimal endowed revenue', color="Green")
    plt.xlabel("Rounds")
    plt.ylabel("Revenue")
    plt.show()
                