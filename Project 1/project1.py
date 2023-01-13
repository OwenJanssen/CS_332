import pandas as pd
import random 
import numpy as np

df = pd.read_csv('Project 1/bid_data.csv')

def exact_win_prob_utility(our_bid, value):
    # calculate winning prob 
    wins = 0
    for col_header in df.head():
        value_col = df[col_header]
        for opponent_bid in value_col:
            if our_bid > float(opponent_bid):
                wins += 1
            elif our_bid == float(opponent_bid):
                # flip a coin
                if random.random() > 0.5:
                    wins += 1

    win_prob = wins/df.size

    # calculate utility
    expected_utility = (value - our_bid) * win_prob

    return win_prob, expected_utility

def evaluate_our_bids_exact(our_bids):
    bid_results = [0 for i in range(len(our_bids))]
    for i in range(len(our_bids)):
        win_prob, expected_utility = exact_win_prob_utility(our_bids[i], (i+1)*10)
        bid_results[i] = [our_bids[i], expected_utility, win_prob]

    # print("Exact our bid results")
    # print(bid_results)
    return bid_results

def optimal_bid_exact(value):
    # optimal bid calculation
    # run for int 1 to value and find expected utility then return the greatest
    expected_utilities = [0 for i in range(value)]
    win_probs = [0 for i in range(value)]
    
    for i in range(value):
        win_prob, expected_utility = exact_win_prob_utility(i, value)
        win_probs[i] = win_prob
        expected_utilities[i] = expected_utility

    optimal_bid = np.argmax(expected_utilities)
    return optimal_bid, expected_utilities[optimal_bid], win_probs[optimal_bid]

def evaluate_optimal_bids_exact():
    values = [10*(i+1) for i in range(10)]
    value_results = [0 for i in range(10)]
    for i in range(len(values)):
        value_results[i] = optimal_bid_exact(values[i])

    # print("Exact optimal bid result")
    # print(value_results)
    return value_results


## Functions for evaluating bids using all/exact data
def monte_carlo_win_prob_utility(our_bid, value):
    alpha = 0.05
    epsilon = 0.15
    n = int((np.log(2 / alpha)) / (2 * (pow(epsilon, 2))))

    # winning prob calc
    wins = 0
    df_as_matrix = df.to_numpy()
    df_flat = df_as_matrix.flatten()

    for i in range(n):
        # draw a random bid
        opponent_bid = np.random.choice(df_flat) 
        if our_bid > float(opponent_bid):
            wins += 1
        elif our_bid == float(opponent_bid):
            # flip a coin
            if random.random() > 0.5:
                wins += 1

    win_prob = wins/n
        
    # calculate utility
    expected_utility = (value - our_bid) * win_prob

    return win_prob, expected_utility
   
def evaluate_our_bids_monte_carlo(our_bids):
    bid_results = [0 for i in range(len(our_bids))]
    for i in range(len(our_bids)):
        win_prob, expected_utility = monte_carlo_win_prob_utility(our_bids[i], (i+1)*10)
        bid_results[i] = [our_bids[i], expected_utility, win_prob]

    # print("MC our bid results")
    # print(bid_results)
    return bid_results

def optimal_bid_monte_carlo(value):
    expected_utilities = [0 for i in range(value)]
    win_probs = [0 for i in range(value)]
    
    for i in range(value):
        win_prob, expected_utility = monte_carlo_win_prob_utility(i, value)
        win_probs[i] = win_prob
        expected_utilities[i] = expected_utility

    optimal_bid = np.argmax(expected_utilities)
    return optimal_bid, expected_utilities[optimal_bid], win_probs[optimal_bid]

def evaluate_optimal_bids_monte_carlo():
    values = [10*(i+1) for i in range(10)]
    value_results = [0 for i in range(10)]
    for i in range(len(values)):
        value_results[i] = optimal_bid_monte_carlo(values[i])

    # print("MC optimal bid results")
    # print(value_results)
    return value_results

def column(matrix, i):
    return [row[i] for row in matrix]

def part1():
    our_bids = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    our_exact = evaluate_our_bids_exact(our_bids)
    opt_exact = evaluate_optimal_bids_exact()
    our_mc = evaluate_our_bids_monte_carlo(our_bids)
    opt_mc = evaluate_optimal_bids_monte_carlo()

    output = {'value': [10*(i+1) for i in range(10)], 
            'Our Bids Exact: Bid': column(our_exact, 0),
            'Our Bids Exact: E[U]': column(our_exact, 1),
            'Our Bids Exact: Pr[W]': column(our_exact, 2),
            '': ['' for i in range(10)],
            'Optimal Bids Exact: Bid': column(opt_exact, 0),
            'Optimal Bids Exact: E[U]': column(opt_exact, 1),
            'Optimal Bids Exact: Pr[W]': column(opt_exact, 2),
            ' ': ['' for i in range(10)],
            'Our Bids MC: Bid': column(our_mc, 0),
            'Our Bids MC: E[U]': column(our_mc, 1),
            'Our Bids MC: Pr[W]': column(our_mc, 2),
            '  ': ['' for i in range(10)],
            'Optimal Bids MC: Bid': column(opt_mc, 0),
            'Optimal Bids MC: E[U]': column(opt_mc, 1),
            'Optimal Bids MC: Pr[W]': column(opt_mc, 2)}

    output_df = pd.DataFrame(output)
    output_df.to_csv('Project 1/bid_data_results_new.csv', index=False)

def find_optimal_bids(data):
    values = [10*(i+1) for i in range(10)]
    optimal_bids = [0 for i in range(10)]
    
    for i in range(len(values)):
        # use data to figure out a value somehow
        # could just be to do what we did for part 1
        optimal_bids[i] = 0 # bid

    return optimal_bids

def compare_bids(a, b):
    # maybe return average difference between expected utilities?
    return []

def part2():
    exact_optimal_bids = [6, 12, 12, 21, 22, 31, 32, 41, 41, 52]
    results = [0 for i in range(df.shape[0])]

    for i in range(len(results)):
        # get random subset of data of size i
        subset = df.sample(n = i) 
        subset_optimal_bids = find_optimal_bids(subset)
        results[i] = compare_bids(exact_optimal_bids, subset_optimal_bids)

    # plot results vs i
    # determine best fit line
    # we should expect to see the results getting lower as i increases