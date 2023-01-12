import pandas as pd
import random 
import numpy as np

df = pd.read_csv('bid_data.csv')
our_bids = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
optimal_bids = [6, 12, 12, 21, 22, 31, 32, 41, 41, 52]

def exact_winning_prob_utility(our_bid, value):
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

    winning_prob = wins/df.size

    # calculate utility
    utility = value - our_bid
    print(value)
    print(our_bid)
    print(utility)

    return winning_prob, utility

def evaluate_our_bids_exact():
    bid_results = [0 for i in range(len(optimal_bids))]
    for i in range(len(optimal_bids)):
        bid_results[i] = exact_winning_prob_utility(optimal_bids[i], (i+1)*10)
    print("Exact our bid results")
    print(bid_results)

def exact_optimal_bid(value):
    # optimal bid calculation
    # run for int 1 to value and find expected utility then return the greatest
    expected_values = [0 for i in range(value)]
    
    for i in range(value):
        prob_and_utility = exact_winning_prob_utility(i, value)
        expected_values[i] = prob_and_utility[0] * prob_and_utility[1]

    best_value = np.argmax(expected_values)
    return best_value, expected_values[best_value]

def evaluate_optimal_bids_exact():
    values = [10*(i+1) for i in range(10)]
    value_results = [0 for i in range(10)]
    for i in range(len(values)):
        value_results[i] = (exact_optimal_bid(values[i]))

    print("Exact optimal bid result")
    print(value_results)

def monte_carlo_winning_prob_utility(our_bid, value):
    # from the value col in excel draw a random bid 
    
    alpha = 0.05
    epsilon = 0.15
    n = int((np.log(2 / alpha)) / (2 * (pow(epsilon, 2))))

    # winning prob calc
    wins = 0
    df_as_matrix = df.to_numpy()
    df_flat = df_as_matrix.flatten()

    for i in range(n):
        opponent_bid = np.random.choice(df_flat) 
        if our_bid > float(opponent_bid):
            wins += 1
        elif our_bid == float(opponent_bid):
            # flip a coin
            if random.random() > 0.5:
                wins += 1

    winning_prob = wins/n
        
    # calculate utility
    utility = value-our_bid

    return winning_prob, utility
   
def evaluate_our_bids_monte_carlo():
    bid_results = [0 for i in range(len(our_bids))]
    for i in range(len(our_bids)):
        bid_results[i] = monte_carlo_winning_prob_utility(our_bids[i], (i+1)*10)
    print("MC our bid results")
    print(bid_results)
    
def optimal_bid_monte_carlo(value):
    expected_values = [0 for i in range(value)]
    
    for i in range(value):
        prob_and_utility = monte_carlo_winning_prob_utility(i, value)
        expected_values[i] = prob_and_utility[0] * prob_and_utility[1]

    best_value = np.argmax(expected_values)
    return best_value, expected_values[best_value]

def evaluate_optimal_bids_monte_carlo():
    values = [10*(i+1) for i in range(10)]
    value_results = [0 for i in range(10)]
    for i in range(len(values)):
        value_results[i] = (optimal_bid_monte_carlo(values[i]))

    print("MC optimal bid results")
    print(value_results)
    

evaluate_our_bids_exact()
#evaluate_optimal_bids_exact()
#evaluate_our_bids_monte_carlo()
#evaluate_optimal_bids_monte_carlo()