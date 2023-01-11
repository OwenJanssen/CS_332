import pandas as pd
import random 

df = pd.read_csv('bid_data.csv')

def exact_winning_prob(bid, value):

    # calculate winning prob

    bid_str = str(bid)
    bid_str = bid_str[:-1]
    bid_val = int(bid_str)

    winning_prob = bid_val / 10

    # calculate utility

    payment = []
    for i in range(10, 100, 10):
        if bid >= i:
            payment.append(i)
    
    for i in range(len(payment)):
        payment[i] = value - payment[i]
        
    expected_utility = sum(payment) / 10

    return winning_prob, expected_utility

def monte_carlo(value):

    # from the value col in excel draw a random bid 
    value_col = df['What is your bid when your value is' + value + '?']
    value_lst = value_col.tolist()
    n = 0 # weird calculation figure this out

    winning_prob_lst = []
    expected_utility_lst = []

    for i in range(n):
        bid = random.choice(value_lst)

        # winning prob calc
        bid_str = str(bid)
        bid_str = bid_str[:-1]
        bid_val = int(bid_str)

        winning_prob = bid_val / 10
        winning_prob_lst.append(winning_prob)

        # utility calc
        payment = []
        for i in range(10, 100, 10):
            if bid >= i:
                payment.append(i)
    
        for i in range(len(payment)):
            payment[i] = value - payment[i]
            

        expected_utility = sum(payment) / 10
        expected_utility_lst.append(expected_utility)

    average_winning_prob = sum(winning_prob_lst) / len(winning_prob_lst)
    average_expected_utility = sum(expected_utility_lst) / len(expected_utility_lst)
    
    # run the same winning prob/utility and add to lists

    # run this a bunch of times
    # average the results together
    return average_winning_prob, average_expected_utility