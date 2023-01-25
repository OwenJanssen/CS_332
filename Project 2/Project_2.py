import numpy as np
import matplotlib.pyplot as plt
import random

# inputs: 
# v, an array of payoffs for each action at a given turn
# epsilon, the learning rate
# h, the range of the payoffs
# outputs:
# weights, weights for each action at a given turn
# pi, optimal probabilities for picking each action at a given turn
def exponential_weights(v, epsilon, h):
    V = np.cumsum(v, axis=1)
    weights = np.power((1 + epsilon), V[:, :-1]/h)
    weights = np.insert(weights, 0, np.ones(v.shape[0]), axis=1)
    pi = np.divide(weights, np.sum(weights, axis=0))
    return weights, pi

# a
def adversarial_fair_payoffs(actions, rounds):
    V = np.zeros(shape=[actions, ])
    payoffs = np.zeros(shape=[actions, rounds])
    for i in range(rounds):
        x = np.random.uniform(0,1)
        j_star = np.argmin(V)
        payoffs[j_star][i] = x
        V[j_star] += x

    return payoffs

#b
def bernoulli_payoffs(actions, rounds):
    probabilities = list(map(lambda a: np.random.uniform(0, 0.5), np.zeros(shape=[actions, ])))
    payoffs = np.zeros(shape=[actions, rounds])
    for i in range(rounds):
        for j in range(actions):
            if np.random.uniform(0, 1) < probabilities[j]:
                payoffs[j][i] = 1
               
    return payoffs

def random_pick(probability_array):
    random_number = random.random()
    cumulative_probability = 0.0
    for index, probability in enumerate(probability_array):
        cumulative_probability += probability
        if random_number < cumulative_probability:
            return index
    return len(probability_array) - 1

def regret(payoffs, EW):
    optimal_sum = 0
    EW_sum = 0
    n = payoffs.shape[1]

    for i in range(n):
        j = np.argmax(payoffs[:, i])
        optimal_sum += payoffs[j][i]
        j_i = random_pick(EW[:, i]) # randomly pick an action according to the EW probabilities
        EW_sum += payoffs[j_i, i]

    return 1 / n * (optimal_sum-EW_sum)

def analyze_payoffs():
    N = 100
    STEP = 250
    END = 10000
    a_rounds_regret = np.zeros(int(END/STEP))
    b_rounds_regret = np.zeros(int(END/STEP))
    actions = 5
    for rounds in range(STEP, END, STEP):
        # average results from N samples
        a_sum = 0
        b_sum = 0
        epsilon = np.power(np.log(actions)/rounds, 1/2)
        for i in range(N):
            a_payoffs = adversarial_fair_payoffs(actions, rounds)
            b_payoffs = bernoulli_payoffs(actions, rounds)
            _, a_EW = exponential_weights(a_payoffs, epsilon, 1)
            _, b_EW = exponential_weights(b_payoffs, epsilon, 1)
            a_sum += regret(a_payoffs, a_EW)
            b_sum += regret(b_payoffs, b_EW)
        a_rounds_regret[int(rounds/STEP)] = a_sum/N
        b_rounds_regret[int(rounds/STEP)] = b_sum/N

    a_actions_regret = np.zeros(int(END/STEP))
    b_actions_regret = np.zeros(int(END/STEP))
    rounds = 20
    for actions in range(STEP, END, STEP):
        # average results from N samples
        a_sum = 0
        b_sum = 0
        epsilon = np.power(np.log(actions)/rounds, 1/2)
        for _ in range(N):
            a_payoffs = adversarial_fair_payoffs(actions, rounds)
            b_payoffs = bernoulli_payoffs(actions, rounds)
            _, a_EW = exponential_weights(a_payoffs, epsilon, 1)
            _, b_EW = exponential_weights(b_payoffs, epsilon, 1)
            a_sum += regret(a_payoffs, a_EW)
            b_sum += regret(b_payoffs, b_EW)
        a_actions_regret[int(actions/STEP)] = a_sum/N
        b_actions_regret[int(actions/STEP)] = b_sum/N
        
    plt.plot(list(range(0, END, STEP)), a_rounds_regret, color="red", label="Adversarial Fair Payoff")
    plt.plot(list(range(0, END, STEP)), b_rounds_regret, color="blue", label="Bernoulli Payoff")
    plt.xlabel('Rounds')
    plt.ylabel('Regret')
    plt.title('Regret vs Rounds')
    plt.legend()
    plt.show()

    plt.plot(list(range(0, END, STEP)), a_actions_regret, color="red", label="Adversarial Fair Payoff")
    plt.plot(list(range(0, END, STEP)), b_actions_regret, color="blue", label="Bernoulli Payoff")
    plt.xlabel('Actions')
    plt.ylabel('Regret')
    plt.title('Regret vs Actions')
    plt.legend()
    plt.show()
    
#c  
def data_in_the_wild():
    return
  
#d  
def adversarial_generative_model():
    # markov chains
    return

# testing
def test_example_from_class():
    weights, pi = exponential_weights(np.array([[1, 1, 0, 0], [0, 0, 1, 1]]), 1, 1)
    assert np.array_equal(weights, [[1, 2, 4, 4], [1, 1, 1, 2]])
    assert np.array_equal(pi, [[1/2, 2/3, 4/5, 2/3], [1/2, 1/3, 1/5, 1/3]])
 
if __name__ == "__main__":
    test_example_from_class()
    print("All tests passed")
    
    analyze_payoffs()