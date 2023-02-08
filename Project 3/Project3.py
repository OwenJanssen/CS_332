import numpy as np
import nashpy as nash
import math
import matplotlib.pyplot as plt

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
    # because we're referencing the previous column, the first
    # column of ones gets removed so we need to add it back
    weights = np.insert(weights, 0, np.ones(v.shape[0]), axis=1)
    # print(f"SUM:{np.sum(weights, axis=0)}" )
    pi = np.divide(weights, np.sum(weights, axis=0))
    return weights, pi

# returns actual Nash eq given 2 2x2 arrays
def bimatrix_game(A, B):
    game = nash.Game(A, B)
    eq = game.support_enumeration()
    return eq #returns optimal pick for each player 

# generates a random bi-matrix game
def generate_random_game():
    A = np.random.randint(0, 10, size=(2, 2))
    B = np.random.randint(0, 10, size=(2, 2))
    return A, B

# EW tests that return probabilities for all actions/rounds 
def test_exponential_weights(A, B, epsilon=0.1):
    ROUNDS = 25
    payoffs_A = np.zeros([2, ROUNDS])
    payoffs_B = np.zeros([2, ROUNDS])
    for round in range(ROUNDS):
        _, pi_A = exponential_weights(payoffs_A[:, 0:round+1], epsilon, 1)
        _, pi_B = exponential_weights(payoffs_B[:, 0:round+1], epsilon, 1)
        action_A = random_pick(pi_A[:, round])
        action_B = random_pick(pi_B[:, round])
        payoffs_A[0][round] = A[0][action_B]
        payoffs_A[1][round] = A[1][action_B]
        # print(f"B {B[action_A][action_B]}")
        payoffs_B[0][round] = B[action_A][0]
        payoffs_B[1][round] = B[action_A][1]
        # print(f"B Payoffs {payoffs_B}")
    _, pi_A = exponential_weights(payoffs_A, epsilon, 1)
    _, pi_B = exponential_weights(payoffs_B, epsilon, 1)
    return pi_A, pi_B

# randomly pick an index in the array according to the given probabilities in each element of the array
def random_pick(probability_array):
    random_number = np.random.random()
    cumulative_probability = 0.0
    for index, probability in enumerate(probability_array):
        cumulative_probability += probability
        if random_number < cumulative_probability:
            return index
    return len(probability_array) - 1

# Monte Carlo tests to determine if EW converges to the nash equilibrium
def monte_carlo_EW(epsilon):
    N = 200
    converges_to_nash_count = 0
    for i in range(N):
        A, B = generate_random_game()
        pi_A, pi_B = test_exponential_weights(A, B, epsilon)
        eq = bimatrix_game(A, B)
        for i in eq:
            # test if the results from exponential weights are within a reasonable threshold of the nash equilibrium
            a_true_nash = i[0]
            a_results = pi_A[:, 24]
            a_converges = (abs(a_true_nash[0] - np.round(a_results[0])) < 0.1) & (abs(a_true_nash[1] - np.round(a_results[1])) < 0.1)
            b_true_nash = i[1]
            b_results = pi_B[:, 24]
            b_converges = (abs(b_true_nash[0] - np.round(b_results[0])) < 0.1) & (abs(b_true_nash[1] - np.round(b_results[1])) < 0.1)            
            if a_converges & b_converges:
                converges_to_nash_count += 1
        # print(pi_A[:, 24], pi_B[:, 24])
    # print (converges_to_nash_count/N)
    return converges_to_nash_count/N

#strategy: look at the nash eq --> always pick the one where we have the highest payoff
def beat_learning_alg(A, B):
    game = nash.game(A,B)
    eq = game.support_enumeration()
    if eq[1][0] == 1:
        col = 0 
    else:
        col = 1

    play = max(B[0][col], B[1][col])
    return play 

def test_learning_rates_for_EW():
    epsilons = [i/10 for i in range(500)]
    results = [0 for i in range(len(epsilons))]
    for i, e in enumerate(epsilons):
        results[i] = monte_carlo_EW(e)
    plt.plot(epsilons, results)
    plt.xlabel('Epsilon - Learning Rate')
    plt.ylabel('Convergence to Nash Rate')
    plt.show()
    
test_learning_rates_for_EW()