import numpy as np
import nashpy as nash
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
    pi = np.divide(weights, np.sum(weights, axis=0))
    return weights, pi

# returns actual Nash eq given 2 2x2 arrays
def bimatrix_game(A, B):
    game = nash.Game(A, B)
    eq = game.support_enumeration()
    return eq #returns optimal pick for each player 

def generate_random_game():
    A = np.random.rand(2, 2)
    B = np.random.rand(2, 2)
    return A, B

def test_exponential_weights(A, B):
    ROUNDS = 25
    EPSILON = 1
    payoffs_A = np.empty([2, ROUNDS])
    payoffs_B = np.empty([2, ROUNDS])
    for round in range(ROUNDS):
        _, pi_A = exponential_weights(payoffs_A[:, 0:round+1], EPSILON, 1)
        _, pi_B = exponential_weights(payoffs_B[:, 0:round+1], EPSILON, 1)
        action_A = random_pick(pi_A[:, round])
        action_B = random_pick(pi_B[:, round])
        payoffs_A[action_A][round] = A[action_A][action_B]
        payoffs_B[action_B][round] = B[action_A][action_B]
    _, pi_A = exponential_weights(payoffs_A, EPSILON, 1)
    _, pi_B = exponential_weights(payoffs_B, EPSILON, 1)
    return pi_A, pi_B

def random_pick(probability_array):
    random_number = np.random.random()
    cumulative_probability = 0.0
    for index, probability in enumerate(probability_array):
        cumulative_probability += probability
        if random_number < cumulative_probability:
            return index
    return len(probability_array) - 1

A, B = generate_random_game()
pi_A, pi_B = test_exponential_weights(A, B)
eq = bimatrix_game(A, B)
print(A, B)
for i in eq:
    print(i)
print(pi_A[:, 24], pi_B[:, 24])