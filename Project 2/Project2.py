import numpy as np

# inputs: 
# v, an array of payoffs for each action at a given turn
# epsilon, the learning rate
# h, the range of the payoffs
# outputs:
# weights, weights for each action at a given turn
# pi, optimal probabilities for picking each action at a given turn
def exponential_weights(v, epsilon, h):
    V = np.empty(shape=v.shape)
    weights = np.ones(shape=v.shape)
    pi = np.empty(shape=v.shape) # probabilities
    
    for j in range(len(V)): # j is an action
        for i in range(len(V[j])): # i is a turn
            V[j][i] = np.sum(v[j][:i+1])
            
    for i in range(1, len(V[0])):
        for j in range(len(V)):
            weights[j][i] = np.power((1+epsilon), V[j][i-1]/h)
            
    for j in range(len(weights)):
        for i in range(len(weights[j])):
            pi[j][i] = weights[j][i] / np.sum(weights[:, i])
    
    return weights, pi

# testing
def test_example_from_class():
    weights, pi = exponential_weights(np.array([[1, 1, 0, 0], [0, 0, 1, 1]]), 1, 1)
    assert np.array_equal(weights, [[1, 2, 4, 4], [1, 1, 1, 2]])
    assert np.array_equal(pi, [[1/2, 2/3, 4/5, 2/3], [1/2, 1/3, 1/5, 1/3]])
 
if __name__ == "__main__":
    test_example_from_class()
    print("All tests passed")