import numpy as np
import matplotlib.pyplot as plt

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

# a
def adversarial_fair_payoffs(actions, rounds):
    V = np.empty(shape=[actions, ])
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

def regret(payoffs, EW):
    optimal_sum = 0
    EW_sum = 0
    n = payoffs.shape[1]

    for i in range(n):
        j = np.argmax(payoffs[:, i])
        optimal_sum += payoffs[j][i]
        j_i = np.argmax(EW[1][:, i])
        EW_sum += payoffs[j_i, i]

    return 1 / n * (optimal_sum-EW_sum)

def analyze_payoffs():
    # a_rounds_regret = np.empty(100)
    # b_rounds_regret = np.empty(100)
    # actions = 5
    # for rounds in range(0, 100):
    #     epsilon = np.power(np.log(actions)/rounds, 1/2)
    #     a_payoffs = adversarial_fair_payoffs(actions, rounds)
    #     b_payoffs = bernoulli_payoffs(actions, rounds)
    #     a_EW = exponential_weights(a_payoffs, epsilon, 1)
    #     b_EW = exponential_weights(b_payoffs, epsilon, 1)
    #     a_rounds_regret[rounds] = regret(a_payoffs, a_EW)
    #     b_rounds_regret[rounds] = regret(b_payoffs, b_EW)
        
    # plt.plot(list(range(1, 101)), a_rounds_regret)
    # plt.xlabel('Rounds')
    # plt.ylabel('Regret')
    # plt.title('Adversarial Fair Payoff: Regret vs Rounds')
    # plt.show()

    # plt.plot(list(range(1, 101)), b_rounds_regret)
    # plt.xlabel('Rounds')
    # plt.ylabel('Regret')
    # plt.title('Bernoulli Payoffs: Regret vs Rounds')
    # plt.show()

    a_actions_regret = np.empty(100)
    b_actions_regret = np.empty(100)
    rounds = 20
    for actions in range(1, 100):
        a_sum = 0
        b_sum = 0
        for _ in range(10):
            epsilon = np.power(np.log(actions)/rounds, 1/2)
            a_payoffs = adversarial_fair_payoffs(actions, rounds)
            b_payoffs = bernoulli_payoffs(actions, rounds)
            a_EW = exponential_weights(a_payoffs, epsilon, 1)
            b_EW = exponential_weights(b_payoffs, epsilon, 1)
            a_sum += regret(a_payoffs, a_EW)
            b_sum += regret(b_payoffs, b_EW)
        a_actions_regret[actions] = a_sum/10
        b_actions_regret[actions] = b_sum/10

    plt.plot(list(range(1, 101)), a_actions_regret)
    plt.xlabel('Actions')
    plt.ylabel('Regret')
    plt.title('Adversarial Fair Payoff: Regret vs Actions')
    plt.show()

    plt.plot(list(range(1, 101)), b_actions_regret)
    plt.xlabel('Actions')
    plt.ylabel('Regret')
    plt.title('Bernoulli Payoffs: Regret vs Actions')
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
    # test_example_from_class()
    # print("All tests passed")
    
    analyze_payoffs()

    # a = adversarial_fair_payoffs(5,2)
    # print(exponential_weights(a, 1, 1)[1])
    # print(a)
    # print(bernoulli_payoffs(5,2))