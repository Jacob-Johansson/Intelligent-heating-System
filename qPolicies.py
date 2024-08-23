import numpy as np

# Returns the action based on the epsilon-greedy policy.
def EpsiloneGreedyPolicy(qTable, state, numActions, epsilon):
    
    # Select a random action with probability epsilon.
    if np.random.rand() < epsilon:
        action = np.random.randint(0, numActions)
    # Else select the action with the highest Q-value.
    else:
        action = np.argmax(qTable[state[0], state[1], :])
    
    return action