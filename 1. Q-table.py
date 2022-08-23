import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import gym

# Create a game (environment)
env = gym.make('FrozenLake-v1')

# I see Q-Table as Tensor which represents all states of the Grid World
# as its own grid matrix MAP_SIZExMAP_SIZE and for each of the states
# it would contain a vector 1xACTIONS_NUM representing the value of each
# action in a given cell
MAP_SIZE = env.observation_space.n
ACTIONS_NUM = env.action_space.n
Q = np.zeros(shape = (MAP_SIZE, ACTIONS_NUM))

print(Q[0])

# Hyper-parameters
num_episodes = 2000
num_steps = 99
lr = .8
y = .95
rList = []

for i in range(num_episodes):
    #Reset environment and get first new observation
    s = env.reset()
    rAll = 0
    d = False
    j = 0
    #The Q-Table learning algorithm
    while j < 99:
        j+=1
        #Choose an action by greedily (with noise) picking from Q table
        a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))
        #Get new state and reward from environment
        s1,r,d,_ = env.step(a)
        #Update Q-Table with new knowledge
        Q[s,a] = Q[s,a] + lr*(r + y*np.max(Q[s1,:]) - Q[s,a])
        rAll += r
        s = s1
        if d == True:
            break
    #jList.append(j)
    rList.append(rAll)

table = pd.DataFrame(Q)
print(table)