import numpy as np
import matplotlib.pyplot as plt
import time
import gym

# Create a game (environment)
env = gym.make('FrozenLake-v1')

# I see Q-Table as Tensor which represents all states of the Grid World
# as its own grid matrix MAP_SIZExMAP_SIZE and for each of the states
# it would contain a vector 1xACTIONS_NUM representing the value of each
# action in a given cell
MAP_SIZE = 4
ACTIONS_NUM = 4
Q_Table = np.zeros(shape = (MAP_SIZE, MAP_SIZE, ACTIONS_NUM))

print(Q_Table)

# Hyper-parameters
number_of_episods = 2000
number_of_steps = 25
learning_rate = .8
gamma = .95

for i in range(number_of_episods):
    previous_state = env.reset()
    action = 0

    for j in range(number_of_steps):

        past_col = previous_state % MAP_SIZE
        past_row = previous_state - (int(previous_state / MAP_SIZE) * MAP_SIZE)

        noise = np.random.randn(1, ACTIONS_NUM)
        decrease_explanatory = 1./(i+1)
        noise_for_actions = noise * decrease_explanatory

        q_values_for_state = Q_Table[past_row][past_col][:]
        action = np.argmax(q_values_for_state + noise_for_actions)

        print("noise_for_actions: ", noise_for_actions, "q_values_for_state: ", q_values_for_state, "action: ", action)

        current_state_number, reward, done, _ = env.step(action)  # apply it

        col = current_state_number % MAP_SIZE
        row = current_state_number - (int(current_state_number/MAP_SIZE) * MAP_SIZE)

        Q_Table[row][col][action] = Q_Table[row][col][action] + learning_rate*(reward
                                    + gamma*np.max(Q_Table[row][col][:]) - Q_Table[past_row][past_col][action])

        previous_state = current_state_number

        # Terminal point
        if done:
            break

print(Q_Table)

# env.reset()
# while True:
#     print("hi")
#     time.sleep(3)