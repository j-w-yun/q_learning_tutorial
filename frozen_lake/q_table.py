import gym
import numpy as np

VERBOSE = False

# Load environment
environment = gym.make('FrozenLake-v0', map_name='4x4', is_slippery=True)
actions = ['LEFT', 'DOWN', 'RIGHT', 'UP']

print(environment.render())
print('Observation space: {}'.format(environment.observation_space.n))
print(end='\n')
print(actions)
print('Action space: {}'.format(environment.action_space.n))
print('-' * 80)
print(end='\n')

# Initialize table with all zeros
q_table = np.zeros([environment.observation_space.n, environment.action_space.n])

# Set learning params
learning_rate = 0.005
discount_factor = 0.95
n_episodes = 100000
max_steps = 100
ma_window = 100

print('Initial Q-table:')
for row in q_table:
    for elem in row:
        print('{0:.15f}'.format(elem), end='\t')
    print(end='\n')
print('-' * 80)
print(end='\n')

# Create lists to contain total rewards and steps per episode
reward_list = list()
steps_list = list()
for episode in range(n_episodes):

    # Reset environment and get first observation
    state = environment.reset()
    episode_reward = 0
    episode_steps = 0

    # Continue episode until game is over or maximum step limit is reached
    for step in range(max_steps):

        # Greedy policy + noise
        action = np.argmax(q_table[state, :] + np.random.randn(1, environment.action_space.n) * (1 / (episode + 1)))

        # Make a step in the environment for the current episode
        next_state, reward, is_done, _ = environment.step(action)

        if VERBOSE:
            print('Episode: {} Step: {}'.format(episode, step))
            print('Current state: {}'.format(state))
            print('Action: {}'.format(actions[action]))
            print('Reward: {}'.format(reward))
            print('Next state: {}'.format(next_state))
            print('Game over: {}'.format(is_done))
            print('Current Q-table:')
            for row in q_table:
                for elem in row:
                    print('{0:.15f}'.format(elem), end='\t')
                print(end='\n')
            print('-' * 10)
            print(end='\n')

        # Update Q-table
        delta = learning_rate * (reward + discount_factor * np.max(q_table[next_state, :]) - q_table[state, action])
        q_table[state, action] += delta

        episode_reward += reward
        episode_steps = step + 1
        state = next_state

        # Game over
        if is_done:
            break

    reward_list.append(episode_reward)
    steps_list.append(episode_steps)

    if (episode + 1) % (n_episodes / 100) == 0:
        print('Episode: {} complete'.format(episode))
        print('Average of {} latest rewards: {}'.format(ma_window, sum(reward_list[-ma_window:]) / ma_window))
        print('Average of {} latest steps: {}'.format(ma_window, sum(steps_list[-ma_window:]) / ma_window))
        print('-' * 80)
        print(end='\n')

print('Final Q-table:')
for row in q_table:
    for elem in row:
        print('{0:.15f}'.format(elem), end='\t')
    print(end='\n')
print('-' * 80)
print(end='\n')

print('Learned policy:')
for i, row in enumerate(q_table):
    if np.sum(row) == 0:
        learned_best_action = 'N/A'
    else:
        learned_best_action = actions[np.argmax(row)]
    print('{: <5}'.format(learned_best_action), end='\t')
    if (i + 1) % np.sqrt(environment.observation_space.n) == 0:
        print(end='\n')
print('-' * 80)
print(end='\n')
