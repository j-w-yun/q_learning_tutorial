import gym
from matplotlib import pyplot as plt

CROP_TOP = 34
CROP_BOTTOM = 16

env = gym.envs.make('Breakout-v0')

n_actions = env.action_space.n
action_labels = env.unwrapped.get_action_meanings()
n_observation = env.observation_space.shape
n_observation_cropped = (n_observation[0] - (CROP_TOP + CROP_BOTTOM), n_observation[1], n_observation[2])

print('Action space size: {}'.format(n_actions))
print('Action labels: {}'.format(action_labels))
print('Observation space shape: {}'.format(n_observation))
print('Cropped observation space shape: {}'.format(n_observation_cropped))

states = list()

# Take some actions and get state
initial_state = env.reset()
states.append(initial_state)
# Fire
env.step(1)
state = env.render(mode='rgb_array')
states.append(state)
# Right
for _ in range(2):
	env.step(2)
state = env.render(mode='rgb_array')
states.append(state)
# Left
for _ in range(6):
	env.step(3)
state = env.render(mode='rgb_array')
states.append(state)

# Top row showing original game screen
for i, state in enumerate(states):
	plt.subplot('2{}{}'.format(len(states), 1 + i))
	plt.title('Original {}'.format(i))
	plt.imshow(state)

# Bottom row showing cropped game screen
for i, state in enumerate(states):
	plt.subplot('2{}{}'.format(len(states), 1 + len(states) + i))
	plt.title('Cropped {}'.format(i))
	plt.imshow(state[CROP_TOP:-CROP_BOTTOM, :, :])

plt.show()
