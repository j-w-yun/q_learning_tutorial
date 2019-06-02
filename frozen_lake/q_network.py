import gym
import numpy as np
import tensorflow as tf


VERBOSE = False

# Load environment
environment = gym.make('FrozenLake-v0', map_name='4x4', is_slippery=False)
actions = ['LEFT', 'DOWN', 'RIGHT', 'UP']

print(environment.render())
print('Observation space: {}'.format(environment.observation_space.n))
print(end='\n')
print(actions)
print('Action space: {}'.format(environment.action_space.n))
print('-' * 80)
print(end='\n')

encoded_state = list()
for line in environment.desc.tolist():
	for elem in line:
		if 'S' in str(elem):
			encoded_state.append([1, 0, 0, 0, 0])
		elif 'F' in str(elem):
			encoded_state.append([0, 1, 0, 0, 0])
		elif 'H' in str(elem):
			encoded_state.append([0, 0, 1, 0, 0])
		elif 'G' in str(elem):
			encoded_state.append([0, 0, 0, 1, 0])
encoded_state = np.expand_dims(np.asarray(encoded_state), 0)

# Model and training parameters
input_dim = [None, 16, 5]
output_dim = [None, 4]
learning_rate = 0.0001
discount_factor = 0.5
initial_epsilon = 1.0
n_episodes = 10000
max_steps = 50
ma_window = n_episodes // 100

# Build model
inputs = tf.placeholder(shape=input_dim, dtype=tf.float32)

inputs_2d = tf.reshape(inputs, shape=[-1, 16, 1, 5])
k_1 = tf.get_variable('k_1', shape=[1, 1, 5, 1], initializer=tf.truncated_normal_initializer(stddev=0.01))
conv_1 = tf.nn.conv2d(inputs_2d, k_1, strides=[1, 1, 1, 1], padding='VALID')
conv_out_1 = tf.reshape(conv_1, shape=[-1, 16])

w_1 = tf.get_variable('w_1', shape=[16, 4], initializer=tf.truncated_normal_initializer(stddev=0.01))
b_1 = tf.get_variable('b_1', shape=[4], initializer=tf.truncated_normal_initializer(stddev=0.01))
outputs = tf.nn.xw_plus_b(conv_out_1, w_1, b_1)
# outputs = tf.nn.leaky_relu(outputs)
outputs = tf.nn.sigmoid(outputs)

# with tf.Session() as sess:
# 	sess.run(tf.initialize_all_variables())
# 	r = sess.run(
# 		outputs,
# 		feed_dict={
# 			inputs: encoded_state
# 		}
# 	)
# 	print(r)
# 	print(r.shape)
# exit(0)

# Model train step
next_outputs = tf.placeholder(shape=output_dim, dtype=tf.float32)
loss = tf.reduce_sum(tf.square(next_outputs - outputs))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_step = optimizer.minimize(loss)

# Misc operations
policy_sum = tf.reduce_sum(tf.squeeze(outputs))

# Train the network
reward_list = list()
steps_list = list()
epsilon = initial_epsilon
with tf.Session() as sess:
	# Initialize model variables
	sess.run(tf.global_variables_initializer())

	# Get learned policy
	inputs_val = list()
	for i in range(16):
		s = encoded_state.copy()
		s[0, i] = [0, 0, 0, 0, 1]
		inputs_val.append(s)
	inputs_val = np.concatenate(inputs_val, 0)
	outputs_val = sess.run(
		outputs,
		feed_dict={
			inputs: inputs_val
		}
	)
	actions_val = np.argmax(outputs_val, axis=1)

	print('Learned policy:')
	row, col = 0, 0
	for i, action_val in enumerate(actions_val):
		if 'H' in str(environment.desc.tolist()[row][col]):
			print('{: <8}'.format('N/A'), end='\t')
		else:
			print('{: <8}'.format(actions[action_val]), end='\t')
		if (i + 1) % np.sqrt(environment.observation_space.n) == 0:
			print(end='\n')
		col += 1
		if col == 4:
			row += 1
			col = 0
	print('-' * 80)
	print(end='\n')

	print('Learned policy (raw):')
	for i, output_val in enumerate(outputs_val):
		for elem in output_val:
			print('{0:.4f}'.format(elem), end='\t')
		print(end='\n')
	print('-' * 80)
	print(end='\n')

	for episode in range(n_episodes):

		# Reset environment and get first observation
		state = environment.reset()
		episode_reward = 0
		episode_steps = 0

		# Continue episode until game is over or maximum step limit is reached
		for step in range(max_steps):
			# Prepare encoded inputs for current step
			inputs_val = encoded_state.copy()
			inputs_val[0, state] = [0, 0, 0, 0, 1]

			# Get outputs
			outputs_val = sess.run(
				outputs,
				feed_dict={
					inputs: inputs_val
				}
			)

			actions_val = np.argmax(outputs_val, axis=1)

			# print(inputs_val)
			# print(outputs_val)
			# print(actions_val)
			# print('-' * 80)

			# Explore with probability epsilon
			if np.random.rand(1) < epsilon:
				actions_val[0] = environment.action_space.sample()

			# Make a step in the environment for the current episode
			next_state, reward, is_done, _ = environment.step(actions_val[0])

			# Prepare encoded inputs for next step
			next_inputs_val = encoded_state.copy()
			next_inputs_val[0, next_state] = [0, 0, 0, 0, 1]

			# Get next outputs
			next_outputs_val = sess.run(
				outputs,
				feed_dict={
					inputs: next_inputs_val
				}
			)

			# print(next_inputs_val)
			# print(next_outputs_val)
			# exit(0)

			# Calculate Q-values
			max_next_outputs_val = np.max(next_outputs_val)
			target_next_outputs_val = outputs_val
			target_next_outputs_val[0, actions_val[0]] = reward + discount_factor * max_next_outputs_val

			# Run train step
			sess.run(
				train_step,
				feed_dict={
					inputs: inputs_val,
					next_outputs: target_next_outputs_val
				}
			)

			if VERBOSE:
				print('Episode: {}/{} Step: {}/{}'.format(episode, n_episodes, step, max_steps))
				print('Current state: {}'.format(state))
				print('Action: {}'.format(actions[actions_val[0]]))
				print('Reward: {}'.format(reward))
				print('Next state: {}'.format(next_state))
				print('Game over: {}'.format(is_done))
				print(end='\n')

			episode_reward += reward
			episode_steps = step + 1
			state = next_state

			if is_done:
				# Reduce probability of exploration on next episode
				epsilon = initial_epsilon * (1 - (episode / n_episodes))
				break

		reward_list.append(episode_reward)
		steps_list.append(episode_steps)

		if episode % (n_episodes / 100) == 0:
			print('Progress: {0:.2f}% complete'.format(100 * episode / n_episodes))
			print('Episode: {}/{}'.format(episode, n_episodes))
			print('Epsilon: {0:.4f}'.format(epsilon))
			print('Moving window: {}'.format(ma_window))
			print('Average rewards: {0:.2f}'.format(sum(reward_list[-ma_window:]) / ma_window))
			print('Average steps: {0:.2f}'.format(sum(steps_list[-ma_window:]) / ma_window))
			print('-' * 80)
			print(end='\n')

	# Get learned policy
	inputs_val = list()
	for i in range(16):
		s = encoded_state.copy()
		s[0, i] = [0, 0, 0, 0, 1]
		inputs_val.append(s)
	inputs_val = np.concatenate(inputs_val, 0)
	outputs_val = sess.run(
		outputs,
		feed_dict={
			inputs: inputs_val
		}
	)
	actions_val = np.argmax(outputs_val, axis=1)

	print('Learned policy:')
	row, col = 0, 0
	for i, action_val in enumerate(actions_val):
		if 'H' in str(environment.desc.tolist()[row][col]):
			print('{: <8}'.format('N/A'), end='\t')
		else:
			print('{: <8}'.format(actions[action_val]), end='\t')
		if (i + 1) % np.sqrt(environment.observation_space.n) == 0:
			print(end='\n')
		col += 1
		if col == 4:
			row += 1
			col = 0
	print('-' * 80)
	print(end='\n')

	print('Learned policy (raw):')
	for i, output_val in enumerate(outputs_val):
		for elem in output_val:
			print('{0:.4f}'.format(elem), end='\t')
		print(end='\n')
	print('-' * 80)
	print(end='\n')
