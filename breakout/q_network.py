import os
from abc import ABC, abstractmethod
from collections import namedtuple

import gym
import numpy as np
import tensorflow as tf
from gym import wrappers
from matplotlib import pyplot as plt


USE_CPU = False
INPUT_SHAPE = [210, 160, 3]
CROP_TOP = 34
CROP_BOTTOM = 16
PROCESSED_SHAPE = [84, 84]


class ImageProcessor:
	"""
	Image processor for converting raw screen image into cropped, rescaled, and
	grayscaled image.
	"""

	def __init__(self, scope='image_processor'):
		self.scope = scope
		self._build(scope)

	def _build(self, scope):
		with tf.variable_scope(scope):
			self.input = tf.placeholder(shape=INPUT_SHAPE, dtype=tf.uint8, name='input')
			self.output = tf.image.rgb_to_grayscale(self.input)
			self.output = tf.image.crop_to_bounding_box(
				self.output,
				offset_height=CROP_TOP,
				offset_width=0,
				target_height=INPUT_SHAPE[0] - (CROP_TOP + CROP_BOTTOM),
				target_width=INPUT_SHAPE[1]
			)
			self.output = tf.image.resize_images(
				self.output,
				size=PROCESSED_SHAPE,
				method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
			)
			self.output = tf.squeeze(self.output)

	def run(self, sess, input_image):
		"""
		Processes raw Atari image into cropped, rescaled, and grayscaled image.

		Args:
			sess: TensorFlow session object
			input_image: An image array of shape (210, 160, 3)

		Returns:
			A processed image of shape (84, 84) ranging in value 0 through 255,
			inclusive.
		"""
		retval = sess.run(
			self.output,
			feed_dict={
				self.input: input_image
			}
		)
		return retval


class Estimator:
	"""
	Neural network used to estimate Q-value for reinforcement learning.

	This network is used for both the Q-network and the target Q-network,
	as described in "Human-level control through deep reinforcement learning."

	https://deepmind.com/research/publications/human-level-control-through-deep-reinforcement-learning/
	"""

	def __init__(self, n_actions, scope='estimator'):
		self.n_actions = n_actions
		self.scope = scope
		self._build(scope)

	def _build(self, scope):
		with tf.variable_scope(scope):
			self.state = tf.placeholder(shape=[None, PROCESSED_SHAPE[0], PROCESSED_SHAPE[1], 4], dtype=tf.uint8, name='state')
			self.action = tf.placeholder(shape=[None], dtype=tf.int32, name='action')
			self.target = tf.placeholder(shape=[None], dtype=tf.float32, name='target')

			input_scaled = tf.cast(self.state, dtype=tf.float32) / 255.0
			batch_size = tf.shape(self.state)[0]

			conv_1 = tf.contrib.layers.conv2d(
				input_scaled,
				num_outputs=32,
				kernel_size=8,
				stride=4,
				padding='SAME',
				activation_fn=tf.nn.relu
			)
			conv_2 = tf.contrib.layers.conv2d(
				conv_1,
				num_outputs=64,
				kernel_size=4,
				stride=2,
				padding='SAME',
				activation_fn=tf.nn.relu
			)
			conv_3 = tf.contrib.layers.conv2d(
				conv_2,
				num_outputs=64,
				kernel_size=3,
				stride=1,
				padding='SAME',
				activation_fn=tf.nn.relu
			)
			conv_3_flat = tf.contrib.layers.flatten(conv_3)
			fc_1 = tf.contrib.layers.fully_connected(conv_3_flat, 512)
			self.predictions = tf.contrib.layers.fully_connected(fc_1, self.n_actions)

			# Get value of selected action
			gather_indices = tf.range(batch_size) * self.n_actions + self.action
			self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)

			# Learn target value of selected action
			self.losses = tf.squared_difference(self.target, self.action_predictions)
			self.loss = tf.reduce_mean(self.losses)

			# Train step
			self.optimizer = tf.train.RMSPropOptimizer(
				learning_rate=0.00025,
				decay=0.99,
				momentum=0.0,
				epsilon=1e-6
			)
			self.train_step = self.optimizer.minimize(self.loss, global_step=tf.train.get_global_step())

		with tf.name_scope(scope):
			self.summaries = tf.summary.merge([
				tf.summary.scalar('loss', self.loss),
				tf.summary.scalar('avg_q_value', tf.reduce_mean(self.predictions)),
				tf.summary.scalar('max_q_value', tf.reduce_max(self.predictions)),
				tf.summary.histogram('losses_histogram', self.losses),
				tf.summary.histogram('q_values_histogram', self.predictions)
			])

	def run(self, sess, state):
		"""
		Predicts action values.

		Args:
			sess: TensorFlow session object
			state: Processed image of shape (batch_size, 84, 84, 4)

		Returns:
			Tensor of shape (batch_size, n_actions) containing the estimated
			action values.
		"""
		retval = sess.run(
			self.predictions,
			feed_dict={
				self.state: state
			}
		)
		return retval

	def update(self, sess, state, action, target):
		"""
		Updates the estimator towards the given targets.

		Args:
			sess: TensorFlow session object
			state: Processed image of shape (batch_size 84, 84, 4)
			action: Chosen actions of shape (batch_size)
			target: Target value for chosen actions of shape (batch_size)

		Returns:
			The calculated loss on the batch.
		"""
		summaries, global_step, _, loss = sess.run(
			[self.summaries, tf.train.get_global_step(), self.train_step, self.loss],
			feed_dict={
				self.state: state,
				self.action: action,
				self.target: target
			}
		)
		return summaries, global_step, loss


class NetworkCopier:
	"""
	Copy model parameters from one to another.
	"""

	def __init__(self, origin_model, target_model, scope='network_copier'):
		"""
		Create TensorFlow graph for copying model parameters.

		Args:
			origin_model: Network from which to copy the parameters
			target_model: Network in which to paste the parameters
		"""
		self.origin_params = [t for t in tf.trainable_variables() if t.name.startswith(origin_model.scope)]
		self.origin_params = sorted(self.origin_params, key=lambda v: v.name)
		self.target_params = [t for t in tf.trainable_variables() if t.name.startswith(target_model.scope)]
		self.target_params = sorted(self.target_params, key=lambda v: v.name)
		self._build(scope)

	def _build(self, scope):
		with tf.variable_scope(scope):
			self.update_ops = []
			for origin_variable, target_variable in zip(self.origin_params, self.target_params):
				op = target_variable.assign(origin_variable)
				self.update_ops.append(op)

	def run(self, sess):
		"""
		Copies the model parameters from origin to target networks.

		Args:
			sess: TensorFlow session object
		"""
		sess.run(self.update_ops)


def make_epsilon_greedy_policy(estimator, n_actions):
	"""
	Creates an epsilon-greedy policy based on a given Q-function approximator
	and epsilon.

	Args:
		estimator: Estimator that returns Q-values for a given state
		n_actions: Number of actions in the environment

	Returns:
		A function that takes the (sess, observation, epsilon) as an argument
		and returns the probabilities for each action in the form of a numpy
		array of length `n_actions`.
	"""
	def policy_fn(sess, state, epsilon):
		action_probabilities = np.ones(n_actions, dtype=float) * epsilon / n_actions
		q_values = estimator.run(sess, np.expand_dims(state, 0))[0]
		best_action = np.argmax(q_values)
		action_probabilities[best_action] += (1 - epsilon)
		return action_probabilities
	return policy_fn


class ReplayMemory(ABC):

	def __init__(self, size, state_shape, save_directory):
		self.size = size
		self.state_shape = state_shape
		self.save_directory = save_directory

	@abstractmethod
	def append(self, state, action, reward, next_state, is_done):
		pass

	@abstractmethod
	def get_batch(self, size):
		pass


class MemmapReplayMemory(ReplayMemory):

	def __init__(self, size, state_shape, save_directory):
		super().__init__(size, state_shape, save_directory)
		self.n = 0
		self.index = 0

		# Define memmap shapes
		state_shape = (size, *self.state_shape)
		action_shape = (size, 1)
		reward_shape = (size, 1)
		next_state_shape = (size, *self.state_shape)
		is_done_shape = (size, 1)

		# Define memmap paths
		state_filepath = os.path.join(self.save_directory, 'state.rm')
		action_filepath = os.path.join(self.save_directory, 'action.rm')
		reward_filepath = os.path.join(self.save_directory, 'reward.rm')
		next_state_filepath = os.path.join(self.save_directory, 'next_state.rm')
		is_done_filepath = os.path.join(self.save_directory, 'is_done.rm')

		# Create or open memmap
		self.state_memmap = self._create_or_open_memmap(state_filepath, state_shape)
		self.action_memmap = self._create_or_open_memmap(action_filepath, action_shape)
		self.reward_memmap = self._create_or_open_memmap(reward_filepath, reward_shape)
		self.next_state_memmap = self._create_or_open_memmap(next_state_filepath, next_state_shape)
		self.is_done_memmap = self._create_or_open_memmap(is_done_filepath, is_done_shape)

		# Fetch stored data, if any
		metadata_filepath = os.path.join(self.save_directory, 'metadata.rm')
		metadata_shape = (2,)
		if os.path.isfile(metadata_filepath):
			self.metadata_memmap = self._create_or_open_memmap(metadata_filepath, metadata_shape)
			self.n = int(self.metadata_memmap[0])
			self.index = int(self.metadata_memmap[1])
		else:
			self.metadata_memmap = self._create_or_open_memmap(metadata_filepath, metadata_shape)

	def _create_or_open_memmap(self, filepath, shape):
		mode = 'w+'
		if os.path.isfile(filepath):
			mode = 'r+'
		return np.memmap(
			filepath,
			dtype='float32',
			mode=mode,
			shape=shape
		)

	def append(self, state, action, reward, next_state, is_done):
		self.state_memmap[self.index] = state
		self.action_memmap[self.index] = action
		self.reward_memmap[self.index] = reward
		self.next_state_memmap[self.index] = next_state
		self.is_done_memmap[self.index] = is_done
		# self.state_memmap.flush()
		# self.action_memmap.flush()
		# self.reward_memmap.flush()
		# self.next_state_memmap.flush()
		# self.is_done_memmap.flush()

		self.index = (self.index + 1) % self.size

		if self.n < self.size:
			self.n += 1
			self.metadata_memmap[0] = self.n
			# self.metadata_memmap.flush()

	def get_batch(self, size):
		indices = np.random.random_integers(0, self.n-1, size)

		state_batch = np.take(self.state_memmap, indices, axis=0)
		action_batch = np.take(self.action_memmap, indices, axis=0)
		reward_batch = np.take(self.reward_memmap, indices, axis=0)
		next_state_batch = np.take(self.next_state_memmap, indices, axis=0)
		is_done_batch = np.take(self.is_done_memmap, indices, axis=0)

		return state_batch, action_batch, reward_batch, next_state_batch, is_done_batch


def deep_q_learning(
		sess,
		env,
		q_estimator,
		target_estimator,
		image_processor,
		n_episodes,
		save_directory,
		replay_memory_size=500000,
		replay_memory_init_size=50000,
		update_target_estimator_every=10000,
		discount_factor=0.99,
		epsilon_start=1.0,
		epsilon_end=0.1,
		epsilon_decay_steps=500000,
		batch_size=32,
		record_video_every=50):
	"""
	Deep Q-learning with experience replay.

	Args:
		sess: TensorFlow session object
		env: OpenAI environment
		q_estimator: Estimator object used for the q values
		target_estimator: Estimator object used for the targets
		image_processor: ImageProcessor object
		n_episodes: Number of episodes to run
		save_directory: Directory in which to save checkpoints and summaries
		replay_memory_size: Size of replay memory
		replay_memory_init_size: Number of random experiences to sample when
			initializing replay memory
		update_target_estimator_every: Number of steps before copying parameters
			from Q-estimator to target Q-estimator
		discount_factor: Gamma discount factor
		epsilon_start: Chance to sample a random action when taking an action
		epsilon_end: Minimum value of epsilon when decaying is done
		epsilon_decay_steps: Number of steps before decaying epsilon
		batch_size: Size of random sample taken from replay memory
		record_video_every: Number of episodes before recording video

	Returns:
		An EpisodeStats object with two numpy arrays for episode_lengths and
		episode_rewards.
	"""
	EpisodeStats = namedtuple("Stats", ["episode_lengths", "episode_rewards"])

	stats = EpisodeStats(
		episode_lengths=np.zeros(n_episodes),
		episode_rewards=np.zeros(n_episodes)
	)

	copier = NetworkCopier(
		origin_model=q_estimator,
		target_model=target_estimator
	)

	# Global step
	global_step = sess.run(tf.train.get_global_step())

	# Define save directories
	replay_memory_directory = os.path.join(save_directory, 'replay_memory')
	checkpoint_directory = os.path.join(save_directory, 'checkpoints')
	monitor_directory = os.path.join(save_directory, 'monitor')
	summary_directory = os.path.join(save_directory, 'summary')
	if not os.path.exists(replay_memory_directory):
		os.makedirs(replay_memory_directory)
	if not os.path.exists(checkpoint_directory):
		os.makedirs(checkpoint_directory)
	if not os.path.exists(monitor_directory):
		os.makedirs(monitor_directory)
	if not os.path.exists(summary_directory):
		os.makedirs(summary_directory)

	# Load latest checkpoint if it exists
	saver = tf.train.Saver()
	latest_checkpoint = tf.train.latest_checkpoint(checkpoint_directory)
	if latest_checkpoint:
		print('Loading model checkpoint: {}'.format(latest_checkpoint))
		saver.restore(sess, latest_checkpoint)
		print('Loaded model checkpoint')

	# Epsilon decay schedule
	epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

	# Epsilon-greedy policy
	policy = make_epsilon_greedy_policy(
		estimator=q_estimator,
		n_actions=env.action_space.n
	)

	# Populate replay memory
	print('Loading replay memory: {}'.format(replay_memory_directory))
	replay_memory = MemmapReplayMemory(
		size=replay_memory_size,
		state_shape=(*PROCESSED_SHAPE, 4),
		save_directory=replay_memory_directory
	)
	print('Loaded replay memory')
	if replay_memory.n < replay_memory_init_size:
		state = env.reset()
		state = image_processor.run(sess, state)
		state = np.stack([state] * 4, axis=2)
		for i in range(replay_memory_init_size - replay_memory.n):
			if (i+replay_memory.n) % (replay_memory_init_size // 100) == 0:
				print('Populating replay memory: {}%'.format((i+replay_memory.n) * 100 // replay_memory_init_size))
			epsilon = epsilons[min(global_step, epsilon_decay_steps - 1)]
			action_probabilities = policy(
				sess=sess,
				state=state,
				epsilon=epsilon
			)
			action = np.random.choice(
				np.arange(len(action_probabilities)),
				p=action_probabilities
			)
			next_state, reward, is_done, _ = env.step(action)
			next_state = image_processor.run(sess, next_state)
			next_state = np.append(state[:, :, 1:], np.expand_dims(next_state, 2), axis=2)

			# Append to replay memory
			replay_memory.append(state, action, reward, next_state, is_done)

			if is_done:
				state = env.reset()
				state = image_processor.run(sess, state)
				state = np.stack([state] * 4, axis=2)
			else:
				state = next_state

	# Record video
	env = wrappers.Monitor(
		env=env,
		directory=monitor_directory,
		video_callable=lambda count: count % record_video_every == 0,
		resume=True,
	)

	# Summary writer
	summary_writer = tf.summary.FileWriter(summary_directory)

	# Run through episodes
	print('Starting episodes')
	for episode in range(n_episodes):
		# Save checkpoint
		saver.save(
			sess,
			save_path=os.path.join(checkpoint_directory, 'model'),
			global_step=tf.train.get_global_step()
		)

		# Reset environment
		state = env.reset()
		state = image_processor.run(sess, state)
		state = np.stack([state] * 4, axis=2)

		# Do episode
		step = 0
		is_done = False
		while not is_done:
			# Update target network
			if global_step % update_target_estimator_every == 0:
				print('Copying Q-network parameters to target network')
				copier.run(sess)

			# Take a step in episode
			epsilon = epsilons[min(global_step, epsilon_decay_steps-1)]
			action_probabilities = policy(
				sess=sess,
				state=state,
				epsilon=epsilon
			)
			action = np.random.choice(
				np.arange(len(action_probabilities)),
				p=action_probabilities
			)
			next_state, reward, is_done, _ = env.step(action)
			next_state = image_processor.run(sess, next_state)
			next_state = np.append(state[:, :, 1:], np.expand_dims(next_state, 2), axis=2)

			# Append to replay memory
			replay_memory.append(state, action, reward, next_state, is_done)

			# Update statistics
			stats.episode_rewards[episode] += reward
			stats.episode_lengths[episode] = step

			# Sample a batch from replay memory
			state_batch, action_batch, reward_batch, next_state_batch, is_done_batch = replay_memory.get_batch(batch_size)

			# Squeeze into a single dimension
			action_batch = np.squeeze(action_batch)
			reward_batch = np.squeeze(reward_batch)
			is_done_batch = np.squeeze(is_done_batch)

			# print('state_batch shape: {}'.format(state_batch.shape))
			# print('action_batch shape: {}'.format(action_batch.shape))
			# print('reward_batch shape: {}'.format(reward_batch.shape))
			# print('next_state_batch shape: {}'.format(next_state_batch.shape))
			# print('is_done_batch shape: {}'.format(is_done_batch.shape))

			# Get Q-values from frozen estimator
			next_q_value_batch = target_estimator.run(sess, next_state_batch)

			# Final step is just `reward_batch` so multiply the future reward by 0 if `is_done` is True
			future_reward = np.invert(is_done_batch.astype(np.bool)).astype(np.float32) * discount_factor * np.amax(next_q_value_batch, axis=1)
			target_batch = reward_batch + future_reward

			# Run a train step
			summary, global_step, loss = q_estimator.update(
				sess=sess,
				state=state_batch,
				action=action_batch,
				target=target_batch
			)

			# Add summaries to TensorBoard
			summary_writer.add_summary(summary, global_step)

			print('{: <18}'.format('#: {}'.format(global_step)), end=' ')
			print('{: <18}'.format('Step: {}'.format(step)), end=' ')
			print('{: <18}'.format('Episode: {}'.format(episode + 1)), end=' ')
			print('{: <18}'.format('Loss: {:.4f}'.format(loss)))

			state = next_state
			step += 1

		# Add summaries to TensorBoard
		summary = tf.Summary()
		summary.value.add(tag='episode/rewards', simple_value=stats.episode_rewards[episode])
		summary.value.add(tag='episode/steps', simple_value=stats.episode_lengths[episode])
		summary.value.add(tag='episode/epsilon', simple_value=epsilon)
		summary_writer.add_summary(summary, global_step)

		# Write TensorBoard summary
		summary_writer.flush()

		yield global_step, EpisodeStats(
			episode_lengths=stats.episode_lengths[:episode + 1],
			episode_rewards=stats.episode_rewards[:episode + 1]
		)

	return stats


def test_image_processor(env):
	image_processor = ImageProcessor(scope='test_image_processor')

	with tf.Session() as sess:
		states = list()
		processed_states = list()

		# Take some actions and get state
		state = env.reset()
		states.append(state)
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

		# Process states
		for state in states:
			state_processed = image_processor.run(sess, state)
			processed_states.append(state_processed)

		print('Original image shape: {}'.format(states[0].shape))
		print('Processed image shape: {}'.format(processed_states[0].shape))
		print('Sample processed image:')
		print(processed_states[0])

		# Plot
		for i, state in enumerate(states):
			plt.subplot('2{}{}'.format(len(states), 1 + i))
			plt.title('Original {}'.format(i))
			plt.imshow(state)
		for i, state in enumerate(processed_states):
			plt.subplot('2{}{}'.format(len(states), 1 + len(states) + i))
			plt.title('Processed {}'.format(i))
			plt.imshow(state, cmap='gray', vmin=0, vmax=255)
		plt.show()


def test_estimator(env):
	tf.reset_default_graph()

	# Global step variable
	tf.Variable(0, name='global_step', trainable=False)

	estimator = Estimator(n_actions=env.action_space.n, scope='test_estimator')
	image_processor = ImageProcessor(scope='test_image_processor')

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		# Create fake input state data
		state_1 = env.reset()
		processed_state_1 = image_processor.run(sess, state_1)
		processed_state_1 = np.stack([processed_state_1] * 4, axis=2)

		for i in range(20):
			env.step(1)
			env.step(2)
		state_2 = env.render(mode='rgb_array')
		processed_state_2 = image_processor.run(sess, state_2)
		processed_state_2 = np.stack([processed_state_2] * 4, axis=2)
		states = np.array([processed_state_1, processed_state_2])

		# Make prediction
		prediction = estimator.run(sess, states)
		print('Prediction:')
		print(prediction)
		print('-' * 80)

		# Create fake target and action values to learn
		target = np.array([10.0, 10.0])
		action = np.array([1, 2])

		print('States shape: {}'.format(states.shape))
		print('Action shape: {}'.format(action.shape))
		print('Target shape: {}'.format(target.shape))

		# Train
		for i in range(1000):
			loss = estimator.update(sess, states, action, target)
			if i % 100 == 0:
				print('Loss: {}'.format(loss))
				print('-' * 80)

		# Check if target action has been learned
		prediction = estimator.run(sess, states)
		print('Prediction:')
		print(prediction)
		print('-' * 80)


def run(env):
	tf.reset_default_graph()

	# Global step variable
	tf.train.get_or_create_global_step()

	q_estimator = Estimator(n_actions=env.action_space.n, scope='q_estimator')
	target_estimator = Estimator(n_actions=env.action_space.n, scope='target_estimator')
	image_processor = ImageProcessor(scope='image_processor')

	save_directory = 'C:\\Users\\Jae\\Desktop\\dqn'
	config = tf.ConfigProto()
	if USE_CPU:
		config = tf.ConfigProto(device_count={'GPU': 0})
	with tf.Session(config=config) as sess:
		sess.run(tf.global_variables_initializer())
		for global_step, stats in deep_q_learning(
				sess=sess,
				env=env,
				q_estimator=q_estimator,
				target_estimator=target_estimator,
				image_processor=image_processor,
				n_episodes=10000,
				save_directory=save_directory):
			print('Episode reward: {}'.format(stats.episode_rewards[-1]))
			print('-' * 80)


def main():
	env = gym.make('Breakout-v0')
	run(env)


if __name__ == '__main__':
	main()
