
from learner import LearnerFC
from learner import LearnerCNN
import gym
import numpy as np

# batch_size = 512
# total_episodes = 1000
# steps_per_episode = 1000
# learning_rate = 1e-3
# env_name = 'CartPole-v0'

batch_size = 16
total_episodes = 1000
steps_per_episode = 1000
learning_rate = 1e-3
env_name = 'Breakout-v0'

learning_count = 5

env = gym.make(env_name)

print env.observation_space
print env.action_space

if len(env.observation_space.shape) == 1:
  flat_input = True
elif len(env.observation_space.shape) == 3:
  flat_input = False

if flat_input:
  num_inputs  = env.observation_space.shape[0]
else:
  width = env.observation_space.shape[0]
  height = env.observation_space.shape[1]
  channels = env.observation_space.shape[2]

num_outputs = env.action_space.n

if flat_input:
  learner = LearnerFC(num_inputs, num_outputs, batch_size=batch_size, learning_rate=learning_rate)
else:
  learner = LearnerCNN(width, height, channels, num_outputs, batch_size=batch_size, learning_rate=learning_rate)

q_max_avg = 0
q_min_avg = 0

count = 0
for i_episode in range(total_episodes):

  # if i_episode % 1000 == 0:
  #   learner.save_model()

  last_observation = observation = env.reset()

  total_reward = 0

  if i_episode % 1 == 0:
    show = True
  else:
    show = False

  # if show:
  #   print("Showing episode no: {} (epsilon: {})".format(i_episode, learner.epsilon))

  for t in range(steps_per_episode):
    if show:
      env.render()

    action, q = learner.action(observation)

    if len(q) > 0:
      # if show:
      #   print action, q[0], q[1]

      q_max = max(q)
      q_min = min(q)
      q_max_avg = 0.999 * q_max_avg + 0.001 * q_max
      q_min_avg = 0.999 * q_min_avg + 0.001 * q_min

    observation, reward, done, info = env.step(action)
    total_reward += reward

    if learning_count > 0:
      e = (last_observation, action, reward, observation, done)
      learner.add_experience(e)
      learner.step()

    last_observation = observation

    count += 1

    if done:
      if t < steps_per_episode - 1:
        learning_count = 5
      break

    if t == steps_per_episode - 1 and learning_count > 0:
      learning_count -= 1
      learner.epsilon = 0

  print("Episode {0:05d} finished after {1:03d} timesteps. Total Reward: {2:03.2f} (epsilon: {3:.2f}, avg. q_max: {4:.2f}, q_min: {5:.2f}) Epoch: {6:.2f}".format(i_episode, t+1, total_reward, learner.epsilon, q_max_avg, q_min_avg, count/50000.0))
