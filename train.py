
from learner import LearnerFC
from learner import LearnerCNN
import os
import gym
import numpy as np

batch_size = 512
temporal_window_size = 4
total_episodes = 1000
steps_per_episode = 1000
learning_rate = 1e-3
env_name = 'CartPole-v0'

# batch_size = 16
# temporal_window_size = 4
# total_episodes = 10000
# steps_per_episode = 100000
# learning_rate = 1e-3
# env_name = 'Breakout-v0'

learning_count = 5


def convert_observation(temporal_window):
  if flat_input:
    return np.hstack(temporal_window)
  else:
    return np.dstack(temporal_window)


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
  width  = env.observation_space.shape[0]
  height = env.observation_space.shape[1]
  channels = env.observation_space.shape[2]

num_outputs = env.action_space.n

if flat_input:
  learner = LearnerFC(num_inputs*temporal_window_size, num_outputs, batch_size=batch_size, learning_rate=learning_rate)
else:
  learner = LearnerCNN(width, height, channels*temporal_window_size, num_outputs, batch_size=batch_size, learning_rate=learning_rate)

show = True

directory = 'models/' + env_name
if not os.path.exists(directory):
    os.makedirs(directory)

q_max_avg = 0
q_min_avg = 0
count = 0
for i_episode in range(total_episodes):
  learner.save_model(directory + '/model-'+str(i_episode))

  # temporal_window = [env.reset()]
  # while (len(temporal_window) < temporal_window_size):
  #   temporal_window.append(env.step(env.action_space.sample())[0])
  temporal_window = [env.reset()] * temporal_window_size

  total_reward = 0
  for t in range(steps_per_episode):
    if show:
      env.render()

    last_observation = convert_observation(temporal_window)
    action, q = learner.action(last_observation)

    if len(q) > 0:
      q_max = max(q)
      q_min = min(q)
      q_max_avg = 0.999 * q_max_avg + 0.001 * q_max
      q_min_avg = 0.999 * q_min_avg + 0.001 * q_min

    ob, reward, done, info = env.step(action)
    total_reward += reward

    temporal_window.append(ob)
    temporal_window.pop(0)
    observation = convert_observation(temporal_window)

    if learning_count > 0:
      e = (last_observation, action, reward, observation, done)
      learner.add_experience(e)
      learner.step()

    count += 1

    if done:
      if t < steps_per_episode - 1:
        learning_count = 5
      break

    if t == steps_per_episode - 1 and learning_count > 0:
      learning_count -= 1
      learner.epsilon = 0

  print("Episode {0:05d} finished after {1:03d} timesteps. Total Reward: {2:03.2f} (epsilon: {3:.2f}, avg. q_max: {4:.2f}, q_min: {5:.2f}) Epoch: {6:.2f}".format(i_episode, t+1, total_reward, learner.epsilon, q_max_avg, q_min_avg, count/50000.0))
