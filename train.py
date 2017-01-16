
from learner import LearnerFC
from learner import LearnerCNN
import os
import gym
import numpy as np
from PIL import Image

# load_model = "mimic/model-3095"
show = False
random = False
train = True

# batch_size = 512
# temporal_window_size = 16
# total_episodes = 1000
# steps_per_episode = 10000
# learning_rate = 1e-3
# frame_skip = 1
# env_name = 'Acrobot-v1'

# batch_size = 512
# temporal_window_size = 4
# total_episodes = 1000
# steps_per_episode = 1000
# learning_rate = 1e-3
# frame_skip = 1
# env_name = 'CartPole-v0'

batch_size = 32
temporal_window_size = 4
total_episodes = 100000
steps_per_episode = 100000
learning_rate = 1e-4
grayscale = True
downsample = True
frame_skip = 4
dropout = 1.0
env_name = 'Breakout-v0'

# batch_size = 512
# temporal_window_size = 1
# total_episodes = 10000
# steps_per_episode = 100000
# learning_rate = 1e-4
# frame_skip = 1
# env_name = 'Breakout-ram-v0'


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
  width  = env.observation_space.shape[0]
  height = env.observation_space.shape[1]
  channels = env.observation_space.shape[2]
  if downsample:
    width = width/2
    height = height/2
  if grayscale:
    channels = 1


num_outputs = env.action_space.n

if flat_input:
  learner = LearnerFC(num_inputs * temporal_window_size, num_outputs, batch_size=batch_size, learning_rate=learning_rate)
else:
  learner = LearnerCNN(width, height, channels * temporal_window_size, num_outputs, batch_size=batch_size, learning_rate=learning_rate, dropout=dropout)

if not train:
  learner.epsilon = 0.0

directory = 'models/' + env_name
if not os.path.exists(directory):
    os.makedirs(directory)

# learner.load_model(directory + '/' + load_model)

def preprocess_observation(observation):
  if flat_input:
    return observation

  o = observation
  if downsample:
    o = o[::2,::2,:]
  if grayscale:
    o = o[:,:,0]
  # Image.fromarray(o).save('im.png')
  return o

def window_to_observation(temporal_window):
  if flat_input:
    return np.hstack(temporal_window)
  else:
    return np.dstack(temporal_window)

q_max_avg = 0
q_min_avg = 0
count = 0

for i_episode in range(total_episodes):
  if train:
    learner.save_model(directory + '/model-'+str(i_episode))

  temporal_window = [preprocess_observation(env.reset())] * temporal_window_size
  done = False

  total_reward = 0
  for t in range(steps_per_episode):
    last_observation = window_to_observation(temporal_window)

    if random:
      action = env.action_space.sample()
      q = []
    else:
      action, q = learner.action(last_observation)

    if len(q) > 0:
      q_max = max(q)
      q_min = min(q)
      q_max_avg = 0.99 * q_max_avg + 0.01 * q_max
      q_min_avg = 0.99 * q_min_avg + 0.01 * q_min

    reward = 0
    for i in range(frame_skip):
      if show:
        env.render()

      if not done:
        ob, r, done, info = env.step(action)
        reward += r
        temporal_window.append(preprocess_observation(ob))
        temporal_window.pop(0)
    total_reward += reward

    observation = window_to_observation(temporal_window)
    # Image.fromarray(observation[:,:,0:3]).save('im.png')

    if train:
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

  print("Episode {0:05d} finished after {1:03d} timesteps. Total Reward: {2:03.2f} (epsilon: {3:.2f}, avg. q_max: {4:.2f}, q_min: {5:.2f}) Epoch: {6:.2f} (exp. size: {7})".format(i_episode, t+1, total_reward, learner.epsilon, q_max_avg, q_min_avg, count/50000.0, learner.experience_size()))
