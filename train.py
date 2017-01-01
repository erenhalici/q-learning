
from learner import Learner
import gym
import numpy as np

batch_size = 512
total_episodes = 1000
steps_per_episode = 1000
env_name = 'CartPole-v0'

env = gym.make(env_name)

num_inputs  = env.observation_space.shape[0]
num_outputs = env.action_space.n

learner = Learner(num_inputs, num_outputs, batch_size=batch_size)

q_max_avg = 0
q_min_avg = 0

count = 0
for i_episode in range(total_episodes):

  # if i_episode % 1000 == 0:
  #   learner.save_model()

  episode_experiences = []

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
      q_max_avg = 0.9 * q_max_avg + 0.1 * q_max
      q_min_avg = 0.9 * q_min_avg + 0.1 * q_min

    observation, reward, done, info = env.step(action)
    total_reward += reward

    e = (last_observation, action, reward, observation, done)
    episode_experiences.append(e)
    learner.add_experience(e)
    learner.step()

    last_observation = observation

    count += 1

    if done:
      break

  print("Episode {0:05d} finished after {1:03d} timesteps. Total Reward: {2:03.0f} (epsilon: {3:.2f}, avg. q_max: {4:.2f}, q_min: {5:.2f}) Epoch: {6:.2f}".format(i_episode, t+1, total_reward, learner.epsilon, q_max_avg, q_min_avg, count/50000.0))
