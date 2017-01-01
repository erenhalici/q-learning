
from learner import Learner
import gym
import numpy as np

total_episodes = 1000
steps_per_episode = 1000
env_name = 'CartPole-v0'

env = gym.make(env_name)

num_inputs  = env.observation_space.shape[0]
num_outputs = env.action_space.n

learner = Learner(num_inputs, num_outputs, batch_size=batch_size)
learner.load_model('./model.ckpt')

for i_episode in range(total_episodes):
  last_observation = observation = env.reset()

  total_reward = 0

  for t in range(steps_per_episode):
    env.render()

    action, q = learner.action(observation, best=True)

    observation, reward, done, info = env.step(action)
    total_reward += reward

    if done:
      break

  print("Episode {0:05d} finished after {1:03d} timesteps. Total Reward: {2:03.0f}".format(i_episode, t+1, total_reward))
