
from learner import Learner
import gym
import numpy as np

temporal_window = 1
batch_size = 512

# env = gym.make('SpaceInvadersDeterministic-v3')
env = gym.make('CartPole-v0')

print(env.action_space)
print(env.observation_space)

num_inputs  = 4 * temporal_window
num_outputs = 2

learner = Learner(num_inputs, num_outputs, batch_size=batch_size)


q_max_avg = 0
q_min_avg = 0

max_experiences = []
min_experiences = []
end_experiences = []

count = 0
for i_episode in range(200000):

  if i_episode % 1000 == 0:
    learner.save_model()

  episode_experiences = []

  last_observation = observation = env.reset()
  # last_observation2 = None

  total_reward = 0

  if i_episode % 50 == 0:
    show = True
  else:
    show = False

  if show:
    print("Showing episode no: {} (epsilon: {})".format(i_episode, learner.epsilon))

  for t in range(400):
    if show:
      env.render()

    # action, q_max, q_min = learner.action(np.concatenate((observation, last_observation)), best=show)
    action, q = learner.action(observation, best=show)
    # action, q = learner.action(observation, best=True)

    if len(q) > 0:
      if show:
        print action, q[0], q[1]

      q_max = max(q)
      q_min = min(q)
      q_max_avg = 0.9 * q_max_avg + 0.1 * q_max
      q_min_avg = 0.9 * q_min_avg + 0.1 * q_min

    observation, reward, done, info = env.step(action)
    # if action == 0:
    #   reward = 0
    # else:
    #   reward = 10
    total_reward += reward

    # if last_observation2 != None:
      # e = ((np.concatenate((last_observation, last_observation2)), action, reward, np.concatenate((observation, last_observation)), done))
    e = (last_observation, action, reward, observation, done)
    episode_experiences.append(e)
    learner.add_experience(e)
    learner.step()

    # last_observation2 = last_observation
    last_observation = observation

    count += 1

    if done:
      break

  print("Episode {0:05d} finished after {1:04d} timesteps. Total Reward: {2:6.2f} (avg. q_max: {3:.2f}, q_min: {4:.2f}) Epoch: {5:.2f}".format(i_episode, t+1, total_reward, q_max_avg, q_min_avg, count/50000.0))

  max_experiences.append((episode_experiences, total_reward))

  if len(max_experiences) > batch_size:
    min_reward = total_reward

    for i in range(len(max_experiences)):
      (_, r) = max_experiences[i]
      if r <= min_reward:
        min_reward = r
        min_index = i

    if min_reward != total_reward:
      print("Eliminated run with reward: {}".format(min_reward))

    max_experiences.pop(min_index)

  end_experiences.append(e)
  if (len(end_experiences) > len(max_experiences)):
    end_experiences.pop(0)

  ll = [item for (exps, r) in max_experiences for item in exps]

  for _ in range(10):
    learner.step_with(ll)
    learner.step_with(end_experiences)
