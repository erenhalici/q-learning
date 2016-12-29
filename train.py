
from learner import Learner
import gym

# env = gym.make('SpaceInvadersDeterministic-v3')
env = gym.make('CartPole-v0')
learner = Learner(4, 2)

print(env.action_space)
print(env.observation_space)

q_max_avg = 0
q_min_avg = 0

max_run = 0
max_experiences = []

for i_episode in range(200000):

  experiences = []

  last_observation = observation = env.reset()

  total_reward = 0

  if i_episode % 400 < 4:
    show = True
  else:
    show = False

  if show:
    print("Showing episode no: {} (epsilon: {})".format(i_episode, learner.epsilon))

  for t in range(1000):
    if show:
      env.render()

    action, q_max, q_min = learner.action(observation, best=show)
    # action, q_max, q_min = learner.action(observation, best=True)

    q_max_avg = 0.99 * q_max_avg + 0.1 * q_max
    q_min_avg = 0.99 * q_min_avg + 0.1 * q_min

    observation, reward, done, info = env.step(action)
    total_reward += reward

    e = ((last_observation, action, reward, observation, done))
    experiences.append(e)
    learner.add_experience(e)
    learner.step()
    learner.train_with(max_experiences)

    last_observation = observation

    if done:
      print("Episode {0:05d} finished after {1:04d} timesteps. Total Reward: {2:4.2f} (avg. q_max: {3:.2f}, q_min: {4:.2f})".format(i_episode, t+1, total_reward, q_max_avg, q_min_avg))

      if total_reward > max_run:
        max_run = total_reward
        max_experiences = experiences
        print("Updated best run to: {}".format(max_run))

      break

