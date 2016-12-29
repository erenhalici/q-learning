
from model_fc import Model
import tensorflow as tf
import numpy as np
import random

class Learner(object):
  def __init__(self, num_inputs, num_outputs, batch_size=64, exp_size=300000, min_epsilon=0.05, epsilon_decay=0.99999):
    self._num_inputs  = num_inputs
    self._num_outputs = num_outputs
    self._model = Model(num_inputs, num_outputs)
    self._batch_size = batch_size
    self._sess  = tf.Session()
    self._sess.run(tf.global_variables_initializer())
    self._experiences = []
    self._exp_size = exp_size

    self._epsilon = 1.0
    self._min_epsilon = min_epsilon
    self._epsilon_decay = epsilon_decay

  def action(self, observation, best=False):
    if best==False and random.random() < self._min_epsilon:
      return random.randrange(self._num_outputs), 0, 0

    model = self._model
    # return self._sess.run(model.action, feed_dict={model.x0: [observation]})[0]
    q0 = self._sess.run(model.q0, feed_dict={model.x0: [observation]})[0]
    action = np.argmax(q0)
    q_max = q0[action]
    q_min = min(q0)
    return action, q_max, q_min


  def add_experience(self, e):
    self._experiences.append(e)
    while (len(self._experiences) > self._exp_size):
      self._experiences.pop(0)

  def step_with(self, experiences):
    if (len(experiences) > self._batch_size):
      self.train_with(random.sample(experiences, self._batch_size))
    else:
      self.train_with(experiences)

  def step(self):
    if (len(self._experiences) > self._batch_size):
      self.train_with(random.sample(self._experiences, self._batch_size))

  def train_with(self, experiences):
    if len(experiences) <= 0:
      return

    m = self._model

    x0 = []
    a  = []
    r  = []
    x1 = []
    f  = []

    count = 0
    for (last_observation, action, reward, observation, done) in experiences:
      x0.append(last_observation)
      a.append([count, action])
      r.append(reward)
      x1.append(observation)
      if done:
        f.append(0)
      else:
        f.append(1)

    self._sess.run(m.step, feed_dict={m.x0: x0, m.a: a, m.r: r, m.x1: x1, m.f: f})

    self._epsilon = self._epsilon * self._epsilon_decay
    if self._epsilon < self._min_epsilon:
      self._epsilon = self._min_epsilon


  @property
  def epsilon(self):
    return self._epsilon
