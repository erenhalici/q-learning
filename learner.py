
from model import ModelFC
from model import ModelCNN
import tensorflow as tf
import numpy as np
import random

class Learner(object):
  def action(self, observation, best=False):
    if best==False and random.random() < self._epsilon:
      return random.randrange(self._num_outputs), []

    model = self._model
    # return self._sess.run(model.action, feed_dict={model.x0: [observation]})[0]
    q0 = self._sess.run(model.q0, feed_dict={model.x0: [observation], model.keep_prob: 1.0})[0]
    action = np.argmax(q0)
    return action, q0

  def add_experience(self, e):
    self._experiences.append(e)
    while (len(self._experiences) > self._exp_size):
      self._experiences.pop(0)

  def experience_size(self):
    return len(self._experiences)

  def step_with(self, experiences):
    if (len(experiences) >= self._batch_size):
      self.train_with(random.sample(experiences, self._batch_size))

  def step(self):
    if (len(self._experiences) >= self._batch_size):
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

    for (last_observation, action, reward, observation, done) in experiences:
      x0.append(last_observation)
      a.append(action)
      r.append(reward)
      x1.append(observation)
      if done:
        f.append(0)
      else:
        f.append(1)

    self._sess.run(m.step, feed_dict={m.x0: x0, m.a: a, m.r: r, m.x1: x1, m.f: f, m.keep_prob: self._dropout})

    self._epsilon = self._epsilon * self._epsilon_decay
    if self._epsilon < self._min_epsilon:
      self._epsilon = self._min_epsilon
  def save_model(self, model_file):
    self._saver.save(self._sess, model_file)

  def load_model(self, model_file):
    self._saver.restore(self._sess, model_file)

  @property
  def epsilon(self):
    return self._epsilon
  @epsilon.setter
  def epsilon(self, epsilon):
    self._epsilon = epsilon

class LearnerFC(Learner):
  def __init__(self, num_inputs, num_outputs, batch_size=64, exp_size=1000000, min_epsilon=0.05, epsilon_decay=0.9995, learning_rate=1e-4, dropout=0.5):
    self._num_inputs  = num_inputs
    self._num_outputs = num_outputs
    self._batch_size  = batch_size
    self._learning_rate = learning_rate
    self._dropout = dropout

    self._model = ModelFC(num_inputs, num_outputs, batch_size=batch_size, learning_rate=learning_rate)

    self._saver = tf.train.Saver()

    self._sess  = tf.Session()
    self._sess.run(tf.global_variables_initializer())

    self._experiences = []
    self._exp_size = exp_size

    self._epsilon = 1.0
    self._min_epsilon = min_epsilon
    self._epsilon_decay = epsilon_decay


class LearnerCNN(Learner):
  def __init__(self, width, height, channels, num_outputs, batch_size=64, exp_size=500000, min_epsilon=0.1, epsilon_decay=0.99999, learning_rate=1e-4, dropout=0.5):
    self._width  = width
    self._height = height
    self._channels = channels
    self._num_outputs = num_outputs
    self._batch_size  = batch_size
    self._learning_rate = learning_rate
    self._dropout = dropout

    self._model = ModelCNN(width, height, channels, num_outputs, batch_size=batch_size, learning_rate=learning_rate)

    self._saver = tf.train.Saver()

    self._sess  = tf.Session()
    self._sess.run(tf.global_variables_initializer())

    self._experiences = []
    self._exp_size = exp_size

    self._epsilon = 1.0
    self._min_epsilon = min_epsilon
    self._epsilon_decay = epsilon_decay
