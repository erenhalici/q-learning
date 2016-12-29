
import tensorflow as tf

class Model(object):
  def __init__(self, num_inputs, num_outputs, fc_sizes=[5, 5], gamma=0.7, learning_rate=1e-4):
    x0 = self._x0 = tf.placeholder(tf.float32, [None, num_inputs])
    x1 = self._x1 = tf.placeholder(tf.float32, [None, num_inputs])
    r  = self._r  = tf.placeholder(tf.float32, [None])
    f  = self._f  = tf.placeholder(tf.float32, [None])
    a  = self._a  = tf.placeholder(tf.int32, [None, 2])

    weights = []
    last_size = num_inputs
    for size in fc_sizes:
      weights.append((self.weight_variable([last_size, size]), self.bias_variable([size])))
      last_size = size
    weights.append((self.weight_variable([last_size, num_outputs]), self.bias_variable([num_outputs])))

    q0 = self._q0 = self.q_value(x0, weights)
    q1 = self.q_value(x1, weights)

    self._action = tf.argmax(q0, 1)

    y = r + f * gamma * tf.reduce_max(q1, axis=1)
    x = tf.gather_nd(q0, a)

    error = tf.square(x - y)

    self._step = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-4).minimize(error)

  def q_value(self, x, weights):
    h = x

    for (W, b) in weights:
      h = tf.nn.relu(tf.matmul(h, W) + b)

    return h

  def weight_variable(self, shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

  def bias_variable(self, shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

  @property
  def x0(self):
    return self._x0
  @property
  def x1(self):
    return self._x1
  @property
  def q0(self):
    return self._q0
  @property
  def r(self):
    return self._r
  @property
  def f(self):
    return self._f
  @property
  def a(self):
    return self._a
  @property
  def action(self):
    return self._action
  @property
  def step(self):
    return self._step
