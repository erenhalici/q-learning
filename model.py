
import tensorflow as tf
import math

class Model(object):
  def weight_variable(self, shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

  def bias_variable(self, shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

  def conv2d(self, x, W, stride=1):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")

  @property
  def temp1(self):
    return self._temp1
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
  def keep_prob(self):
    return self._keep_prob
  @property
  def action(self):
    return self._action
  @property
  def step(self):
    return self._step

class ModelFC(Model):
  def __init__(self, num_inputs, num_outputs, fc_sizes=[128, 128], gamma=0.995, batch_size=64, learning_rate=1e-4):
    x0 = self._x0 = tf.placeholder(tf.float32, [None, num_inputs])
    x1 = self._x1 = tf.placeholder(tf.float32, [None, num_inputs])
    r  = self._r  = tf.placeholder(tf.float32, [None])
    f  = self._f  = tf.placeholder(tf.float32, [None])
    a  = self._a  = tf.placeholder(tf.int32, [None])

    keep_prob = self._keep_prob = tf.placeholder(tf.float32)

    weights = []
    last_size = num_inputs
    for size in fc_sizes:
      weights.append((self.weight_variable([last_size, size]), self.bias_variable([size])))
      last_size = size
    weights.append((self.weight_variable([last_size, num_outputs]), self.bias_variable([num_outputs])))

    q0 = self._q0 = self.q_value(x0, weights, keep_prob)
    q1 = self.q_value(x1, weights, 1)

    self._action = tf.argmax(q0, 1)

    temp1 = tf.Variable(tf.zeros([batch_size * num_outputs]), trainable=False, validate_shape=False)
    q0_flat = temp1.assign(tf.reshape(q0, [-1]))
    temp2 = tf.Variable(tf.zeros([batch_size * num_outputs]), trainable=False, validate_shape=False)
    y_flat = temp2.assign(tf.scatter_update(q0_flat, a + range(0, batch_size*num_outputs, num_outputs), r + f * gamma * tf.reduce_max(q1, axis=1)))
    y = tf.reshape(y_flat, [batch_size, num_outputs])
    error = tf.reduce_mean(tf.square(q0 - y))

    self._step = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-4).minimize(error)

  def q_value(self, x, weights, keep_prob):
    h = x

    for (W, b) in weights[:-1]:
      h = tf.nn.relu(tf.matmul(h, W) + b)

    (W,b) = weights[-1]
    h = tf.matmul(tf.nn.dropout(h, keep_prob), W) + b

    return h

class ModelCNN(Model):
  def __init__(self, width, height, channels, num_outputs, gamma=0.995, batch_size=32, learning_rate=1e-4, dropout=0.5):
    x0 = self._x0 = tf.placeholder(tf.float32, [None, width, height, channels])
    x1 = self._x1 = tf.placeholder(tf.float32, [None, width, height, channels])
    r  = self._r  = tf.placeholder(tf.float32, [None])
    f  = self._f  = tf.placeholder(tf.float32, [None])
    a  = self._a  = tf.placeholder(tf.int32, [None])

    keep_prob = self._keep_prob = tf.placeholder(tf.float32)

    w = width
    h = height

    WCNN1 = self.weight_variable([8, 8, channels, 16])
    bCNN1 = self.bias_variable([16])

    WCNN2 = self.weight_variable([4, 4, 16, 32])
    bCNN2 = self.bias_variable([32])

    fc_size = int(math.ceil(math.ceil(width/4.0)/2.0) * math.ceil(math.ceil(height/4.0)/2.0) * 32.0)
    WFC1 = self.weight_variable([fc_size, 256])
    bFC1 = self.weight_variable([256])

    WFC2 = self.weight_variable([256, num_outputs])
    bFC2 = self.weight_variable([num_outputs])

    weights = (WCNN1, bCNN1, WCNN2, bCNN2, WFC1, bFC1, WFC2, bFC2)

    q0 = self._q0 = self.q_value(x0, weights, fc_size, keep_prob)
    q1 = self.q_value(x1, weights, fc_size, 1)

    self._action = tf.argmax(q0, 1)

    temp1 = tf.Variable(tf.zeros([batch_size * num_outputs]), trainable=False, validate_shape=False)
    q0_flat = temp1.assign(tf.reshape(q0, [-1]))
    temp2 = tf.Variable(tf.zeros([batch_size * num_outputs]), trainable=False, validate_shape=False)
    y_flat = temp2.assign(tf.scatter_update(q0_flat, a + range(0, batch_size*num_outputs, num_outputs), r + f * gamma * tf.reduce_max(q1, axis=1)))
    y = tf.reshape(y_flat, [batch_size, num_outputs])
    error = tf.reduce_mean(tf.square(q0 - y))

    self._step = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-4).minimize(error)

  def q_value(self, x, weights, fc_size, keep_prob):
    (WCNN1, bCNN1, WCNN2, bCNN2, WFC1, bFC1, WFC2, bFC2) = weights

    h1 = tf.nn.relu(self.conv2d(x,  WCNN1, stride=4) + bCNN1)
    h2 = tf.nn.relu(self.conv2d(h1, WCNN2, stride=2) + bCNN2)
    h3 = tf.reshape(h2, [-1, fc_size])
    h4 = tf.nn.relu(tf.matmul(h3, WFC1) + bFC1)
    h5 = tf.matmul(tf.nn.dropout(h4, keep_prob), WFC2) + bFC2

    return h5
