
import tensorflow as tf

class Model(object):
  def __init__(self, num_inputs, num_outputs, fc_sizes=[128, 128], gamma=0.995, batch_size=64, learning_rate=1e-4):
    x0 = self._x0 = tf.placeholder(tf.float32, [None, num_inputs])
    x1 = self._x1 = tf.placeholder(tf.float32, [None, num_inputs])
    r  = self._r  = tf.placeholder(tf.float32, [None])
    f  = self._f  = tf.placeholder(tf.float32, [None])
    a  = self._a  = tf.placeholder(tf.int32, [None])

    weights = []
    last_size = num_inputs
    for size in fc_sizes:
      weights.append((self.weight_variable([last_size, size]), self.bias_variable([size])))
      last_size = size
    weights.append((self.weight_variable([last_size, num_outputs]), self.bias_variable([num_outputs])))

    q0 = self._q0 = self.q_value(x0, weights)
    q1 = self.q_value(x1, weights)

    self._action = tf.argmax(q0, 1)

    temp1 = tf.Variable(tf.zeros([batch_size * num_outputs]), trainable=False, validate_shape=False)
    q0_flat = temp1.assign(tf.reshape(q0, [-1]))
    temp2 = tf.Variable(tf.zeros([batch_size * num_outputs]), trainable=False, validate_shape=False)
    y_flat = temp2.assign(tf.scatter_update(q0_flat, a + range(0, batch_size*num_outputs, 2), r + f * gamma * tf.reduce_max(q1, axis=1)))
    y = tf.reshape(y_flat, [batch_size, num_outputs])
    error = tf.reduce_mean(tf.square(q0 - y))

    # temp1 = tf.Variable(tf.zeros([batch_size]), trainable=False, validate_shape=False)
    # y = temp1.assign(r + f * gamma * tf.reduce_max(q1, axis=1))
    # x = tf.gather_nd(q0, a)
    # error = tf.reduce_mean(tf.square(x - y))

    # error = -tf.reduce_mean(x * tf.log(y))
    self._step = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-4).minimize(error)

  def q_value(self, x, weights):
    h = x

    for (W, b) in weights[:-1]:
      h = tf.nn.relu(tf.matmul(h, W) + b)

    (W,b) = weights[-1]
    h = tf.matmul(h, W) + b

    return h

  def weight_variable(self, shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

  def bias_variable(self, shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

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
  def action(self):
    return self._action
  @property
  def step(self):
    return self._step
