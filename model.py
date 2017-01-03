
import tensorflow as tf

class Model(object):
  def weight_variable(self, shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

  def bias_variable(self, shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

  def conv2d(self, x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")

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
  def __init__(self, width, height, channels, num_outputs, grayscale=False, resize=True, filter_count=16, layer_count=4, fc_sizes=[128, 128], gamma=0.995, batch_size=64, learning_rate=1e-4):
    x0 = self._x0 = tf.placeholder(tf.float32, [None, width, height, channels])
    x1 = self._x1 = tf.placeholder(tf.float32, [None, width, height, channels])

    r  = self._r  = tf.placeholder(tf.float32, [None])
    f  = self._f  = tf.placeholder(tf.float32, [None])
    a  = self._a  = tf.placeholder(tf.int32, [None])

    keep_prob = self._keep_prob = tf.placeholder(tf.float32)

    w = width
    h = height

    if grayscale:
      channels = 1
      x0 = tf.image.rgb_to_grayscale(x0)
      x1 = tf.image.rgb_to_grayscale(x1)

    if resize:
      w = w/2
      h = h/2
      x0 = tf.image.resize_images(x0, [w, h])
      x1 = tf.image.resize_images(x1, [w, h])

    cnn_weights = []
    last_size = channels
    size = filter_count

    for i in range(layer_count):
      cnn_weights.append((self.weight_variable([3, 3, last_size, size]), self.bias_variable([size]), self.weight_variable([3, 3, size, size]), self.bias_variable([size])))
      last_size = size
      size = size * 2
      w = w/2
      h = h/2

    fc_weights = []

    last_size = fc_shape = w * h * last_size

    for size in fc_sizes:
      fc_weights.append((self.weight_variable([last_size, size]), self.bias_variable([size])))
      last_size = size
    fc_weights.append((self.weight_variable([last_size, num_outputs]), self.bias_variable([num_outputs])))

    q0 = self._q0 = self.q_value(x0, cnn_weights, fc_weights, fc_shape, keep_prob)
    q1 = self.q_value(x1, cnn_weights, fc_weights, fc_shape, 1)

    self._action = tf.argmax(q0, 1)

    temp1 = tf.Variable(tf.zeros([batch_size * num_outputs]), trainable=False, validate_shape=False)
    q0_flat = temp1.assign(tf.reshape(q0, [-1]))
    temp2 = tf.Variable(tf.zeros([batch_size * num_outputs]), trainable=False, validate_shape=False)
    y_flat = temp2.assign(tf.scatter_update(q0_flat, a + range(0, batch_size*num_outputs, num_outputs), r + f * gamma * tf.reduce_max(q1, axis=1)))
    y = tf.reshape(y_flat, [batch_size, num_outputs])
    error = tf.reduce_mean(tf.square(q0 - y))

    self._step = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-4).minimize(error)

  def q_value(self, x, cnn_weights, fc_weights, fc_shape, keep_prob):
    h = x

    for (W1, b1, W2, b2) in cnn_weights:
      h1 = tf.nn.relu(self.conv2d(h, W1) + b1)
      h2 = tf.nn.relu(self.conv2d(h1, W2) + b2)
      h  = tf.nn.max_pool(h2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    h = tf.reshape(h, [-1, fc_shape])

    for (W, b) in fc_weights[:-1]:
      h = tf.nn.relu(tf.matmul(h, W) + b)

    (W,b) = fc_weights[-1]
    h = tf.matmul(tf.nn.dropout(h, keep_prob), W) + b

    return h
