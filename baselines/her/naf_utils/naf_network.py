import tensorflow as tf
from baselines.her.util import store_args, nn

class Network:
  @store_args
  def __init__(self, inputs_tf, dimo, dimg, dimu, max_u, o_stats, g_stats, hidden, layers,
               **kwargs):
    """The actor-critic network and related training code.

    Args:
        inputs_tf (dict of tensors): all necessary inputs for the network: the
            observation (o), the goal (g), and the action (u)
        dimo (int): the dimension of the observations
        dimg (int): the dimension of the goals
        dimu (int): the dimension of the actions
        max_u (float): the maximum magnitude of actions; action outputs will be scaled
            accordingly
        o_stats (baselines.her.Normalizer): normalizer for observations
        g_stats (baselines.her.Normalizer): normalizer for goals
        hidden (int): number of hidden units that should be used in hidden layers
        layers (int): number of hidden layers
    """
    self.o_tf = inputs_tf['o']
    self.g_tf = inputs_tf['g']
    self.u_tf = inputs_tf['u']

    # Prepare inputs for actor and critic.
    o = self.o_stats.normalize(self.o_tf)
    g = self.g_stats.normalize(self.g_tf)
    input_pi = tf.concat(axis=1, values=[o, g])

    hid_outs = {}

    # Networks.
    self.random_uniform_init = tf.random_uniform_initializer(-0.02, 0.02, seed=123)

    with tf.variable_scope('hidden'):
      hidden_weights_1 = tf.get_variable(name="hw_1", shape=[input_pi.get_shape()[-1],
                                                             self.hidden],
                                         dtype=tf.float32, initializer=self.random_uniform_init)

      hidden_bias_1 = tf.get_variable(name="bias_1", shape=[self.hidden],
                                      dtype=tf.float32, initializer=self.random_uniform_init)

      hidden_weights_2 = tf.get_variable(name="hw_2", shape=[self.hidden,self.hidden],
                                         dtype=tf.float32, initializer=self.random_uniform_init)

      hidden_bias_2 = tf.get_variable(name="bias_2", shape=[self.hidden],
                                      dtype=tf.float32, initializer=self.random_uniform_init)

      hidden_weights_3 = tf.get_variable(name="hw_3", shape=[self.hidden,self.hidden],
                                         dtype=tf.float32, initializer=self.random_uniform_init)

      hidden_bias_3 = tf.get_variable(name="bias_3", shape=[self.hidden],
                                      dtype=tf.float32, initializer=self.random_uniform_init)

      hidden_1_out = tf.nn.tanh(
        tf.nn.xw_plus_b(input_pi, weights=hidden_weights_1, biases=hidden_bias_1))
      hidden_2_out = tf.nn.tanh(
        tf.nn.xw_plus_b(hidden_1_out, weights=hidden_weights_2, biases=hidden_bias_2))
      hidden_3_out = tf.nn.tanh(
        tf.nn.xw_plus_b(hidden_2_out, weights=hidden_weights_3, biases=hidden_bias_3))

      hid_outs['value'], hid_outs['l'], hid_outs['pi_tf'] = hidden_3_out, hidden_3_out, hidden_3_out

    with tf.variable_scope('value'):
      weights_val = tf.get_variable(name="val_w", shape=[self.hidden,self.hidden/2],
                                         dtype=tf.float32, initializer=self.random_uniform_init)

      bias_val = tf.get_variable(name="val_b", shape=[self.hidden/2],
                                      dtype=tf.float32, initializer=self.random_uniform_init)
      self.value = tf.nn.tanh(tf.nn.xw_plus_b(hid_outs["value"], weights=weights_val, biases=bias_val))

      weights_val_2 = tf.get_variable(name="val_w_2", shape=[self.hidden/2, 1],
                                    dtype=tf.float32, initializer=self.random_uniform_init)

      bias_val_2 = tf.get_variable(name="val_b_2", shape=[1],
                                 dtype=tf.float32, initializer=self.random_uniform_init)
      self.value = tf.nn.tanh(tf.nn.xw_plus_b(self.value, weights=weights_val_2, biases=bias_val_2))


    with tf.variable_scope('advantage'):
      weights_l = tf.get_variable(name="l_w", shape=[self.hidden, dimu*(dimu+1)/2],
                                    dtype=tf.float32, initializer=self.random_uniform_init)

      bias_l = tf.get_variable(name="l_b", shape=[dimu*(dimu+1)/2],
                                 dtype=tf.float32, initializer=self.random_uniform_init)
      self.l = tf.nn.tanh(tf.nn.xw_plus_b(hid_outs["l"], weights=weights_l, biases=bias_l))

      weights_pi_tf = tf.get_variable(name="pi_tf_w", shape=[self.hidden, dimu],
                                    dtype=tf.float32, initializer=self.random_uniform_init)

      bias_pi_tf = tf.get_variable(name="pi_tf_b", shape=[dimu],
                                 dtype=tf.float32, initializer=self.random_uniform_init)
      self.pi_tf = self.max_u * tf.nn.tanh(tf.nn.xw_plus_b(hid_outs["pi_tf"], weights=weights_pi_tf, biases=bias_pi_tf))

      pivot = 0
      rows = []
      for idx in range(dimu):
        count = dimu - idx

        diag_elem = tf.exp(tf.slice(self.l, (0, pivot), (-1, 1)))
        non_diag_elems = tf.slice(self.l, (0, pivot + 1), (-1, count - 1))
        row = tf.pad(tf.concat((diag_elem, non_diag_elems), 1), ((0, 0), (idx, 0)))
        rows.append(row)

        pivot += count

      L = tf.transpose(tf.stack(rows, axis=1), (0, 2, 1))
      P = tf.matmul(L, tf.transpose(L, (0, 2, 1)))

      tmp = tf.expand_dims(self.u_tf - self.pi_tf, -1)
      A = -tf.matmul(tf.transpose(tmp, [0, 2, 1]), tf.matmul(P, tmp)) / 2
      self.A = tf.reshape(A, [-1, 1])
      self.Q = self.A + self.value

