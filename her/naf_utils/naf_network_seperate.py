import tensorflow as tf
from her.util import store_args,nn


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
    with tf.variable_scope('value'):
        self.value = tf.nn.tanh(nn(input_pi, layers_sizes=[64,64,64,1], reuse=False, flatten=False, name="value_scope"))

    with tf.variable_scope('advantage'):
        self.l = tf.nn.tanh(nn(input_pi, layers_sizes=[64, 64, 64, dimu*(dimu+1)/2], reuse=False, flatten=False, name="adv_scope"))


    with tf.variable_scope('policy'):
        self.pi_tf =self.max_u * tf.nn.tanh(nn(input_pi, layers_sizes=[64, 64, 64, dimu],
                                               reuse=False, flatten=False, name="pi_scope"))
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


