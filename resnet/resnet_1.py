'''
This is the resnet structure
'''
import numpy as np
from resnet_1_hyper_parameters import *
from tensorflow.python.training import moving_averages
from tensorflow.python.framework import ops
from tensorflow.python.ops import variables

BN_EPSILON = 0.001


def activation_summary(x):
    '''
    :param x: A Tensor
    :return: Add histogram summary and scalar summary of the sparsity of the tensor
    '''
    tensor_name = x.op.name
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def create_variables(name, shape, initializer=tf.contrib.layers.xavier_initializer(), is_fc_layer=False):
    '''
    :param name: A string. The name of the new variable
    :param shape: A list of dimensions
    :param initializer: User Xavier as default.
    :param is_fc_layer: Want to create fc layer variable? May use different weight_decay for fc
    layers.
    :return: The created variable
    '''

    ## TODO: to allow different weight decay to fully connected layer and conv layer
    #if is_fc_layer is True:
    #    regularizer = tf.contrib.layers.l2_regularizer(scale=FLAGS.weight_decay)
    #else:
    #    regularizer = tf.contrib.layers.l2_regularizer(scale=FLAGS.weight_decay)

    #new_variables = tf.get_variable(name, shape=shape, initializer=initializer,
    #                                regularizer=regularizer)
    new_variables = tf.get_variable(name, shape=shape, initializer=initializer)
    return new_variables


def fc_layer(input_layer, num_labels):
    '''
    :param input_layer: 2D tensor
    :param num_labels: int. How many output labels in total? (10 for cifar10 and 100 for cifar100)
    :return: output layer Y = WX + B
    '''
    input_dim = input_layer.get_shape().as_list()[-1]
    fc_w = create_variables(name='fc_weights', shape=[input_dim, num_labels], is_fc_layer=True,
                            initializer=tf.truncated_normal_initializer())
    fc_b = create_variables(name='fc_bias', shape=[num_labels], initializer=tf.zeros_initializer())

    fc_h = tf.matmul(input_layer, fc_w) + fc_b
    return fc_h

def batch_normalization_layer_(x, is_training=True):
    """
    Batch normalization on convolutional maps.
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    Args:
        x:           Tensor, 4D BHWD input maps
        is_training: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope('batch_norm'):
        shape = x.get_shape().as_list()
        n_out = shape[-1]
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                           name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                            name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, list(range(len(shape)-1)), name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.99)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(tf.cast(is_training, tf.bool),
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
        return normed

def batch_normalization_layer(x, is_training, scope='batch_norm', epsilon=0.001, decay=0.999):
    """ 
    Returns a batch normalization layer that automatically switch between train and test phases based on the
    tensor is_training

    Args:
        x: input tensor
        scope: scope name
        is_training: boolean tensor or variable
        epsilon: epsilon parameter - see batch_norm_layer
        decay: epsilon parameter - see batch_norm_layer

    Returns:
        The correct batch normalization layer based on the value of is_training
    """
    assert isinstance(is_training, (ops.Tensor, variables.Variable)) and is_training.dtype == tf.bool

    return tf.cond(
        is_training,
        lambda: batch_norm_layer(x=x, scope=scope, epsilon=epsilon, decay=decay, is_training=True, reuse=None),
        lambda: batch_norm_layer(x=x, scope=scope, epsilon=epsilon, decay=decay, is_training=False, reuse=True),
    )

def batch_norm_layer(x, scope, is_training, epsilon=0.001, decay=0.999, reuse=None):
    """
    Performs a batch normalization layer

    Args:
        x: input tensor
        is_training: python boolean value
        epsilon: the variance epsilon - a small float number to avoid dividing by 0
        decay: the moving average decay

    Returns:
        The ops of a batch normalization layer
    """
    with tf.variable_scope(scope, reuse=reuse):
        shape = x.get_shape().as_list()
        # gamma: a trainable scale factor
        gamma = tf.get_variable("gamma", shape[-1], initializer=tf.constant_initializer(1.0), trainable=True)
        # beta: a trainable shift value
        beta = tf.get_variable("beta", shape[-1], initializer=tf.constant_initializer(0.0), trainable=True)
        moving_avg = tf.get_variable("moving_avg", shape[-1], initializer=tf.constant_initializer(0.0), trainable=False)
        moving_var = tf.get_variable("moving_var", shape[-1], initializer=tf.constant_initializer(1.0), trainable=False)
        if is_training:
            # tf.nn.moments == Calculate the mean and the variance of the tensor x
            avg, var = tf.nn.moments(x, list(range(len(shape)-1)))
            update_moving_avg = moving_averages.assign_moving_average(moving_avg, avg, decay)
            update_moving_var = moving_averages.assign_moving_average(moving_var, var, decay)
            control_inputs = [update_moving_avg, update_moving_var]
        else:
            avg = moving_avg
            var = moving_var
            control_inputs = []
        with tf.control_dependencies(control_inputs):
            output = tf.nn.batch_normalization(x, avg, var, offset=beta, scale=gamma, variance_epsilon=epsilon)

    return output

def batch_normalization_layer_(input_layer, is_training):#, dimension):
    '''
    Helper function to do batch normalziation
    :param input_layer: 4D tensor
    :param dimension: input_layer.get_shape().as_list()[-1]. The depth of the 4D tensor
    :param is_training: if is in training phase
    :return: the 4D tensor after being normalized
    '''
    #axes_ = list(range(len(input_layer.get_shape().as_list()) - 1))
    #mean, variance = tf.nn.moments(input_layer, axes=axes_)
    #beta = tf.get_variable('beta', dimension, tf.float32,
                           #initializer=tf.constant_initializer(0.0, tf.float32))
    #gamma = tf.get_variable('gamma', dimension, tf.float32,
                            #initializer=tf.constant_initializer(1.0, tf.float32))
    #bn_layer = tf.nn.batch_normalization(input_layer, mean, variance, beta, gamma, BN_EPSILON)

    bn_layer = tf.contrib.layers.batch_norm(
        input_layer, center=True, scale=True, decay=0.999, is_training=is_training, scope='batch_norm')
    return bn_layer


def conv_bn_relu_layer(input_layer, filter_shape, stride, is_training):
    '''
    A helper function to conv, batch normalize and relu the input tensor sequentially
    :param input_layer: 4D tensor
    :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
    :param stride: stride size for conv
    :param is_training: if is in training phase
    :return: 4D tensor. Y = Relu(batch_normalize(conv(X)))
    '''

    #out_channel = filter_shape[-1]
    filter_ = create_variables(name='conv', shape=filter_shape)

    conv_layer = tf.nn.conv2d(input_layer, filter_, strides=[1, stride, stride, 1], padding='SAME')
    bn_layer = batch_normalization_layer(conv_layer, is_training)

    output = tf.nn.relu(bn_layer)
    return output


def bn_relu_conv_layer(input_layer, filter_shape, stride, is_training):
    '''
    A helper function to batch normalize, relu and conv the input layer sequentially
    :param input_layer: 4D tensor
    :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
    :param stride: stride size for conv
    :param is_training: if is in training phase
    :return: 4D tensor. Y = conv(Relu(batch_normalize(X)))
    '''

    #in_channel = input_layer.get_shape().as_list()[-1]

    bn_layer = batch_normalization_layer(input_layer, is_training)
    relu_layer = tf.nn.relu(bn_layer)

    filter_ = create_variables(name='conv', shape=filter_shape)
    conv_layer = tf.nn.conv2d(relu_layer, filter_, strides=[1, stride[0], stride[1], 1], padding='SAME')
    return conv_layer


def residual_block(input_layer, output_channel, stride, is_training, first_block=False):
    '''
    Defines a residual block in ResNet
    :param input_layer: 4D tensor
    :param output_channel: int. return_tensor.get_shape().as_list()[-1] = output_channel
    :param stride: stride for both height and width. Can be different
    :param is_training: if is in training phase
    :param first_block: if this is the first residual block of the whole network
    :return: 4D tensor.
    '''
    input_channel = input_layer.get_shape().as_list()[-1]
    if input_channel == output_channel:
        stride = [1, 1]
    # When it's time to "shrink" the image size, we use stride = 2
    '''if input_channel * 2 == output_channel:
        increase_dim = True
        if halve_direction == "H":
            stride = [2, 1]
        elif halve_direction == "W":
            stride = [1, 2]
        elif halve_direction == "B":
            stride = [2, 2]
        else:
            raise ValueError("Wrong halve_direction: {}".format(halve_direction))
    elif input_channel == output_channel:
        increase_dim = False
        stride = [1, 1]
    else:
        raise ValueError('Output and input channel does not match in residual blocks!!!')'''
    # The first conv layer of the first residual block does not need to be normalized and relu-ed.
    with tf.variable_scope('conv1_in_block'):
        if first_block:
            filter_ = create_variables(name='conv', shape=[3, 3, input_channel, output_channel])
            conv1 = tf.nn.conv2d(input_layer, filter=filter_, strides=[1, 1, 1, 1], padding='SAME')
        else:
            conv1 = bn_relu_conv_layer(input_layer, [3, 3, input_channel, output_channel], stride, is_training)

    with tf.variable_scope('conv2_in_block'):
        conv2 = bn_relu_conv_layer(conv1, [3, 3, output_channel, output_channel], [1, 1], is_training)

    # When the channels of input layer and conv2 does not match, we add zero pads to increase the
    #  depth of input layers

    # channel weights
    with tf.variable_scope('cw_in_block'):
        e = tf.reduce_mean(conv2, [1, 2]) 
        with tf.variable_scope('fc1'):
            s = fc_layer(e, output_channel / 4)
        s = tf.nn.relu(s)
        with tf.variable_scope('fc2'):
            s = fc_layer(s, output_channel)
        s = tf.nn.sigmoid(s)
        s = tf.reshape(s, [-1, 1, 1, output_channel])

    scaled_conv2 = conv2 + conv2 * s

    padded_input = input_layer
    if stride != [1, 1]:
        padded_input = tf.nn.avg_pool(padded_input, ksize=[1, stride[0], stride[1], 1],
                                      strides=[1, stride[0], stride[1], 1], padding='SAME')
    if input_channel * 2 == output_channel:
        padded_input = tf.pad(padded_input, [[0, 0], [0, 0], [0, 0], [input_channel // 2,
                                                                      input_channel // 2]])
    output = scaled_conv2 + padded_input
    
    return output


def inference(input_tensor_batch, n, is_training):
    '''
    The main function that defines the ResNet. total layers = 1 + 2n + 2n + 2n +1 = 6n + 2
    :param input_tensor_batch: 4D tensor
    :param n: num_residual_blocks
    :param reuse: To build train graph, reuse=False. To build validation graph and share weights
    with train graph, resue=True
    :param is_training: if is in training
    :return: last layer in the network. Not softmax-ed
    '''

    layers = []
    with tf.variable_scope('conv0'):
        conv0 = conv_bn_relu_layer(input_tensor_batch, [7, 7, 3, 32], 1, is_training)
        #activation_summary(conv0)
        layers.append(conv0)

    for i in range(n):
        with tf.variable_scope('conv1_%d' % i):
            if i == 0:
                conv1 = residual_block(layers[-1], 64, [1, 1], is_training, first_block=True)
            else:
                conv1 = residual_block(layers[-1], 64, [1, 1], is_training)
            #activation_summary(conv1)
            layers.append(conv1)

    for i in range(n):
        with tf.variable_scope('conv2_%d' % i):
            conv2 = residual_block(layers[-1], 128, [2, 1], is_training)
            #activation_summary(conv2)
            layers.append(conv2)

    for i in range(n):
        with tf.variable_scope('conv3_%d' % i):
            conv3 = residual_block(layers[-1], 256, [2, 1], is_training)
            #activation_summary(conv3)
            layers.append(conv3)

    for i in range(n):
        with tf.variable_scope('conv4_%d' % i):
            conv4 = residual_block(layers[-1], 512, [2, 2], is_training)
            layers.append(conv4)
        assert conv4.get_shape().as_list()[1:] == [5, 6, 512]

    with tf.variable_scope('fc1'):
        #in_channel = layers[-1].get_shape().as_list()[-1]
        bn_layer = batch_normalization_layer(layers[-1], is_training)
        relu_layer = tf.nn.relu(bn_layer)
        global_pool = tf.reduce_mean(relu_layer, [1, 2])

        assert global_pool.get_shape().as_list()[-1:] == [512]
        fc1 = fc_layer(global_pool, 1024)
        layers.append(fc1)
    with tf.variable_scope('fc2'):
        #in_channel = layers[-1].get_shape().as_list()[-1]
        fc2 = batch_normalization_layer(layers[-1], is_training)
        fc2 = tf.nn.sigmoid(fc2)
        #fc2 = tf.nn.sigmoid(layers[-1])
        fc2 = fc_layer(fc2, 2048)
        layers.append(fc2)
    with tf.variable_scope('fc3'):
        #in_channel = layers[-1].get_shape().as_list()[-1]
        fc3 = batch_normalization_layer(layers[-1], is_training)
        fc3 = tf.nn.sigmoid(fc3)
        #fc3 = tf.nn.sigmoid(layers[-1])
        fc3 = fc_layer(fc3, 2048)
        layers.append(fc3)
    with tf.variable_scope('fc4'):
        #in_channel = layers[-1].get_shape().as_list()[-1]
        fc4 = batch_normalization_layer(layers[-1], is_training)
        fc4 = tf.nn.sigmoid(fc4)
        #fc4 = tf.nn.sigmoid(layers[-1])
        fc4 = fc_layer(fc4, 2048)
        layers.append(fc4)
    with tf.variable_scope('fc5'):
        #in_channel = layers[-1].get_shape().as_list()[-1]
        fc5 = batch_normalization_layer(layers[-1], is_training)
        fc5 = tf.nn.sigmoid(fc5)
        #fc5 = tf.nn.sigmoid(layers[-1])
        fc5 = fc_layer(fc5, 2048)
        layers.append(fc5)
    with tf.variable_scope('fc6'):
        #in_channel = layers[-1].get_shape().as_list()[-1]
        output = batch_normalization_layer(layers[-1], is_training)
        output = tf.nn.sigmoid(output)
        #output = tf.nn.sigmoid(layers[-1])
        output = fc_layer(output, NUM_CLASS)
        layers.append(output)

    return layers[-1]


def test_graph(train_dir='logs'):
    '''
    Run this function to look at the graph structure on tensorboard. A fast way!
    :param train_dir:
    '''
    input_tensor = tf.constant(np.ones([128, 32, 32, 3]), dtype=tf.float32)
    result = inference(input_tensor, 2, False)
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    #summary_writer = tf.train.SummaryWriter(train_dir, sess.graph)
