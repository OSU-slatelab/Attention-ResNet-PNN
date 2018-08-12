#import numpy as np
#from resnet_1_hyper_parameters import *
#from keras.layers.convolutional import UpSampling2D
from resnet_1 import *
import math

P = 1
T = 2
R = 1
MIN_RES = [5, 6]

#def donw_sampling (input_data, output_channel):

def attention_module(input_data, output_channel, scope, is_training):
    """
    :param input_data:
    :param output_channel:
    :param scope:
    :param is_training:
    :return:
    """
    with tf.variable_scope(scope):
        # pre_resnet layer
        pre_res = input_data
        for i in range(P):
            with tf.variable_scope('pre_res_block_{}'.format(i)):
                pre_res = residual_block(pre_res, output_channel, [1, 1], is_training)
        # the residual branch
        with tf.variable_scope('trunk_branch'):
            output_trunk = pre_res
            for i in range(T):
                with tf.variable_scope('res_block_{}'.format(i)):
                    output_trunk = residual_block(output_trunk, output_channel, [1, 1], is_training)
        # the attention branch
        with tf.variable_scope('attention_branch'):
            width = pre_res.get_shape().as_list()[1]
            height = pre_res.get_shape().as_list()[2]
            num_max_w = int(math.log(math.ceil(float(width) / MIN_RES[0]), 2))
            num_max_h = int(math.log(math.ceil(float(height) / MIN_RES[1]), 2))

            filters = []
            if num_max_w > num_max_h:
                filters = [[1, 2, 1, 1] for _ in range(num_max_w - num_max_h)]
            if num_max_h > num_max_w:
                filters = [[1, 1, 2, 1] for _ in range(num_max_w - num_max_h)]
            filters.extend([[1, 2, 2, 1] for _ in range(min(num_max_w, num_max_h))])

            assert (len(filters) == max(num_max_w, num_max_h))
            print ('filiters size: {}'.format(len(filters)))

            data_shape = []
            down_soft_mask_vec = []
            down_skip_connection_vec = []
            down_soft_mask = pre_res

            if len(filters) > 0:
                for i in range(len(filters)):
                    data_shape.append(down_soft_mask.get_shape().as_list())
                    print (scope + ' - Down sampling process {}'.format(i))
                    with tf.variable_scope('down_sampling_{}'.format(i)):
                        down_soft_mask = tf.nn.max_pool(down_soft_mask, ksize=filters[i], strides=filters[i],
                                                      padding='SAME')
                        for j in range(R):
                            with tf.variable_scope('down_block{}_res{}'.format(i, j)):
                                # TODO: add 'is_training' to residual_block function
                                down_soft_mask = residual_block(down_soft_mask, output_channel, [1, 1], is_training)
                        down_soft_mask_vec.append(down_soft_mask)
                    if i < len(filters) - 1:
                        with tf.variable_scope('skip_connection_{}'.format(i)):
                            down_skip_connection = residual_block(down_soft_mask, output_channel, [1, 1], is_training)
                            down_skip_connection_vec.append(down_skip_connection)
                assert (len(down_skip_connection_vec) == len(filters) - 1)
                assert (down_soft_mask_vec[-1].get_shape().as_list() == [input_data.get_shape().as_list()[0], 5, 6, output_channel])
                up_soft_mask = down_soft_mask_vec.pop()
                for i in range(len(filters)):
                    print (scope + ' - Up sampling process {}'.format(i))
                    with tf.variable_scope('up_sampling_{}'.format(i)):
                        for j in range(R):
                            with tf.variable_scope('up_block{}_res{}'.format(i, j)):
                                up_soft_mask = residual_block(up_soft_mask, output_channel, [1, 1], is_training)
                        filter_tmp = filters[-(1+i)]
                        size_new = data_shape.pop()[1:3]
                        print ('size_new: {}'.format(size_new))
                        up_soft_mask = tf.image.resize_images(up_soft_mask, size_new)

                        if j < len(down_skip_connection_vec):
                            up_soft_mask = up_soft_mask + down_skip_connection_vec.pop()
                            #up_soft_mask = up_soft_mask + down_soft_mask_vec.pop() #TODO: check to see if this adding is needed
                        #else:
                            #up_soft_mask = up_soft_mask # + output_trunk #TODO: maybe add res(input) instead of output+trunk
            else:
                with tf.variable_scope('mid_res_block_down'):
                    up_soft_mask = residual_block(down_soft_mask, output_channel, [1, 1], is_training)
                with tf.variable_scope('mid_res_block_up'):
                    up_soft_mask = residual_block(up_soft_mask, output_channel, [1, 1], is_training)
            print(up_soft_mask.get_shape().as_list())
            print(input_data.get_shape().as_list())
            assert (up_soft_mask.get_shape().as_list() == input_data.get_shape().as_list())

            with tf.variable_scope('mask_block'):
                with tf.variable_scope('conv1'):
                    output_attention = \
                        bn_relu_conv_layer(up_soft_mask,
                                           [1, 1, up_soft_mask.get_shape().as_list()[-1], output_channel],
                                           [1, 1], is_training)
                with tf.variable_scope('conv2'):
                    output_attention = \
                        bn_relu_conv_layer(output_attention,
                                           [1, 1, up_soft_mask.get_shape().as_list()[-1], output_channel],
                                           [1, 1], is_training)
                output_attention = tf.nn.sigmoid(output_attention)

        with tf.variable_scope('combined_output'):
            output_last = (1 + output_attention) * output_trunk
            with tf.variable_scope('res_block'):
                output_last = residual_block(output_last, output_channel, [1, 1], is_training)

        return output_last

def inference_attention_module(input_data, is_training):
    """
    :param input_data: 4D input tensor [NHWC]
    :param is_training: whether or not training process
    :return: An tensor with attention applied
    """
    with tf.variable_scope('conv0'):
        start_conv = conv_bn_relu_layer(input_data, [7, 7, 3, 32], 1, is_training)
        #activation_summary(start_conv)
    with tf.variable_scope('conv1'):
        input_channel = start_conv.get_shape().as_list()[-1]
        filter_ = create_variables(name='conv', shape=[3, 3, input_channel, 64])
        start_conv = tf.nn.conv2d(start_conv, filter=filter_, strides=[1, 1, 1, 1], padding='SAME')

    attention_block1 = attention_module(
        start_conv, 64, scope='attention_module1', is_training=is_training)
    with tf.variable_scope('between_res_block_1'):
        residual_block1 = residual_block(attention_block1, 128, [2, 1], is_training)
    attention_block2 = attention_module(
        residual_block1, 128, scope='attention_module2', is_training=is_training)
    with tf.variable_scope('between_res_block_2'):
        residual_block2 = residual_block(attention_block2, 256, [2, 1], is_training)
    attention_block3 = attention_module(
        residual_block2, 256, scope='attention_module3', is_training=is_training)
    with tf.variable_scope('between_res_block_3'):
        residual_block3 = residual_block(attention_block3, 512, [2, 2], is_training)
    attention_block4 = attention_module(
        residual_block3, 512, scope='attention_module4', is_training=is_training)
    assert attention_block4.get_shape().as_list()[1:] == [5, 6, 512]
    with tf.variable_scope('between_res_block_4'):
        residual_block4 = residual_block(attention_block4, 1024, [1, 1], is_training)

    with tf.variable_scope('fc1'):
        #in_channel = residual_block4.get_shape().as_list()[-1]
        bn_layer = batch_normalization_layer(residual_block4, is_training)
        relu_layer = tf.nn.relu(bn_layer)
        global_pool = tf.reduce_mean(relu_layer, [1, 2])

        assert global_pool.get_shape().as_list()[-1:] == [1024]
        fc1 = fc_layer(global_pool, 2048)

    with tf.variable_scope('fc2'):
        #in_channel = fc1.get_shape().as_list()[-1]
        fc2 = batch_normalization_layer(fc1, is_training)
        fc2 = tf.nn.sigmoid(fc2)
        fc2 = fc_layer(fc2, 2048)
    with tf.variable_scope('fc3'):
        #in_channel = fc2.get_shape().as_list()[-1]
        fc3 = batch_normalization_layer(fc2, is_training)
        fc3 = tf.nn.sigmoid(fc3)
        fc3 = fc_layer(fc3, 2048)
    with tf.variable_scope('fc4'):
        #in_channel = fc3.get_shape().as_list()[-1]
        fc4 = batch_normalization_layer(fc3, is_training)
        fc4 = tf.nn.sigmoid(fc4)
        fc4 = fc_layer(fc4, 2048)
    with tf.variable_scope('fc5'):
        #in_channel = fc4.get_shape().as_list()[-1]
        fc5 = batch_normalization_layer(fc4, is_training)
        fc5 = tf.nn.sigmoid(fc5)
        fc5 = fc_layer(fc5, 2048)
    with tf.variable_scope('fc6'):
        #in_channel = fc5.get_shape().as_list()[-1]
        output = batch_normalization_layer(fc5, is_training)
        output = tf.nn.sigmoid(output)
        output = fc_layer(output, NUM_CLASS)

    return output
