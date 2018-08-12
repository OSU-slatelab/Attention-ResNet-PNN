# Coder: Wenxin Xu
# Github: https://github.com/wenxinxu/resnet_in_tensorflow
# ==============================================================================

from resnet_1 import *
from datetime import datetime
from multiprocessing import Queue, Process
from threading import Thread
import time, datetime
import os
import math
# from resnet_1_input import *
from processing import batchdispenser, kaldiIO
#import pandas as pd


class Train(object):
    '''
    This Object is responsible for all the training and validation process
    '''

    def __init__(self):
        # Set up all the placeholders
        self.placeholders()
        self.end_epoch = False
        self.vali_end_epoch = False

    def placeholders(self):
        '''
        There are five placeholders in total.
        image_placeholder and label_placeholder are for train images and labels
        vali_image_placeholder and vali_label_placeholder are for validation imgaes and labels
        lr_placeholder is for learning rate. Feed in learning rate each time of training
        implements learning rate decay easily
        '''
        self.input_placeholder = tf.placeholder(dtype=tf.float32,
                                                shape=[FLAGS.train_batch_size_multi_gpu, INPUT_HEIGHT,
                                                       INPUT_WIDTH, INPUT_DEPTH])
        self.label_placeholder = tf.placeholder(dtype=tf.int32, shape=[FLAGS.train_batch_size_multi_gpu])

        #self.vali_input_placeholder = tf.placeholder(dtype=tf.float32, shape=[FLAGS.validation_batch_size,
        #                                                                      INPUT_HEIGHT, INPUT_WIDTH, INPUT_DEPTH])
        #self.vali_label_placeholder = tf.placeholder(dtype=tf.int32, shape=[FLAGS.validation_batch_size])

        self.lr_placeholder = tf.placeholder(dtype=tf.float32, shape=[])

        self.is_training_placeholder = tf.placeholder(dtype=tf.bool, shape=[])

        #self.reuse_placeholder = tf.placeholder(dtype=tf.bool, shape=[])

    def tower_loss(self, scope):
        """Calculate the total loss on a single tower running the CIFAR model.
        Args:
          scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'
        Returns:
           Tensor of shape [] containing the total loss for a batch of data
        """
        # Get images and labels for CIFAR-10.
        #images, labels = cifar10.distorted_inputs()

        # Build inference Graph.
        logits = inference(self.input_placeholder, FLAGS.num_residual_blocks, #self.reuse_placeholder,
                           self.is_training_placeholder)

        # Build the portion of the Graph calculating the losses. Note that we will
        # assemble the total_loss using a custom function below.
        loss = self.loss(logits, self.label_placeholder)
        #_ = cifar10.loss(logits, labels)

        # Assemble all of the losses for the current tower only.
        #losses = tf.get_collection('losses', scope)

        # Calculate the total loss for the current tower.
        #total_loss = tf.add_n(losses, name='total_loss')

        # Compute the moving average of all individual losses and the total loss.
        #loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
        #loss_averages_op = loss_averages.apply(losses + [total_loss])

        # Attach a scalar summary to all individual losses and the total loss; do the
        # same for the averaged version of the losses.
        #for l in losses + [total_loss]:
        #    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
        #    # session. This helps the clarity of presentation on tensorboard.
        #    loss_name = re.sub('%s_[0-9]*/' % cifar10.TOWER_NAME, '', l.op.name)
        #    # Name each loss as '(raw)' and name the moving average version of the loss
        #    # as the original loss name.
        #    tf.scalar_summary(loss_name +' (raw)', l)
        #    tf.scalar_summary(loss_name, loss_averages.average(l))

        #with tf.control_dependencies([loss_averages_op]):
        #    total_loss = tf.identity(total_loss)
        return loss

    def average_gradients(self, tower_grads):
        """Calculate the average gradient for each shared variable across all towers.
        Note that this function provides a synchronization point across all towers.
        Args:
          tower_grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.
        Returns:
           List of pairs of (gradient, variable) where the gradient has been averaged
           across all towers.
        """
        #print (type(tower_grads))
        #print (len(tower_grads))
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            #print (type(grad_and_vars))
            #print ("----------------------")
            #print (grad_and_vars)
            #print ("----------------------")
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                #print ("----------------------")
                #print (g)
                #print ("----------------------")
                #print (xx)
                #print ("----------------------")
                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads

    def queue_batch(self, queue, dispenser, test_flag):
        while True:
            if not queue.full() and not self.end_epoch:
                #print ("aaaaaaaaaa")
                batch_data, batch_labels, end_epoch = dispenser.get_batch()
                if not end_epoch:
                    batch_data = np.array(self.stack_batch(batch_data))
                    assert (batch_data.shape == (int(FLAGS.train_batch_size_multi_gpu), 40, 11, 3))
                    queue.put((batch_data, batch_labels, end_epoch))
                else:
                    print ("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    if not test_flag:
                        self.end_epoch = True
                    else:
                        self.vali_end_epoch = True
                    dispenser.reset()

    #def queue_batch_valid(self, queue, dispenser):
    #    batch_data, batch_labels, end_epoch = dispenser.get_batch()
    #    batch_data = np.array(self.stack_batch(batch_data))
    #    assert (batch_data.shape == (int(FLAGS.train_batch_size_multi_gpu), 40, 11, 3)) 
    #    queue.put((batch_data, batch_labels, end_epoch))
    #    if end_epoch:
    #        dispenser.reset()

    def build_train_validation_graph(self):
        '''
        This function builds the train graph and validation graph at the same time.

        '''
        global_step = tf.Variable(0, trainable=False)
        validation_step = tf.Variable(0, trainable=False)

        # Logits of training data and valiation data come from the same graph. The inference of
        # validation data share all the weights with train data. This is implemented by passing
        # reuse=True to the variable scopes of train graph
        opt = tf.train.AdamOptimizer(learning_rate=self.lr_placeholder)

        tower_grads = []
        tower_loss = []
        tower_train_error = []
        with tf.variable_scope("multi_gpu_graph"):
            for i in range(2):   # number of GPU
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % ('tower', i)) as scope:
                        logits = inference(self.input_placeholder, FLAGS.num_residual_blocks, #self.reuse_placeholder,
                                           self.is_training_placeholder)
                        #vali_logits = inference(self.vali_input_placeholder, FLAGS.num_residual_blocks, reuse=True, is_training=False)

                        # The following codes calculate the train loss, which is consist of the
                        # softmax cross entropy and the relularization loss
                        #regu_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                        loss = self.loss(logits, self.label_placeholder)
                        tower_loss.append(loss)
                        
                        tf.get_variable_scope().reuse_variables()

                        predictions = tf.nn.softmax(logits)
                        train_top1_error = self.top_k_error(predictions, self.label_placeholder, 1)
                        tower_train_error.append(train_top1_error)

                        grads = opt.compute_gradients(loss)
                        tower_grads.append(grads)
                        #self.full_loss = tf.add_n([loss] + regu_losses)
        #self.full_loss = loss
        self.train_top1_error = tf.reduce_mean(tower_train_error)
        loss = tf.reduce_mean(tower_loss)
        grads = self.average_gradients(tower_grads)


        # Validation loss
        #self.vali_loss = self.loss(vali_logits, self.vali_label_placeholder)
        self.full_loss = loss
        self.vali_loss = loss
        #vali_predictions = tf.nn.softmax(vali_logits)
        #self.vali_top1_error = self.top_k_error(vali_predictions, self.vali_label_placeholder, 1)
        self.vali_top1_error = self.train_top1_error

        tmp_count = 0
        #for grad, var in grads:
        #    print ("{}: grad: {}".format(tmp_count, grad))
        #    print ("{}: var : {}".format(tmp_count, var))
        #    tmp_count += 1

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            #opt = tf.train.AdamOptimizer(learning_rate=self.lr_placeholder)
            self.train_op = opt.apply_gradients(grads, global_step=global_step)
        #self.train_op = self.train_operation(opt, global_step, grads, self.train_top1_error)
        #self.val_op = self.validation_op(validation_step, self.vali_top1_error, self.vali_loss)

    def train(self):
        '''
        This is the main function for training
        '''

        # For the first step, we are loading all training images and validation images into the
        # memory
        # all_data, all_labels = prepare_train_data(padding_size=FLAGS.padding_size)
        # vali_data, vali_labels = read_validation_data()

        # create a feature reader
        featdir = FLAGS.featdir
        aligdir = FLAGS.aligdir

        featreader = kaldiIO.TableReader(featdir + '/feats_delta_5fr_sp_bi.scp')
        lablreader = kaldiIO.LabelReader(aligdir + '/clean_lab.scp')
        uid_list_lab = lablreader.get_uid_list()
        featreader.remove_lab_difference(uid_list_lab)
        # get the shuffled order of the feat list, and shuffle the lable list with the order
        scp_order = featreader.get_scp_order()
        lablreader.shuffle_utt(scp_order)

        vali_featdir = FLAGS.vali_featdir
        vali_aligdir = FLAGS.vali_aligdir

        vali_featreader = kaldiIO.TableReader(vali_featdir + '/feats_delta_5fr_sp_bi.scp', False)
        vali_lablreader = kaldiIO.LabelReader(vali_aligdir + '/clean_lab.scp')
        vali_uid_list_feat = vali_featreader.get_uid_list()
        vali_lablreader.remove_utt_difference(vali_uid_list_feat)

        vali_scp_order = vali_featreader.get_scp_order()
        vali_lablreader.shuffle_utt(vali_scp_order)
        # create a target coder
        # xsr6064 coder = target_coder.AlignmentCoder(lambda x, y: x, num_labels)
        print ("9999999999999999999999999")

        dispenser = batchdispenser.BatchDispenser(featreader, lablreader, int(FLAGS.bulk_size),
                                                  int(FLAGS.train_batch_size_multi_gpu), FLAGS.train_shuffle_flag)
        vali_dispenser = batchdispenser.BatchDispenser(vali_featreader, vali_lablreader, int(FLAGS.bulk_size),
                                                       int(FLAGS.validation_batch_size_multi_gpu), FLAGS.valid_shuffle_flag)
        data_queue = Queue(100)
        vali_data_queue = Queue(100)

        p_train = Process(target=self.queue_batch, args=(data_queue, dispenser, False))
        p_train.daemon = True
        p_valid = Process(target=self.queue_batch, args=(vali_data_queue, vali_dispenser, True))
        p_valid.daemon = True

        print ("000000000000000000000")

        p_train.start()
        p_valid.start()

        print ("1111111111111111111111")

        # Build the graph for train and validation
        self.build_train_validation_graph()

        print ("22222222222222222222222")

        # Initialize a saver to save checkpoints. Merge all summaries, so we can run all
        # summarizing operations by running summary_op. Initialize a new session
        saver = tf.train.Saver(tf.global_variables())
        #summary_op = tf.summary.merge_all()
        init = tf.initialize_all_variables()
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))

        # If you want to load from a checkpoint
        if FLAGS.is_use_ckpt_multi_gpu is True:
            print ('Restoring from checkpoint... {}'.format(FLAGS.multi_gpu_ckpt_path))
            saver.restore(sess, FLAGS.multi_gpu_ckpt_path)
            print ('Restored model ...')
        else:
            sess.run(init)

        # This summary writer object helps write summaries on tensorboard
        #summary_writer = tf.summary.FileWriter(FLAGS.train_dir + '/log', sess.graph)

        # These lists are used to save a csv file at last
        #step_list = []
        #train_error_list = []
        #val_error_list = []

        print ('Start training...')
        print ('----------------------------')

        best_model = 0

        #validation_loss_value, validation_error_value = self.full_validation(loss=self.vali_loss,
        #                                                                     top1_error=self.vali_top1_error,
        #                                                                     vali_dispenser=vali_dispenser,
        #                                                                     session=sess)
        #                                                                     #batch_data=train_batch_data,
        #                                                                     #batch_label=train_batch_labels
        #vali_featreader.reset()
        #vali_lablreader.reset(vali_featreader.get_scp_order())
        #vali_dispenser.reset()
        validation_loss_value = 10000
        validation_error_value = 10000

        #_, _, _ = vali_dispenser.get_batch()
        print ("Starting Validation Loss: {} | Staring Validation Error: {}".format(validation_loss_value, validation_error_value))
        print ('------------------------------------------------------------')
        print('\n')

        lr = FLAGS.init_lr_res
        half_lr = 0   #--------------------------------------------
        #stop_training = False

        for epoch in range(FLAGS.train_epoch):   #---------------------------------
            self.end_epoch = False
            start_time = time.time()

            # Training
            iter = 0
            #for x in range(100):
            print ('Start training Epoch {}'.format(epoch))
            print ('The Learning rate of the epoch is: {}'.format(lr))
            while not self.end_epoch or not data_queue.empty():
                #train_batch_data, train_batch_labels, end_epoch = dispenser.get_batch()
                train_batch_data, train_batch_labels, end_epoch = data_queue.get()
                if train_batch_data == []:
                    continue
                #train_batch_data = np.array(self.stack_batch(train_batch_data))
                #assert (train_batch_data.shape == (int(FLAGS.train_batch_size_multi_gpu), 40, 11, 3))

                _, train_loss_value, train_error_value = sess.run([self.train_op, #self.train_ema_op,
                                                                      self.full_loss, self.train_top1_error],
                                                                     {self.input_placeholder: train_batch_data,
                                                                      self.label_placeholder: train_batch_labels,
                                                                      #self.vali_input_placeholder: validation_batch_data,
                                                                      #self.vali_label_placeholder: validation_batch_labels,
                                                                      #self.reuse_placeholder: False,
                                                                      self.is_training_placeholder: True,
                                                                      self.lr_placeholder: lr}
                                                                     )
                #train_error_list.append(train_error_value)
                print("Epoch {}, Iter {}: training loss: {}, training error: {}, end_epoch: {}".format(epoch, iter,
                    train_loss_value, train_error_value, end_epoch))
                iter = iter + 1
                if math.isnan(train_loss_value):
                    break
            duration = time.time() - start_time
            #featreader.reset()
            #lablreader.reset(featreader.get_scp_order())
            #dispenser.reset()
            print ("Time spend for epoch {}: {}".format(epoch, str(datetime.timedelta(seconds=duration))))
            print ('\n')

            # Check the validation
            # loss first
            if not math.isnan(train_loss_value):
                print ('Strat Validation ...')
                validation_loss_value_new, validation_error_value_new = self.full_validation(loss=self.vali_loss,
                                                                                             top1_error=self.vali_top1_error,
                                                                                             vali_dispenser=vali_data_queue,
                                                                                             #vali_dispenser=vali_dispenser,
                                                                                             #vali_labels=vali_labels,
                                                                                             session=sess
                                                                                             #batch_data=train_batch_data,
                                                                                             #batch_label=train_batch_labels
                                                                                            )
                #vali_featreader.reset()
                #vali_lablreader.reset(vali_featreader.get_scp_order())
                vali_dispenser.reset()
            else:
                validation_loss_value_new = 10000
                validation_error_value_new = 10000

            print ("Validation Loss: {} | Validation Error: {}".format(validation_loss_value_new,
                    validation_error_value_new))

            #vali_summ = tf.Summary()
            #vali_summ.value.add(tag='full_validation_error',
            #                    simple_value=validation_error_value.astype(np.float))
            #summary_writer.add_summary(vali_summ, epoch)
            #summary_writer.flush()

            #val_error_list.append(validation_error_value)

            #train_batch_data, train_batch_labels, _ = dispenser.get_batch()
            #train_batch_data = np.array(self.stack_batch(train_batch_data))

            #validation_batch_data, validation_batch_labels, _ = vali_dispenser.get_batch()
            #validation_batch_data = np.array(self.stack_batch(validation_batch_data))

            #summary_str = sess.run(summary_op, {self.input_placeholder: train_batch_data,
            #                                    self.label_placeholder: train_batch_labels,
                                                #self.vali_input_placeholder: validation_batch_data,
                                                #self.vali_label_placeholder: validation_batch_labels,
            #                                    self.lr_placeholder: lr})
            #summary_writer.add_summary(summary_str, epoch)

            print ('Validation top1 error = %.4f' % validation_error_value_new)
            print ('Validation loss = ', validation_loss_value_new)
            print ('----------------------------')

            #step_list.append(epoch)

            validation_loss_value_pre = validation_loss_value

            if validation_loss_value_new < validation_loss_value:
                validation_loss_value = validation_loss_value_new
                best_model = epoch
                checkpoint_path = os.path.join(FLAGS.train_dir_multi_gpu, 'model.ckpt')
            else:
                checkpoint_path = os.path.join(FLAGS.train_dir_multi_gpu, 'model.ckpt_reject')

            # Save checkpoints every epoch
            saver.save(sess, checkpoint_path, global_step=epoch)
            # save the training error to csv files
            #df = pd.DataFrame(data={'train_error': train_error_list})
            df.to_csv(FLAGS.train_dir_multi_gpu + '/' + str(epoch) + '_train_error.csv')

            if validation_loss_value_pre != validation_loss_value and \
               (validation_loss_value_pre - validation_loss_value) / validation_loss_value_pre \
                    < FLAGS.stop_training_bar and \
                lr != FLAGS.init_lr:
                print ("(Vali_loss_pre - Vali_loss) / Vali_loss_pre: ({} - {}) / {} = {}".format(validation_loss_value_pre,
                    validation_loss_value, validation_loss_value_pre, (validation_loss_value_pre -
                        validation_loss_value) / validation_loss_value_pre))
                print ('Stop training: Validation Loss decrease is too small')
                break
            if (validation_loss_value_pre - validation_loss_value) / validation_loss_value_pre \
                    < FLAGS.halving_lr_bar:
                half_lr = 1
            if half_lr == 1:
                lr = FLAGS.lr_decay_factor * lr
                print ("(Vali_loss_pre - Vali_loss) / Vali_loss_pre: ({} - {}) / {} = {}".format(validation_loss_value_pre,
                    validation_loss_value, validation_loss_value_pre, (validation_loss_value_pre -
                        validation_loss_value) / validation_loss_value_pre))
                print ('Learning rate decayed to ', lr) 
            print ('------------------------------------------')
            print ('\n')

            if best_model != epoch:
                ckpt_path = os.path.join(FLAGS.train_dir_multi_gpu, 'model.ckpt-' + str(best_model))
                print ('Restoring from checkpoint: {}'.format(ckpt_path))
                saver.restore(sess, ckpt_path)
                print ('Restored from checkpoint ...')

        #df = pd.DataFrame(data={'epoch': step_list, 'validation_error': val_error_list})
        #df.to_csv(FLAGS.train_dir + '/' + str(epoch) + '_validation_error.csv')

        p_train.terminate()
        p_valid.terminate()

        print ('Done training...')
        print ('----------------------------')

    def test(self, test_image_array):
        '''
        This function is used to evaluate the test data. Please finish pre-precessing in advance
        :param test_image_array: 4D numpy array with shape [num_test_images, img_height, img_width,
        img_depth]
        :return: the softmax probability with shape [num_test_images, num_labels]
        '''
        num_test_images = len(test_image_array)
        num_batches = num_test_images // FLAGS.test_batch_size
        remain_images = num_test_images % FLAGS.test_batch_size
        print ('%i test batches in total...' % num_batches)

        # Create the test image and labels placeholders
        self.test_input_placeholder = tf.placeholder(dtype=tf.float32, shape=[FLAGS.test_batch_size,
                                                                              INPUT_HEIGHT, INPUT_WIDTH, INPUT_DEPTH])

        # Build the test graph
        logits = inference(self.test_input_placeholder, FLAGS.num_residual_blocks,
                is_training=self.is_training_placeholder)
        predictions = tf.nn.softmax(logits)

        # Initialize a new session and restore a checkpoint
        saver = tf.train.Saver(tf.all_variables())
        sess = tf.Session()

        saver.restore(sess, FLAGS.test_ckpt_path)
        print ('Model restored from ', FLAGS.test_ckpt_path)

        prediction_array = np.array([]).reshape(-1, NUM_CLASS)
        # Test by batches
        for step in range(num_batches):
            if step % 10 == 0:
                print ('%i batches finished!' % step)
            offset = step * FLAGS.test_batch_size
            test_image_batch = test_image_array[offset:offset + FLAGS.test_batch_size, ...]

            batch_prediction_array = sess.run(predictions,
                                              feed_dict={self.test_input_placeholder: test_image_batch})

            prediction_array = np.concatenate((prediction_array, batch_prediction_array))

        # If test_batch_size is not a divisor of num_test_images
        if remain_images != 0:
            self.test_input_placeholder = tf.placeholder(dtype=tf.float32, shape=[remain_images,
                                                                                  INPUT_HEIGHT, INPUT_WIDTH, INPUT_DEPTH])
            # Build the test graph
            logits = inference(self.test_input_placeholder, FLAGS.num_residual_blocks,
                    is_training=self.is_training_placeholder)
            predictions = tf.nn.softmax(logits)

            test_image_batch = test_image_array[-remain_images:, ...]

            batch_prediction_array = sess.run(predictions, feed_dict={
                self.test_input_placeholder: test_image_batch})

            prediction_array = np.concatenate((prediction_array, batch_prediction_array))

        return prediction_array

    ## Helper functions
    def loss(self, logits, labels):
        '''
        Calculate the cross entropy loss given logits and true labels
        :param logits: 2D tensor with shape [batch_size, num_labels]
        :param labels: 1D tensor with shape [batch_size]
        :return: loss tensor with shape [1]
        '''
        labels = tf.cast(labels, tf.int64)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                       labels=labels, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        return cross_entropy_mean

    def top_k_error(self, predictions, labels, k):
        '''
        Calculate the top-k error
        :param predictions: 2D tensor with shape [batch_size, num_labels]
        :param labels: 1D tensor with shape [batch_size, 1]
        :param k: int
        :return: tensor with shape [1]
        '''
        batch_size = predictions.get_shape().as_list()[0]
        in_top1 = tf.to_float(tf.nn.in_top_k(predictions, labels, k=1))
        num_correct = tf.reduce_sum(in_top1)
        return (batch_size - num_correct) / float(batch_size)

    def generate_vali_batch(self, vali_data, vali_label, vali_batch_size):
        '''
        If you want to use a random batch of validation data to validate instead of using the
        whole validation data, this function helps you generate that batch
        :param vali_data: 4D numpy array
        :param vali_label: 1D numpy array
        :param vali_batch_size: int
        :return: 4D numpy array and 1D numpy array
        '''
        offset = np.random.choice(10000 - vali_batch_size, 1)[0]
        vali_data_batch = vali_data[offset:offset + vali_batch_size, ...]
        vali_label_batch = vali_label[offset:offset + vali_batch_size]
        return vali_data_batch, vali_label_batch

    def train_operation(self, opt, global_step, grads, top1_error):
        '''
        Defines train operations
        :param opt: the optimizer
        :param global_step: tensor variable with shape [1]
        :param grads: gradient
        :param top1_error: tensor with shape [1]
        :return: two operations. Running train_op will do optimization once. Running train_ema_op
        will generate the moving average of train error and train loss for tensorboard
        '''
        # Add train_loss, current learning rate and train error into the tensorboard summary ops
        #tf.summary.scalar('learning_rate', self.lr_placeholder)
        #tf.summary.scalar('train_loss', total_loss)
        #tf.summary.scalar('train_top1_error', top1_error)

        # The ema object help calculate the moving average of train loss and train error
        #ema = tf.train.ExponentialMovingAverage(FLAGS.train_ema_decay, global_step)
        #train_ema_op = ema.apply([total_loss, top1_error])
        #tf.summary.scalar('train_top1_error_avg', ema.average(top1_error))
        #tf.summary.scalar('train_loss_avg', ema.average(total_loss))

        #opt = tf.train.MomentumOptimizer(learning_rate=self.lr_placeholder, momentum=0.9)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            #opt = tf.train.AdamOptimizer(learning_rate=self.lr_placeholder)
            train_op = opt.apply_gradients(grads, global_step=global_step)
        return train_op#, train_ema_op

    def validation_op(self, validation_step, top1_error, loss):
        '''
        Defines validation operations
        :param validation_step: tensor with shape [1]
        :param top1_error: tensor with shape [1]
        :param loss: tensor with shape [1]
        :return: validation operation
        '''

        # This ema object help calculate the moving average of validation loss and error

        # ema with decay = 0.0 won't average things at all. This returns the original error
        ema = tf.train.ExponentialMovingAverage(0.0, validation_step)
        ema2 = tf.train.ExponentialMovingAverage(0.95, validation_step)

        val_op = tf.group(validation_step.assign_add(1), ema.apply([top1_error, loss]),
                          ema2.apply([top1_error, loss]))
        top1_error_val = ema.average(top1_error)
        top1_error_avg = ema2.average(top1_error)
        loss_val = ema.average(loss)
        loss_val_avg = ema2.average(loss)

        # Summarize these values on tensorboard
        #tf.summary.scalar('val_top1_error', top1_error_val)
        #tf.summary.scalar('val_top1_error_avg', top1_error_avg)
        #tf.summary.scalar('val_loss', loss_val)
        #tf.summary.scalar('val_loss_avg', loss_val_avg)
        return val_op

    def full_validation(self, loss, top1_error, vali_dispenser, session):
        '''
        Runs validation on all the 10000 valdiation images
        :param loss: tensor with shape [1]
        :param top1_error: tensor with shape [1]
        :param session: the current tensorflow session
        :param vali_data: 4D numpy array
        :param vali_labels: 1D numpy array
        :param batch_data: 4D numpy array. training batch to feed dict and fetch the weights
        :param batch_label: 1D numpy array. training labels to feed the dict
        :return: float, float
        '''
        #num_batches = 10000 // FLAGS.validation_batch_size
        #order = np.random.choice(10000, num_batches * FLAGS.validation_batch_size)
        #vali_data_subset = vali_data[order, ...]
        #vali_labels_subset = vali_labels[order]

        loss_list = []
        error_list = []

        self.vali_end_epoch = False
        #for x in range(100):
        count = 0
        while not self.vali_end_epoch and not vali_dispenser.empty():
            count += 1
            #vali_feat_batch, vali_labl_batch, end_vali_set = vali_dispenser.get_batch()
            vali_feat_batch, vali_labl_batch, end_vali_set = vali_dispenser.get()
            if vali_feat_batch == []:
                continue
            #vali_feat_batch = np.array(self.stack_batch(vali_feat_batch))
            #print (vali_feat_batch.shape)
            #print ([int(FLAGS.validation_batch_size), INPUT_HEIGHT, INPUT_WIDTH, INPUT_DEPTH])
            #assert (vali_feat_batch.shape == (int(FLAGS.validation_batch_size), INPUT_HEIGHT, INPUT_WIDTH, INPUT_DEPTH))

            #offset = step * FLAGS.validation_batch_size
            feed_dict = {#self.input_placeholder: batch_data, self.label_placeholder: batch_label,
                         self.input_placeholder: vali_feat_batch,
                         self.label_placeholder: vali_labl_batch,
                         #self.reuse_placeholder: True,
                         self.is_training_placeholder: False}
            loss_value, top1_error_value = session.run([loss, top1_error], feed_dict=feed_dict)
            loss_list.append(loss_value)
            error_list.append(top1_error_value)
            print ('Iter {} - Validation Batch Loss: {} | Validation Batch Error: {}'.format(count, loss_value, top1_error_value))

        return np.mean(loss_list), np.mean(error_list)


    def stack_batch(self, input):
        new_batch = []
        num_frames = input.shape[0]
        for i in range(num_frames):
            new_batch.append(self.reshape_frame(input[i], INPUT_HEIGHT, INPUT_DEPTH))
        return np.array(new_batch)

    def reshape_frame(self, frame, width, num_slice):
        res = [[] for _ in range(width)]
        len = frame.shape[0] 
        stride = width * num_slice
        height = int(len / stride)
        for i in range(height):
            tmp = frame[i * stride: (i + 1) * stride]
            for j in range(width):
                tup = []
                for k in range(num_slice):
                    tup.append(tmp[k * width + j])
                res[j].append(tup)
        return np.array(res)

# Initialize the Train object
train = Train()
# Start the training session
train.train()
