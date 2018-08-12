from resnet_1 import *
from datetime import datetime
import time, datetime
import os
import math
import argparse
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

    def placeholders(self):
        '''
        There are five placeholders in total.
        image_placeholder and label_placeholder are for train images and labels
        vali_image_placeholder and vali_label_placeholder are for validation imgaes and labels
        lr_placeholder is for learning rate. Feed in learning rate each time of training
        implements learning rate decay easily
        '''
        self.input_placeholder = tf.placeholder(dtype=tf.float32,
                                                shape=[FLAGS.train_batch_size, INPUT_HEIGHT,
                                                       INPUT_WIDTH, INPUT_DEPTH])
        self.label_placeholder = tf.placeholder(dtype=tf.int32, shape=[FLAGS.train_batch_size])

        #self.vali_input_placeholder = tf.placeholder(dtype=tf.float32, shape=[FLAGS.validation_batch_size,
        #                                                                      INPUT_HEIGHT, INPUT_WIDTH, INPUT_DEPTH])
        #self.vali_label_placeholder = tf.placeholder(dtype=tf.int32, shape=[FLAGS.validation_batch_size])

        self.lr_placeholder = tf.placeholder(dtype=tf.float32, shape=[])

        self.is_training_placeholder = tf.placeholder(dtype=tf.bool, shape=[])

        #self.reuse_placeholder = tf.placeholder(dtype=tf.bool, shape=[])

    def build_train_validation_graph(self):
        '''
        This function builds the train graph and validation graph at the same time.

        '''
        global_step = tf.Variable(0, trainable=False)
        validation_step = tf.Variable(0, trainable=False)

        # Logits of training data and valiation data come from the same graph. The inference of
        # validation data share all the weights with train data. This is implemented by passing
        # reuse=True to the variable scopes of train graph
        logits = inference(self.input_placeholder, FLAGS.num_residual_blocks, #self.reuse_placeholder,
                self.is_training_placeholder)
        #vali_logits = inference(self.vali_input_placeholder, FLAGS.num_residual_blocks, reuse=True, is_training=False)

        # The following codes calculate the train loss, which is consist of the
        # softmax cross entropy and the relularization loss
        #regu_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss = self.loss(logits, self.label_placeholder)
        #self.full_loss = tf.add_n([loss] + regu_losses)
        self.full_loss = loss

        predictions = tf.nn.softmax(logits)
        self.train_top1_error = self.top_k_error(predictions, self.label_placeholder, 1)

        # Validation loss
        #self.vali_loss = self.loss(vali_logits, self.vali_label_placeholder)
        self.vali_loss = loss
        #vali_predictions = tf.nn.softmax(vali_logits)
        #self.vali_top1_error = self.top_k_error(vali_predictions, self.vali_label_placeholder, 1)
        self.vali_top1_error = self.top_k_error(predictions, self.label_placeholder, 1)

        self.train_op, self.train_ema_op = self.train_operation(global_step, self.full_loss,
                                                                self.train_top1_error)
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
        lablreader = kaldiIO.LabelReader(aligdir + '/labels.scp')
        uid_list_lab = lablreader.get_uid_list()
        featreader.remove_lab_difference(uid_list_lab)
        # get the shuffled order of the feat list, and shuffle the lable list with the order
        scp_order = featreader.get_scp_order()
        lablreader.shuffle_utt(scp_order)

        vali_featdir = FLAGS.vali_featdir
        vali_aligdir = FLAGS.vali_aligdir

        vali_featreader = kaldiIO.TableReader(vali_featdir + '/feats_delta_5fr_sp_bi.scp', False)
        vali_lablreader = kaldiIO.LabelReader(vali_aligdir + '/labels.scp')

        vali_uid_list_feat = vali_featreader.get_uid_list()
        vali_lablreader.remove_utt_difference(vali_uid_list_feat)

        vali_uid_list_labl = vali_lablreader.get_uid_list()
        vali_featreader.remove_lab_difference(vali_uid_list_labl)

        vali_scp_order = vali_featreader.get_scp_order()
        vali_lablreader.shuffle_utt(vali_scp_order)
        # create a target coder
        # xsr6064 coder = target_coder.AlignmentCoder(lambda x, y: x, num_labels)

        dispenser = batchdispenser.BatchDispenser(featreader, lablreader, int(FLAGS.bulk_size),
                                                  int(FLAGS.train_batch_size), FLAGS.train_shuffle_flag)
        vali_dispenser = batchdispenser.BatchDispenser(vali_featreader, vali_lablreader, int(FLAGS.bulk_size),
                                                       int(FLAGS.validation_batch_size), FLAGS.valid_shuffle_flag)

        # Build the graph for train and validation
        self.build_train_validation_graph()

        # Initialize a saver to save checkpoints. Merge all summaries, so we can run all
        # summarizing operations by running summary_op. Initialize a new session
        saver = tf.train.Saver(tf.global_variables())
        #summary_op = tf.summary.merge_all()
        init = tf.initialize_all_variables()
        sess = tf.Session()

        # If you want to load from a checkpoint
        if FLAGS.is_use_ckpt is True:
            print ('Restoring from checkpoint... {}'.format(FLAGS.ckpt_path))
            saver.restore(sess, FLAGS.ckpt_path)
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

        best_model = 0   #--------------------------

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

        for epoch in range(0, FLAGS.train_epoch):   #---------------------------------
            end_epoch = False
            start_time = time.time()

            # Training
            iter = 0
            #for x in range(100):
            print ('Start training Epoch {}'.format(epoch))
            print ('The Learning rate of the epoch is: {}'.format(lr))
            while not end_epoch:
                train_batch_data, train_batch_labels, end_epoch = dispenser.get_batch()
                if train_batch_data == []:
                    continue
                train_batch_data = np.array(self.stack_batch(train_batch_data))
                assert (train_batch_data.shape == (int(FLAGS.train_batch_size), 40, 11, 3))

                _, _, train_loss_value, train_error_value = sess.run([self.train_op, self.train_ema_op,
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
                print("Epoch {}, Iter {}: training loss: {}, training error: {}".format(epoch, iter, train_loss_value, train_error_value))
                iter = iter + 1
                if math.isnan(train_loss_value):
                    break
            duration = time.time() - start_time
            featreader.reset()
            lablreader.reset(featreader.get_scp_order())
            dispenser.reset()
            print ("Time spend for epoch {}: {}".format(epoch, str(datetime.timedelta(seconds=duration))))
            print ('\n')

            # Check the validation
            # loss first
            if not math.isnan(train_loss_value):
                print ('Strat Validation ...')
                validation_loss_value_new, validation_error_value_new = self.full_validation(loss=self.vali_loss,
                                                                                             top1_error=self.vali_top1_error,
                                                                                             vali_dispenser=vali_dispenser,
                                                                                             #vali_labels=vali_labels,
                                                                                             session=sess
                                                                                             #batch_data=train_batch_data,
                                                                                             #batch_label=train_batch_labels
                                                                                            )
                vali_featreader.reset()
                vali_lablreader.reset(vali_featreader.get_scp_order())
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
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
            else:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt_reject')

            # Save checkpoints every epoch
            saver.save(sess, checkpoint_path, global_step=epoch)
            # save the training error to csv files
            #df = pd.DataFrame(data={'train_error': train_error_list})
            #df.to_csv(FLAGS.train_dir + '/' + str(epoch) + '_train_error.csv')

            if validation_loss_value_pre != validation_loss_value and \
               (validation_loss_value_pre - validation_loss_value) / validation_loss_value_pre \
                    < FLAGS.stop_training_bar and \
                lr != FLAGS.init_lr_res:
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
                ckpt_path = os.path.join(FLAGS.train_dir, 'model.ckpt-' + str(best_model))
                print ('Restoring from checkpoint: {}'.format(ckpt_path))
                saver.restore(sess, ckpt_path)
                print ('Restored from checkpoint ...')

        #df = pd.DataFrame(data={'epoch': step_list, 'validation_error': val_error_list})
        #df.to_csv(FLAGS.train_dir + '/' + str(epoch) + '_validation_error.csv')

        print ('Done training...')
        print ('----------------------------')


    def test(self):
        '''
        :return: the softmax probability with shape [num_test_images, num_labels]
        '''

        test_featdir = FLAGS.test_featdir
        #test_aligdir = FLAGS.aligdir

        test_featreader = kaldiIO.TableReader(test_featdir + '/feats_delta_5fr_sp_bi.scp')
        test_lablreader = None
        #test_lablreader = kaldiIO.LabelReader(aligdir + '/labels.scp')
        #uid_list_lab = test_lablreader.get_uid_list()
        #test_featreader.remove_lab_difference(uid_list_lab)
        # get the shuffled order of the feat list, and shuffle the lable list with the order
        #scp_order = test_featreader.get_scp_order()
        #test_lablreader.shuffle_utt(scp_order)

        dispenser = batchdispenser.TestBatchDispenser(test_featreader, FLAGS.test_labels, 
                                                      test_lablreader, int(FLAGS.test_batch_size))

        # Create the test image and labels placeholders
        self.test_input_placeholder = tf.placeholder(dtype=tf.float32, shape=[FLAGS.test_batch_size,
                                                     INPUT_HEIGHT, INPUT_WIDTH, INPUT_DEPTH])

        # Build the test graph
        logits = inference(self.test_input_placeholder, FLAGS.num_residual_blocks, 
                self.is_training_placeholder)
        #use_softmax = FLAGS.test_use_softmax
        #if use_softmax:
        #    logits = tf.nn.softmax(logits)

        # Initialize a new session and restore a checkpoint
        saver = tf.train.Saver(tf.all_variables())
        sess = tf.Session()

        saver.restore(sess, FLAGS.test_ckpt_path)
        print ('Model restored from ', FLAGS.test_ckpt_path)

        total_logits_array = []
        uids = []
        # Test by batches
        #zero_batch = np.zeros((128, 40, 11, 3))
        #zero_logit_array = sess.run(logits, feed_dict={self.test_input_placeholder: zero_batch,
        #                                               self.is_training_placeholder: False})
        #print ("zero_logit: {}".format(zero_logit_array))
        #print ("zero_logit_shape: {}".format(zero_logit_array.shape))
        #zero_pred = zero_logit_array[0]
        #print (zero_pred.shape)
        #print ("!!!!!!!!!!!!!!!!!!!!!!!!!!")
        count = 0
        for i in range(dispenser.num_utt()):
            utt_logits_array = np.array([]).reshape(-1, NUM_CLASS)
            num_frames, utt_id = dispenser.fetch_utt()
            num_batches = dispenser.num_batches()
            for step in range(num_batches):
                test_batch, _, _ = dispenser.get_batch()
                test_batch = np.array(self.stack_batch(test_batch))
                #print ("batch {}: {}".format(step, test_batch))
                #print (50 * "=")
                batch_logits_array = sess.run(logits,
                                              feed_dict={self.test_input_placeholder: test_batch,
                                                         self.is_training_placeholder: False})

                utt_logits_array = np.concatenate((utt_logits_array, batch_logits_array))
            print ("{}th utterance: {} -----------".format(i, utt_id))
            print ("num_frame: {}".format(num_frames))
            print ("utt_logits length: {}".format(utt_logits_array.shape[0]))
            print ("num_zeros_frames: {}".format(utt_logits_array[num_frames:].shape))
            #print (utt_logits_array)
            #print (50 * "-")
            #print (utt_logits_array[num_frames:])

            #for i in range(utt_logits_array[num_frames:].shape[0]):
            #    if not np.array_equal(zero_pred, utt_logits_array[num_frames:][i]):
            #        print ("@@@@@@@@@@@@@@@@@@@@@@@@@")

            utt_logits_array = utt_logits_array[:num_frames]
            print ("num_nonzeros_frames: {}".format(utt_logits_array.shape))
            uids.append(utt_id)
            total_logits_array.append(utt_logits_array)

        print ("num_utt: {}, {}".format(len(uids), len(total_logits_array)))

        with kaldiIO.TableWriter(FLAGS.test_dir + '/test_pred.scp', FLAGS.test_dir + '/test_pred.ark') as pred_writer:
            for uid, logits in zip(uids, total_logits_array):
                pred_writer.write(str(uid), logits)
        print ("---- Done testing ----")



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

    def train_operation(self, global_step, total_loss, top1_error):
        '''
        Defines train operations
        :param global_step: tensor variable with shape [1]
        :param total_loss: tensor with shape [1]
        :param top1_error: tensor with shape [1]
        :return: two operations. Running train_op will do optimization once. Running train_ema_op
        will generate the moving average of train error and train loss for tensorboard
        '''
        # Add train_loss, current learning rate and train error into the tensorboard summary ops
        #tf.summary.scalar('learning_rate', self.lr_placeholder)
        #tf.summary.scalar('train_loss', total_loss)
        #tf.summary.scalar('train_top1_error', top1_error)

        # The ema object help calculate the moving average of train loss and train error
        ema = tf.train.ExponentialMovingAverage(FLAGS.train_ema_decay, global_step)
        train_ema_op = ema.apply([total_loss, top1_error])
        #tf.summary.scalar('train_top1_error_avg', ema.average(top1_error))
        #tf.summary.scalar('train_loss_avg', ema.average(total_loss))

        #opt = tf.train.MomentumOptimizer(learning_rate=self.lr_placeholder, momentum=0.9)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            opt = tf.train.AdamOptimizer(learning_rate=self.lr_placeholder)
            train_op = opt.minimize(total_loss, global_step=global_step)
        return train_op, train_ema_op

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

        end_vali_set = False
        #for x in range(100):
        count = 0
        while not end_vali_set:
            count += 1
            vali_feat_batch, vali_labl_batch, end_vali_set = vali_dispenser.get_batch()
            if vali_feat_batch == []:
                continue
            vali_feat_batch = np.array(self.stack_batch(vali_feat_batch))
            #print (vali_feat_batch.shape)
            #print ([int(FLAGS.validation_batch_size), INPUT_HEIGHT, INPUT_WIDTH, INPUT_DEPTH])
            assert (vali_feat_batch.shape == (int(FLAGS.validation_batch_size), INPUT_HEIGHT, INPUT_WIDTH, INPUT_DEPTH))

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('train', 'test'), help='the mode to run the code')
    args = parser.parse_args()
    train = Train()
    if args.mode == 'train':
        # Initialize the Train object
        # Start the training session
        train.train()
    else:
        train.test()
