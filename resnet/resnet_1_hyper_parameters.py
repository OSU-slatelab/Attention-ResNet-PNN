import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

INPUT_WIDTH = 11
INPUT_HEIGHT = 40
INPUT_DEPTH = 3
NUM_CLASS = 1990
#NUM_CLASS = 1948
#NUM_CLASS = 1999

#tf.app.flags.DEFINE_string('featdir', '/data/data22/scratch/xusi/chime2/data/features/tr_denoised_actor_resnet_critic_wrbn_spec2spec',
#tf.app.flags.DEFINE_string('featdir', '/data/data22/scratch/xusi/chime2/data/features/tr',
#tf.app.flags.DEFINE_string('featdir', '/data/data20/scratch/xusi/chime3/data/features/tr_multi_beamformit_5mics',
tf.app.flags.DEFINE_string('featdir', '/data/data20/scratch/xusi/chime3/data/features/tr_multi_noisy',
                           '''Directory of the fetures for the train set''')
#tf.app.flags.DEFINE_string('aligdir', '/data/data20/scratch/xusi/chime3/data/labels/tr_multi_beamformit_5mics',
#tf.app.flags.DEFINE_string('aligdir', '/data/data22/scratch/xusi/chime2/data/labels/tr',
tf.app.flags.DEFINE_string('aligdir', '/data/data20/scratch/xusi/chime3/data/labels/tr_multi_noisy',
                           '''Directory of the labels for the train set''')
#tf.app.flags.DEFINE_string('vali_featdir', '/data/data20/scratch/xusi/chime3/data/features/cv_multi_beamformit_5mics',
#tf.app.flags.DEFINE_string('vali_featdir', '/data/data22/scratch/xusi/chime2/data/features/cv',
#tf.app.flags.DEFINE_string('vali_featdir', '/data/data22/scratch/xusi/chime2/data/features/cv_denoised_actor_resnet_critic_wrbn_spec2spec',
tf.app.flags.DEFINE_string('vali_featdir', '/data/data20/scratch/xusi/chime3/data/features/cv_multi_noisy',
                           '''Directory of the fetures for the cv set''')
#tf.app.flags.DEFINE_string('vali_aligdir', '/data/data22/scratch/xusi/chime2/data/labels/cv',
#tf.app.flags.DEFINE_string('vali_aligdir', '/data/data20/scratch/xusi/chime3/data/labels/cv_multi_beamformit_5mics',
tf.app.flags.DEFINE_string('vali_aligdir', '/data/data20/scratch/xusi/chime3/data/labels/cv_multi_noisy',
                           '''Directory of the labels for the cv set''')
tf.app.flags.DEFINE_boolean('train_shuffle_flag', True, 
                           '''Whether shuffle frames in the dispenser for training''')
tf.app.flags.DEFINE_boolean('valid_shuffle_flag', False, 
                           '''Whether shuffle frames in the dispenser for validation''')
#tf.app.flags.DEFINE_string('test_featdir', '/data/data22/scratch/xusi/chime2/data/features/test',
#tf.app.flags.DEFINE_string('test_featdir', '/data/data20/scratch/xusi/chime3/data/features/et_simu_beamformit_5mics',
tf.app.flags.DEFINE_string('test_featdir', '/data/data20/scratch/xusi/chime3/data/features/et_real_noisy',
                           '''Directory of the fetures for the train set''')
tf.app.flags.DEFINE_boolean('test_shuffle_flag', False, 
                           '''Whether shuffle frames in the dispenser for test''')
tf.app.flags.DEFINE_boolean('test_use_softmax', False, 
                           '''Whether use softmax in test''')


## The following flags are related to save paths, tensorboard outputs and screen outputs

tf.app.flags.DEFINE_string('version', 'models', '''A version number defining the directory to save
logs and checkpoints''')
tf.app.flags.DEFINE_integer('report_freq', 391, '''Steps takes to output errors on the screen
and write summaries''')
tf.app.flags.DEFINE_float('train_ema_decay', 0.999, '''The decay factor of the train error's
moving average shown on tensorboard''')


## The following flags define hyper-parameters regards training

tf.app.flags.DEFINE_integer('train_epoch', 30, '''Max training epochs''')
tf.app.flags.DEFINE_integer('train_steps', 80000, '''Total steps that you want to train''')
tf.app.flags.DEFINE_boolean('is_full_validation', False, '''Validation w/ full validation set or
a random batch''')
tf.app.flags.DEFINE_integer('train_batch_size', 128, '''Train batch size''')
tf.app.flags.DEFINE_integer('validation_batch_size', 128, '''Validation batch size, better to be
a divisor of 10000 for this task''')

tf.app.flags.DEFINE_integer('train_batch_size_multi_gpu', 128, '''Train batch size''')
tf.app.flags.DEFINE_integer('validation_batch_size_multi_gpu', 128, '''Validation batch size, better to be
a divisor of 10000 for this task''')

tf.app.flags.DEFINE_integer('test_batch_size', 128, '''Test batch size''')
tf.app.flags.DEFINE_integer('bulk_size', 1000, '''Test batch size''')
tf.app.flags.DEFINE_boolean('test_labels', False, '''Test batch size''')

tf.app.flags.DEFINE_float('init_lr_res', 0.008, '''Initial learning rate''')     # -----------------------------
tf.app.flags.DEFINE_float('init_lr_att', 0.008, '''Initial learning rate''')     # -----------------------------
tf.app.flags.DEFINE_float('lr_decay_factor', 0.5, '''How much to decay the learning rate each time''')
tf.app.flags.DEFINE_float('lr_decay_factor_attention', 0.1, '''How much to decay the learning rate each time''')  #-------------------------

tf.app.flags.DEFINE_integer('decay_step0', 40000, '''At which step to decay the learning rate''')
tf.app.flags.DEFINE_integer('decay_step1', 60000, '''At which step to decay the learning rate''')

tf.app.flags.DEFINE_integer('stop_training_bar', 0.001, '''At which step to decay the learning rate''')
tf.app.flags.DEFINE_integer('halving_lr_bar', 0.01, '''At which step to decay the learning rate''')

## The following flags define hyper-parameters modifying the training network

tf.app.flags.DEFINE_integer('num_residual_blocks', 5, '''How many residual blocks do you want''')
tf.app.flags.DEFINE_float('weight_decay', 0.0002, '''scale for l2 regularization''')


## The following flags are related to data-augmentation

tf.app.flags.DEFINE_integer('padding_size', 2, '''In data augmentation, layers of zero padding on
each side of the image''')


## If you want to load a checkpoint and continue training
tf.app.flags.DEFINE_string('train_dir', '/data/data20/scratch/xusi/chime3/exp/resnet_20_resblock/models_noenhan',
                           '''Directory to save models''')   #-------------------------------------
tf.app.flags.DEFINE_string('attention_train_dir', '/data/data20/scratch/xusi/chime3/exp/attention_resnet/models_noenhan',
                           '''Directory to save attention models''')   # ---------------------------------
tf.app.flags.DEFINE_string('train_dir_multi_gpu', '/data/data22/scratch/xusi/chime2/exp/resnet_20/models_multi_gpu',
                           '''Directory to save models''')   #-------------------------------------
tf.app.flags.DEFINE_string('ckpt_path', '/data/data20/scratch/xusi/chime3/exp/resnet_20_resblock/models_noenhan/model.ckpt-0',
                           '''Checkpoint directory to restore''')  # ------------------------------------
tf.app.flags.DEFINE_string('attention_ckpt_path',
                           '/data/data20/scratch/xusi/chime3/exp/attention_resnet/models_noenhan/model.ckpt-0',
                           '''Checkpoint path for attention resnet''')   # --------------------------------
tf.app.flags.DEFINE_string('multi_gpu_ckpt_path', '/data/data22/scratch/xusi/chime2/exp/resnet_20/models_multi_gpu/model.ckpt-19',
                           '''Checkpoint directory to restore''')  # ------------------------------------
tf.app.flags.DEFINE_boolean('is_use_ckpt', False,
                            '''Whether to load a checkpoint and continue training''')  #----------------------------------
tf.app.flags.DEFINE_boolean('is_use_ckpt_multi_gpu', False,
                            '''Whether to load a checkpoint and continue training''')  #----------------------------------


tf.app.flags.DEFINE_string('test_dir', '/data/data20/scratch/xusi/chime3/exp/attention_resnet/test_noenhan/decode_nosoftmax_real',
                           '''Directory for testing''')
tf.app.flags.DEFINE_string('test_ckpt_path',
                           '/data/data20/scratch/xusi/chime3/exp/attention_resnet/models_noenhan/model.ckpt-8',
                           '''Checkpoint path for testing attention resnet''')   # --------------------------------
#tf.app.flags.DEFINE_string('test_ckpt_path', 'model_110.ckpt-79999',
#                           '''Checkpoint directory to restore''')


