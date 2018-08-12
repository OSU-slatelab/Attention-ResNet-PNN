'''
#@package batchdispenser
# contain the functionality for read features and batches
# of features for neural network training and testing
'''

from abc import ABCMeta, abstractmethod
from random import shuffle
from processing import kaldiIO
import gzip
import logging
import numpy as np

logging.basicConfig()
log = logging.getLogger("kaldi_batchdispenser")
log.setLevel(logging.DEBUG)


## Class that dispenses batches of data for mini-batch training
class BatchDispenser(object):
    ''' BatchDispenser interface cannot be created but gives methods to its
    child classes.'''
    __metaclass__ = ABCMeta

    '''
    @abstractmethod
    def read_target_file(self, target_path):
        """
        read the file containing the targets

        Args:
            target_path: path to the targets file

        Returns:
            A dictionary containing
                - Key: Utterance ID
                - Value: The target sequence as a string
        """
    '''

    def __init__(self, feature_reader, label_reader, utt_bulk_size, mini_batch_size = 128, shuffle_flag = True):
        """
        Abstract constructor for nonexisting general data sets.

        Args:
            feature_reader: Kaldi ark-file feature reader instance.
            target_coder: a TargetCoder object to encode and decode the target
                sequences
            utt_bulk_size: Specifies how many utterances should be read in
                  each time.
            mini_batch_size: Specifies how many frames in each mini_batch
                    (will return a mini_batch each time calling get_batch() )
            shuffle_flag: whether shuffle the frame in each bulk
        """

        # Store the feature reader and label reader
        self.feature_reader = feature_reader
        self.label_reader = label_reader

        # Get a dictionary connecting training utterances and targets.
        # xsr6064 self.target_dict = self.read_target_file(target_path)

        # Detect the maximum length of the target sequences
        # xsr 6064 self.max_target_length = max([target_coder.encode(targets).size
        #                              for targets in self.target_dict.values()])

        # Store the number of utterances to be read in each time
        self.utt_bulk_size = utt_bulk_size

        # Store the mini_batch_size
        self.mini_batch_size = mini_batch_size

        # Whether shuffle frames in each bulk
        self.shuffle_flag = shuffle_flag

        # Lists where the feature and targets of the current bulk are stored
        self.feature_bulk = []
        self.target_bulk = []

        self.end_scp = False
        self.end_epoch = False

    def reset(self):
        self.feature_reader.reset()
        self.label_reader.reset(self.feature_reader.get_scp_order())
        self.end_scp = False
        self.end_epoch = False

    def shuffle_frames(self):
        """
        Shuffle the order of the frames in the current bulk
        """
        tmp_bulk = list(zip(self.feature_bulk, self.target_bulk))
        shuffle(tmp_bulk)
        self.feature_bulk, self.target_bulk = zip(*tmp_bulk)
        self.feature_bulk = np.asanyarray(self.feature_bulk)
        self.target_bulk = np.asarray(self.target_bulk)

    def finish_bulk(self):
        """
        Test if the bulk has reached reached the end

        Return:
            True/False
        """
        return len(self.feature_bulk) < self.mini_batch_size

    def fetch_val_set(self, val_set_size):
        '''
        :param val_set_size: number of utterance in the
        :return: the features and targets of val set as numpy array (frame shuffled)
        '''

        val_feat = []
        val_targets = []
        while len(val_feat) < val_set_size:
            utt_id, utt_mat, looped = self.feature_reader.fetch()
            if self.label_reader.valid_uid(utt_id) and utt_mat is not None:
                targets = self.label_reader.fetch(utt_id)
                val_feat.append(utt_mat)
                val_targets.append(targets)
            else:
                if not self.label_reader.valid_uid(utt_id):
                    print ('WARNING no targets for %s' % utt_id)

                if utt_mat is None:
                    print ('WARNING %s is an empty utterance' % utt_id)
            self.feature_reader.remove(utt_id)
            self.label_reader.remove(utt_id)
        val_feat = np.concatenate(val_feat)
        val_targets = np.concatenate(val_targets)
        tmp = list(zip(val_feat, val_targets))
        shuffle(tmp)
        val_feat, val_targets = zip(*tmp)

        return np.asarray(val_feat), np.asarray(val_targets)

    def fetch_utt_bulk(self, shuffle=True):
        '''
        Read in utt_bulk_size number of utterances and concatenate them to previous features if any remains.
        All frames read in will be concatenated and shuffled.
        '''

        #set up the data lists.
        bulk_inputs = []
        bulk_targets = []

        while len(bulk_inputs) < self.utt_bulk_size:
            #read utterance
            utt_id, utt_mat, looped = self.feature_reader.fetch()

            #get transcription
            if self.label_reader.valid_uid(utt_id) and utt_mat is not None:
                _, targets, _ = self.label_reader.fetch(utt_id)
                assert (len(utt_mat) == len(targets))
                # xsr6064 encoded_targets = self.target_coder.encode(targets)
                bulk_inputs.append(utt_mat)
                bulk_targets.append(targets)
            else:
                if not self.label_reader.valid_uid(utt_id):
                    print ('WARNING no targets for %s' % utt_id)

                if utt_mat is None:
                    print ('WARNING %s is too short to splice' % utt_id)

            if looped:
                self.end_scp = True
                break

        # bulk_inputs and bulk_targets are lists of ndarrays.
        # Flatten bulk_inputs and bulk_targets into one ndarray which consists all the frames and targets.
        if len(self.feature_bulk) == 0 and \
                len(self.target_bulk) == 0:  # feature_bulk and target_bulk should have the same length
            self.feature_bulk = np.concatenate(bulk_inputs)
            self.target_bulk = np.concatenate(bulk_targets)
        else:
            self.feature_bulk = np.concatenate((self.feature_bulk, np.concatenate(bulk_inputs)))
            self.target_bulk = np.concatenate((self.target_bulk, np.concatenate(bulk_targets)))
        # shuffle the frames in self.feature_bulk and self.target_bulk
        if shuffle:
            self.shuffle_frames()


    def finish_bulk(self):
        """
        Test if the bulk has reached reached the end

        Return:
            True/False
        """
        return len(self.feature_bulk) <= self.mini_batch_size


    def get_batch(self):
        """
        Get a mini_batch of features and targets.

        Returns:
            A pair containing:
                - The features: a list of feature matrices
                - The targets: a list of target vectors
        """
        #print ("BatchDispenser -- Finish bulk: {}".format(self.finish_bulk()))
        if self.finish_bulk():
            log.debug ("Finishing one bulk ... ")
            #print ("BatchDispenser -- End Scp: {}".format(self.end_scp))
            if not self.end_scp:
                self.fetch_utt_bulk(self.shuffle_flag)
            else:
                log.debug ("Finishing one epoch ... ")
                self.end_epoch = True
        if not self.end_epoch:
            mini_batch_feat = self.feature_bulk[:self.mini_batch_size]
            mini_batch_labl = self.target_bulk[:self.mini_batch_size]
            self.feature_bulk = self.feature_bulk[-(len(self.feature_bulk) - self.mini_batch_size):]
            self.target_bulk = self.target_bulk[-(len(self.target_bulk) - self.mini_batch_size):]
            #print ("Length of feature bulk: {} | Length of target bulk: {}".format(len(self.feature_bulk), len(self.target_bulk)))
        else:
            mini_batch_feat = []
            mini_batch_labl = []
        #log.debug ("BatchDispenser -- End Bulk: {}, End Scp: {}, End Epoch: {}".format(self.finish_bulk(), self.end_scp, self.end_epoch))
        return mini_batch_feat, mini_batch_labl, self.end_epoch


# TODO: merge into BatchDispenser
class TestBatchDispenser(object):
    __metaclass__ = ABCMeta

    def __init__(self, feature_reader, test_labels, label_reader = None, mini_batch_size = 128):
        '''
        test_labels: bool: need test labels or not
        '''
        # Store the feature reader and label reader
        self.feature_reader = feature_reader
        self.test_labels = test_labels
        self.label_reader = label_reader

        # Get a dictionary connecting training utterances and targets.
        # xsr6064 self.target_dict = self.read_target_file(target_path)

        # Detect the maximum length of the target sequences
        # xsr 6064 self.max_target_length = max([target_coder.encode(targets).size
        #                              for targets in self.target_dict.values()])

        self.utt_bulk_size = 1    # for test set, only 1 utterance per time

        # Store the mini_batch_size
        self.mini_batch_size = mini_batch_size

        # Lists where the feature and targets of the current bulk are stored
        self.features = None
        self.targets = None

        self.end_scp = False
        self.end_epoch = False

    def num_batches(self):
        return int(len(self.features) / self.mini_batch_size)

    def num_utt(self):
        return self.feature_reader.utt_count()


    def fetch_utt(self):
        #read utterance
        utt_id, utt_mat, looped = self.feature_reader.fetch()

        if utt_mat is not None:
            if self.test_labels:
                if self.label_reader.valid_uid(utt_id):
                    _, targets, _ = self.label_reader.fetch(utt_id)
                    assert (len(utt_mat) == len(targets))
        #        else:
        #            print ('WARNING no targets for %s' % utt_id)
        #else:
        #    print ('WARNING %s is an empty utterance' % utt_id)

        if looped:
            self.end_scp = True

        # bulk_inputs and bulk_targets are lists of ndarrays.
        # Flatten bulk_inputs and bulk_targets into one ndarray which consists all the frames and targets.
        #self.features = utt_mat
        #if self.test_labels:
        #    self.targets = targets

        # bulk_inputs and bulk_targets are lists of ndarrays.
        # path zeors to the feature list
        num_zeros = self.mini_batch_size - (len(utt_mat) % self.mini_batch_size)
        #print ("In test_batchdispenser - frame for the utt: {}".format(len(utt_mat)))
        #print ("num_zeros: {}".format(num_zeros))
        self.features = np.concatenate((utt_mat, np.zeros((num_zeros, utt_mat.shape[1]))))
        #print ("In test_batchdispenser - frame for the self.features: {}".format(len(self.features)))

        #for j in range(len(self.features)):
        #        print (self.features[j].shape)
        #        print ("{}th frame: {}".format(j, self.features[j]))
        #print ("----------------------------------------")
        if self.test_labels:
            self.targets = targets
            #print (type(self.targets))
            # TODO: patch zeros to targets

        return len(utt_mat), utt_id


    def get_batch(self):
        """
        Get a mini_batch of features and targets.

        Returns:
            A pair containing:
                - The features: a list of feature matrices
                - The targets: a list of target vectors
        """
        #print ("BatchDispenser -- Finish bulk: {}".format(self.finish_bulk()))
        if self.end_scp:
            if len(self.features) == self.mini_batch_size:
                log.debug ("Finishing one epoch ... ")
                self.end_epoch = True

        mini_batch_feat = self.features[:self.mini_batch_size]
        self.features = self.features[-(len(self.features) - self.mini_batch_size):]
        if self.test_labels:
            self.target = self.targets[-(len(self.targets) - self.mini_batch_size):]
            mini_batch_labl = self.targets[:self.mini_batch_size]
        else:
            mini_batch_labl = None
        #print ("Length of feature bulk: {} | Length of target bulk: {}".format(len(self.feature_bulk), len(self.target_bulk)))

        return mini_batch_feat, mini_batch_labl, self.end_epoch

    # ========================================================
    #def split(self):
    #    '''
    #    split off the part that has already been read by the batchdispenser

    #    this can be used to read a validation set and then split it off from
    #    the rest
    #    '''
    #    self.feature_reader.split()

    #def get_val_set(self, val_set_size):
    #    '''
    #    :param val_set_size: num of utterances in the val set
    #    :return: featutures and labels of the val set
    #    '''

    #def skip_bulk(self):
    #    '''skip a batch'''

    #    skipped = 0
    #    while skipped < self.utt_bulk_size:
    #        #read nex utterance
    #        utt_id = self.feature_reader.next_id()

    #        if utt_id in self.target_dict:
    #            #update number skipped utterances
    #            skipped += 1

    #'''
    #def return_batch(self):
    #    # Reset to previous batch

    #    skipped = 0

    #    while skipped < self.size:
    #        #read previous utterance
    #        utt_id = self.feature_reader.prev_id()

    #        if utt_id in self.target_dict:
    #            #update number skipped utterances
    #            skipped += 1
    #'''

    #def compute_target_count(self):
    #    '''
    #    compute the count of the targets in the data

    #    Returns:
    #        a numpy array containing the counts of the targets
    #    '''

    #    #create a big vector of stacked encoded targets
    #    encoded_targets = np.concatenate(
    #        [self.target_coder.encode(targets)
    #         for targets in self.target_dict.values()])

    #    #count the number of occurences of each target
    #    count = np.bincount(encoded_targets,
    #                        minlength=self.target_coder.num_labels)

    #    return count

    #@property
    #def num_bulk(self):
    #    '''
    #    The number of utterance bulks in the given data.

    #    The number of bulk is not necessarily a whole number
    #    '''

    #    return self.num_utt/self.utt_bulk_size

    #@property
    #def num_utt(self):
    #    '''The number of utterances in the given data'''

    #    return self.label_reader.utt_count()

    ##@property
    ##def num_labels(self):
    ##    '''the number of output labels'''

    ##    return self.target_coder.num_labels

    ##@property
    ##def max_input_length(self):
    ##    '''the maximal sequence length of the features'''

    ##    return self.feature_reader.max_input_length
