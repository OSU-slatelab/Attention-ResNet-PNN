#!/usr/bin/env python

# Code is adapted from essen's nnet.py

import sys, os, struct, argparse, logging
import numpy as np
import tensorflow as tf
import kaldiIO

np.set_printoptions(threshold=np.nan)

logging.basicConfig()
log = logging.getLogger("kaldi_reader")
log.setLevel(logging.DEBUG)

# this is from kaldi_io (https://github.com/kronos-cm/kaldi-io-for-python/blob/python_2_and_3/kaldi_io.py)

def write_mat_stdout(m, key=''):
    """ write_mat_stdout(m, key=IS_EMPTY)
    Write a binary kaldi matrix to stdout. Supports 32bit and 64bit floats.
    Arguments:
    m: the matrix to be stored,
    key (optional): used for writing ark-file, the utterance-id gets written before the matrix.
    """
    try:
        #
        #m=numpy.roll(m,-1,axis=0)
        #
        if key: 
            sys.stdout.buffer.write(struct.pack('<%dss' % len(key), str.encode(key), b' '))
        sys.stdout.buffer.write(struct.pack('<xc', b'B'))  # we write binary!
        # Data-type,
        if   m.dtype == 'float32': sys.stdout.buffer.write(strcut.pack('<ccc', b'F', b'M', b' '))
        elif m.dtype == 'float64': sys.stdout.buffer.write(struct.pack('<ccc', b'D', b'M', b' '))
        else: raise MatrixDataTypeError
        # Dims,
        sys.stdout.buffer.write(struct.pack('<bi', 4, m.shape[0]))  # rows
        sys.stdout.buffer.write(struct.pack('<bi', 4, m.shape[1]))  # cols
        # Data,
        sys.stdout.buffer.write(m.tobytes())
    finally:
        pass

# end

def load_prior(prior_path):
    with open(prior_path, "r") as f:
        for line in f:
            #log.debug(line.split(" ")[0:-1])
            #log.debug(line.split(" ")[1:-1])
            #counts = np.array(list(map(int, line.split(" ")[1:-1])), dtype=np.int32)
            counts = np.array(list(map(float, line.split(" ")[2:-1])), dtype=np.float32)
            #print (counts)
            #cnt_sum = reduce(lambda x, y: x + y, counts)
            cnt_sum = np.sum(counts)
            rel_freq = counts.astype(np.float32) / cnt_sum
            log_priors = rel_freq + 1e-20
            log_priors = np.log(log_priors)
            log_priors[rel_freq < 1e-10] = np.sqrt(np.finfo(np.float32).max)
    return log_priors

#def comp_softmax(sess):

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_counts', help='path to the file contains the label counts')
    parser.add_argument('--use_softmax', choices=['true', 'false'], help='use softmax or not')
    parser.add_argument('scp_file', help='path the the scp file')
    args = parser.parse_args()

    counts = args.label_counts
    use_softmax = args.use_softmax
    feat_scp = args.scp_file
    log.debug(type(feat_scp))
    log.debug(feat_scp)
    #featdir = '/data/data22/scratch/xusi/chime2/exp/resnet_20_resblock/test/decode_softmax'  #TODO: put it in args
    #featdir = '/data/data20/scratch/xusi/chime3/exp/attention_resnet/test/decode_softmax_real' #TODO: put it in args
    #'/data/data22/scratch/xusi/chime2/exp/attention_resnet_denoised/test/decode_softmax'  #TODO: put it in args
    #/homes/3/xusi/dir11/egs/chime_wsj0/s5/Chime_04-17/exp/tri4a_dnn_delta_cleanali_noisy/ali_train_pdf.counts

    log_prior = np.array(load_prior(counts), dtype=np.float64)
    #log_prior = tf.convert_to_tensor(log_prior)
    featreader = kaldiIO.TableReader(feat_scp, shuffle=False)
    
    #sess = tf.Session(config=tf.ConfigProto(device_count = {'GPU': 0}))      #-------------------------
    #sess = tf.Session()
    #uids = []
    #total_log_like_array = []
    #with sess.as_default():
        #with kaldiIO.TableWriter(featdir + '/log_likelihood.scp', featdir + '/log_likelihood.ark') as log_like_writer:
    for i in range(featreader.utt_count()):
        uid, mat, _ = featreader.fetch()
        log.debug("In nnet-forward, {}th utt, uid - {}, mat dim: {}".format(i, uid, mat.shape))
        nnet_out = mat
        #if use_softmax == 'true':   #------------------------------
        #    #log.debug("!!!!!!!")
        #    nnet_out = tf.nn.softmax(nnet_out)
        #    nnet_out = tf.add(nnet_out, 1e-100)
        #    nnet_out = tf.log(nnet_out)
        #    nnet_out = sess.run(nnet_out)
        out = nnet_out - log_prior
        #log.debug (out)
        #log.debug (50*'=')
        #out = sess.run(log_likelihood)
        #log_like_writer.write(str(uid), out)
        #uids.append(uid)
        #total_log_like_array.append(out)
        write_mat_stdout(out,key=uid)
        #for uid, log_like in zip(uids, total_log_like_array):
        #    log_like_writer.write(str(uid), log_like)

