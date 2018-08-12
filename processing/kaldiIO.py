from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import struct
import fnmatch
import gzip
import numpy as np
import logging
# import random
from os import listdir
from random import shuffle
from os.path import isfile, join, splitext

logging.basicConfig()
log = logging.getLogger("kaldi_reader")
log.setLevel(logging.DEBUG)


class TableReader(object):
    """
    class used to read Kaldi's feature files
    """

    def __init__(self, scp_file, shuffle = True):
        """
        spc_file: file contains the infomation of all the features
        """
        self.map = {
            int: self._fetch_arg_int,
            str: self._fetch_arg_str
        }
        self.scp = scp_file
        self.cur = 0
        self.utt_id_list = []
        self.feat_path_list = []
        self.fopen = None
        # order of the utterances
        # used for keeping the feat and label list having the same order
        # see _shuffle_utt and _get_scp_order
        self.order = []
        self.shuffle = shuffle
        # initialize the path list from the scp file
        with open(self.scp, 'r') as scp_file:
            for line in scp_file:
                uid, feat_info = line.replace('\n', "").split(' ')
                feat_path, pos = feat_info.split(':')
                self.utt_id_list.append(uid)
                self.feat_path_list.append((feat_path, pos))
        #print("Number of utterances for features: {}".format(len(self.feat_path_list)))
        self.scp_order = range(len(self.utt_id_list))
        if self.shuffle:
            self._shuffle_utt()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.fopen:
            log.debug("File {} is now closing\n".format(self.fopen.name))
            self.fopen.close()

    def _shuffle_utt(self):
        shuffle(self.order)
        self.utt_id_list = [self.utt_id_list[i] for i in self.scp_order]
        self.feat_path_list = [self.feat_path_list[i] for i in self.scp_order]

    def _read(self, path, pos):
        """
        :param path: path to a feature file
        :param pos: staring position corresponding to the current utterance
        :return: a feature matrix for an utt: each row corresponds to a frame
        """
        if not self.fopen or self.fopen.name != path:
            if self.fopen:
                log.debug("File {} is now closing\n".format(self.fopen.name))
                self.fopen.close()
            self.fopen = open(path, 'rb')  # noqa
            log.debug("Feature file {} is now open\n".format(self.fopen.name))
        self.fopen.seek(pos)
        bflag = b''.join(struct.unpack('<xc', self.fopen.read(2))).decode()
        if bflag != 'B':
            log.debug("Flag 'B' is expected, but got '{}'".format(bflag))
            exit(1)
        c = self.fopen.read(1)
        self.fopen.seek(-1, 1)
        if c == 'C':
            num_rows, num_cols, mat = self._readCompMat(self.fopen)
        else:
            num_rows, num_cols, mat = self._readMat(self.fopen)
        return np.reshape(mat, (num_rows, num_cols))

    def _readMat(self, f):
        token = b''.join(struct.unpack('<ccc', f.read(3))).decode().strip()
        if token != 'FM' and token != 'DM':
            log.debug("Wrong token: '{}'".format(token))
            exit(1)
        _, num_rows = struct.unpack('<bi', f.read(5))
        _, num_cols = struct.unpack('<bi', f.read(5))
        dt = np.float32 if token == 'FM' else np.float64
        mat = np.frombuffer(
            f.read(num_rows * num_cols * np.dtype(dt).itemsize), dtype=dt
        )
        return num_rows, num_cols, mat

    def _readCompMat(self, f):
        """
        :param f: file object to the compressed matrix
        :return: num_rows, num_cols, feature matrix
        """
        token = b''.join(struct.unpack('<ccc', f.read(3))).decode().strip()
        if token != 'CM':
            log.debug("Only support 'CM' case, 'CM2' and 'CM3' are not supported.")
            log.debug("Got token '{}'".format(token))
            exit(1)
        min_value, rng, num_rows, num_cols = struct.unpack(
            '<ffii', f.read(16)
        )
        exit(1)
        pc_headers = np.frombuffer(f.read(num_cols * 8), dtype=np.uint16)
        pc_headers = np.reshape(pc_headers, (num_cols, 4))
        data = np.frombuffer(f.read(num_rows * num_cols), dtype=np.uint8)
        data = np.reshape(data, (num_cols, num_cols))
        mat = np.zeros(shape=(num_rows, num_cols))
        for i in range(num_cols):
            p = pc_headers[i, :]
            val = data[i, :]
            p0 = self._Uint16ToFloat(min_value, rng, p[0])
            p25 = self._Uint16ToFloat(min_value, rng, p[1])
            p75 = self._Uint16ToFloat(min_value, rng, p[2])
            p100 = self._Uint16ToFloat(min_value, rng, p[3])
            for j in range(num_rows):
                mat[j, i] = self._CharToFloat(p0, p25, p75, p100, val[j])
        return num_rows, num_cols, mat

    def _Uint16ToFloat(self, min_v, rng, val):
        if type(val) is not np.uint16:
            log.debug("Wrong data type in pc_headers: '{}'".format(type(val)))
            exit(1)
        return min_v + rng * 1.52590218966964e-05 * val

    def _CharToFloat(self, p0, p25, p75, p100, val):
        if type(val) is not np.uint8:
            log.debug("Wrong data type for value: '{}'".format(type(val)))
        if val <= 64:
            return p0 + (p25 - p0) * val * (1 / 64.0)
        elif val <= 192:
            return p25 + (p75 - p25) * (val - 64) * (1 / 128.0)
        else:
            return p75 + (p100 - p75) * (val - 192) * (1 / 63.0)

    def fetch(self, s=None):
        """
        :param s: None: fetch the feat which the current cursor points at
                  int: fetch the i'th feat
                  str: use the uid (str) to fetch the feat

        :return: uid, feat, loop
        """
        if s is None:
            return self._fetch()
        else:
            if self.map[type(s)] is not None:
                return self.map[type(s)](s)
            else:
                raise TypeError("Only support int and str input")

    def _fetch(self):
        """
        Fetch the features that the current cursor points at
        """
        loop = False
        #log.debug("current cursor vs. len of feat: {} vs. {}".format(self.cur, len(self.feat_path_list)))
        if self.cur >= len(self.feat_path_list):
            log.debug("Reach the end of the feature list. Go back to the top")
            self.cur = 0
            loop = True
        info = self.feat_path_list[self.cur]
        uid = self.utt_id_list[self.cur]
        self.cur += 1
        #print ("Feature Reader loop: {}".format(loop))
        return uid, self._read(info[0], int(info[1])), loop

    def _fetch_arg_str(self, uid):
        """
        :param uid: utterance ID
        :return: features of the utterance with the utterance ID
        """
        info = self.feat_path_list[self.utt_id_list.index(uid)]
        return uid, self._read(info[0], int(info[1])), False

    def _fetch_arg_int(self, i):
        """
        :param i: index i
        :return: features of the i'th utterance in the feat_path_list
        """
        if i < 0 or i > len(self.feat_path_list):
            log.debug("{} is out of range of the feature list".format(i))
            exit(1)
        info = self.feat_path_list[i]
        return self.utt_id_list[i], self._read(info[0], int(info[1])), False

    def utt_count(self):
        return len(self.feat_path_list)

    def get_scp_order(self):
        return self.scp_order

    def get_uid_list(self):
        return self.utt_id_list

    def remove_lab_difference(self, label_uid_list):
        diff = [x for x in self.utt_id_list if x not in label_uid_list]
        for u in diff:
            print ("Removing utterance: {} ...".format(u))
            self.remove(u)
        self.scp_order = range(len(self.utt_id_list))

    def remove(self, uid):
        '''
        Remove the utterance of uid from self.utt_id_list and self.feat_path_list
        :param uid: id of the utterance
        '''
        if uid in self.utt_id_list:
            ind = self.utt_id_list.index(uid)
            self.utt_id_list.remove(uid)
            del self.feat_path_list[ind]
        else:
            log.debug("WARNING: Undefined utterance ID: {}\n".format(uid))

    def reset(self):
        '''
        Reset the reader. Shuffle the scp if self.shuffle is true
        '''
        self.cur = 0
        if self.shuffle:
            self._shuffle_utt()


    # close file explicitly
    # same function as __exit__()
    def close(self):
        if self.fopen:
            log.debug("File {} is now closing\n".format(self.fopen.name))
            self.fopen.close()


class TableWriter(object):
    """
    Class that write data into Kaldi's matrix format
    """

    def __init__(self, scp_file, ark_file):
        self.scp_path = scp_file
        self.ark_path = ark_file
        self.scp = open(self.scp_path, 'w')
        self.ark = open(self.ark_path, 'wb')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.scp.close()
        self.ark.close()

    def write(self, uid, data):
        scp_writer = self.scp
        ark_writer = self.ark
        if str(data.dtype) == 'float32':
            dt = np.float32
            tok = 'F'
        elif str(data.dtype) == 'float64':
            dt = np.float64
            tok = 'D'
        data = np.asarray(data, dtype=dt)
        num_rows, num_cols = data.shape
        ark_writer.write(struct.pack('<%dss' % len(uid), str.encode(uid), b' '))
        cur_pos = ark_writer.tell()
        ark_writer.write(struct.pack('<xcccc', b'B', str.encode(tok), b'M', b' '))
        ark_writer.write(struct.pack('<bi', 4, num_rows))
        ark_writer.write(struct.pack('<bi', 4, num_cols))
        ark_writer.write(data)
        scp_writer.write("{} {}:{}\n".format(uid, self.ark_path, cur_pos))


class LabelReader(object):
    """
    Class to load the labels for utterances.
    Works for the binary mode
    """

    def __init__(self, path):
        '''
        :param path: 1. file.scp path: use the scp to read in the labels
                     2. directory path: the directory which includes all the alignment files (usually in zip form)
        '''
        self.scp = None
        self.ali_dir = None
        self.cur = 0
        self.label_list = []
        self.utt_id_list = []
        self.labl_path_list = []
        self.fopen = None
        self.map = {
            int: self._fetch_arg_int,
            str: self._fetch_arg_str
        }
        if splitext(path)[1] == ".scp":
            self.scp = path
        else:
            self.ali_dir = path
        if self.scp is not None:
            with open(self.scp, 'r') as scp_file:
                for line in scp_file:
                    uid, feat_info = line.replace('\n', "").split(' ')
                    feat_path, pos = feat_info.split(':')
                    self.utt_id_list.append(uid)
                    self.labl_path_list.append((feat_path, pos))
            print("Number of label sequences: {}".format(len(self.labl_path_list)))
        else:
            files = [f for f in listdir(self.ali_dir) if isfile(
                join(self.ali_dir, f))]
            files = [f for f in files if fnmatch.fnmatch(f, "ali.*.gz")]
            files.sort(key=lambda f: int(f.split('.')[1]))
            ali_file_list = [join(self.ali_dir, f) for f in files]
            self._load_all_labs(ali_file_list)  # file loading is done in init

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.fopen:
            log.debug("File {} is now closing\n".format(self.fopen.name))
            self.fopen.close()

    def _load_all_labs(self, ali_file_list):
        """
        Load into memory all label sequences store in the alignment files
        Used when scp file is not provided
        """
        for fn in ali_file_list:
            count = 0
            with gzip.open(fn, 'rb') as buf:
                while True:
                    self.utt_id_list.append(self._get_uid(buf))
                    self._check_header(buf)
                    self.label_list.append(self._get_ali(buf))
                    count += 1
                    break_flag = buf.read(1)
                    if not break_flag:
                        log.debug(
                            "Exit properly. Load {} labels from {}".format(
                                count, fn
                            )
                        )
                        break
                    buf.seek(-1, 1)

    def _read(self, path, pos):
        """
        Read the labels for one utterance
        """
        if not self.fopen or self.fopen.name != path:
            if self.fopen:
                log.debug("File {} is now closing\n".format(self.fopen.name))
                self.fopen.close()
            self.fopen = open(path, 'rb')  # noqa
            log.debug("Label file {} is now open\n".format(self.fopen.name))
        self.fopen.seek(pos)
        self._check_header(self.fopen)
        return self._get_ali(self.fopen)

    def _get_uid(self, buf):
        uid = ''
        c = buf.read(1)
        while c != ' ':
            uid += c
            c = buf.read(1)
        return uid

    def _check_header(self, buf):
        """
        Check if the input file is in binary mode
        """
        bflag = b''.join(struct.unpack('<xc', buf.read(2))).decode()
        if bflag != 'B':
            log.debug("Flag 'B' is expected, but got '{}'".format(bflag))
            exit(1)

    def _get_ali(self, buf):
        dsize, vlen = struct.unpack('<bi', buf.read(5))
        # log.debug("Size: {}".format(vlen))
        # dsize = dsize if dsize > 0 else -(dsize)
        tmp_list = []
        for _i in range(vlen):
            b = buf.read(5)
            if not b:
                log.debug("Run out of bytes, length of b is {}".format(len(b)))
            _, v = struct.unpack('<bi', b)
            tmp_list.append(v)
        vec = np.array(tmp_list)
        return vec

    def fetch(self, s=None):
        if not self.label_list and not self.labl_path_list:
            raise RuntimeError("Label set is empty")
        if s is None:
            return self._fetch()
        else:
            if self.map[type(s)] is not None:
                return self.map[type(s)](s)
            else:
                raise TypeError("fetch() only support int and str input")

    def _fetch(self):
        """
        Fetch the label that the current cursor points at
        """
        loop = False
        if self.cur >= len(self.labl_path_list):
            log.debug("Reach the end of the feature list. Go back to the top")
            self.cur = 0
            loop = True
        if self.labl_path_list:
            uid = self.utt_id_list[self.cur]
            info = self.labl_path_list[self.cur]
            lab = self._read(info[0], int(info[1]))
        else:
            uid = self.utt_id_list[self.cur]
            lab = self.label_list[self.cur]
        self.cur += 1
        #print ("Label reader loop: {}".format(loop))
        return uid, lab, loop

    def _fetch_arg_str(self, uid):
        """
        Fetch the label for the utterance with uid from the labl_path_list
        """
        if self.labl_path_list:
            info = self.labl_path_list[self.utt_id_list.index(uid)]
            lab = self._read(info[0], int(info[1]))
        else:
            lab = self.label_list[self.utt_id_list.index(uid)]
        return uid, lab, False

    def _fetch_arg_int(self, i):
        """
        Fetch the label for the i'th utterance in the labl_path_list
        """
        if self.labl_path_list:
            if i < 0 or i > len(self.labl_path_list):  # labl_list
                log.debug("{} is out of the range of the label list".format(i))
                exit(1)
            uid = self.utt_id_list[i]
            info = self.labl_path_list[i]
            lab = self._read(info[0], int(info[1]))
        else:
            if i < 0 or i > len(self.label_list):
                log.debug("{} is out of the range of the label list".format(i))
                exit(1)
            uid = self.utt_id_list[i]
            lab = self.label_list[i]
        return uid, lab, False

    def shuffle_utt(self, order):
        assert len(order) == len(self.utt_id_list)
        self.utt_id_list = [self.utt_id_list[i] for i in order]
        if self.label_list:
            self.label_list = [self.label_list[i] for i in order]
        else:
            self.labl_path_list = [self.labl_path_list[i] for i in order]

    def reset(self, shuff_order):
        self.cur = 0
        self.shuffle_utt(shuff_order)

    def valid_uid(self, uid):
        '''
        :param: uid
        :return: True if uid is in self.utt_id_list; else False
        '''
        return uid in self.utt_id_list

    def utt_count(self):
        if self.label_list:
            return len(self.label_list)
        elif self.labl_path_list:
            return len(self.labl_path_list)
        else:
            return 0

    def remove_utt_difference(self, feature_uid_list):
        diff = [x for x in self.utt_id_list if x not in feature_uid_list]
        print ("Length of diff: {}".format(len(diff)))
        for u in diff:
            #print ("Removing target squence: {} ...".format(u))
            self.remove(u)

    def remove(self, uid):
        '''
        Remove the utterance of uid from self.utt_id_list and self.feat_path_list
        :param uid: id of the utterance
        '''
        if uid in self.utt_id_list:
            ind = self.utt_id_list.index(uid)
            self.utt_id_list.remove(uid)
            if self.labl_path_list:
                del self.labl_path_list[ind]
            else:
                del self.label_list[ind]
        else:
            log.debug("WARNING: Undefined utterance ID: {}\n".format(uid))

    def get_uid_list(self):
        return self.utt_id_list


    # close file explicitly
    # same function as __exit__()
    def close(self):
        if self.fopen:
            log.debug("File {} is now closing\n".format(self.fopen.name))
            self.fopen.close()
