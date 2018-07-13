import tensorflow as tf
import pickle
import numpy as np
from CNN_Model_1.bins import *


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def dict2TFRecord(file_list):
    for index, file in enumerate(file_list):
        cif10_batch_dict = unpickle(file)
        data = cif10_batch_dict[b'data']
        labels = cif10_batch_dict[b'labels']
        batch_size = len(labels)
        new_file_name = "dataset\\data-%d.TFRecord" % (index)
        writer = tf.python_io.TFRecordWriter(new_file_name)
        for img_index in range(batch_size):
            img_row = data[img_index].tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'label': int64_feature(labels[img_index]),
                'img': bytes_feature(img_row)
            }))
            writer.write(example.SerializeToString())
        writer.close()
    return 'Done'


class Reader(object):

    def __init__(self, file_pattern, n_epochs=10, shuffle=True):
        self.reader = tf.TFRecordReader(name='cif10Reader')
        files = tf.train.match_filenames_once(file_pattern)
        self.file_queue = tf.train.string_input_producer(files, num_epochs=n_epochs, shuffle=shuffle)
        self._creat_read_node()

    def _creat_read_node(self):
        _, values = self.reader.read(self.file_queue)
        features = tf.parse_single_example(
            values,
            features={
                'img': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([1], tf.int64)
            }
        )
        # print(features['img'])
        # self.images = tf.string_to_number(features['img'])
        # print(self.images)
        # self.images = tf.cast(features['img'], tf.float32)
        # self.images.set_shape([32, 32, 3])
        # self.images = tf.string_to_number(features['img'], tf.float32)
        # print(self.images)
        self.images = tf.decode_raw(features['img'], tf.int8)

        self.images = tf.reshape(self.images, [32, 32, 3])
        print(self.images)
        self.images = tf.cast(self.images, tf.float32)
        print(self.images)
        self.labels = tf.cast(features['label'], tf.int32)

    def input_pipline(self, batch_size, min_after_dequeue=1024, ):
        capcity = min_after_dequeue + 3 * batch_size
        image_batch, label_batch = tf.train.shuffle_batch(
            [self.images, self.labels], batch_size, capcity, min_after_dequeue
        )
        return image_batch, label_batch


if __name__ == '__main__':
    """"""
    files = ['cifar-10-batches-py\\data_batch_%d' % (i) for i in range(1, 6) ]
    # print(files)
    ##TODO: 数据集信息##
    # for file in files:
    #     dict = unpickle(file)
    #     print('dict keys: ', dict.keys())
    #     print('batch_label: ', dict[b'batch_label'])
    #     print('one batch data size: ', dict[b'data'].shape)
    #     print('labels shape: ', len(dict[b'labels']))
    ##TODO: 转为TFRecord文件#
    print(dict2TFRecord(files))

