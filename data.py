from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os
import tensorflow as tf
import numpy as np

from vng_model import transfer_learning


def preprocess_image(image, width, height, scope=None):
    """

    :param image:
    :param width:
    :param height:
    :param scope:
    :return:
    """

    with tf.name_scope(scope, 'preprocess_image', [image, width, height]):
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, tf.float32)

        if height and width:
            image = tf.expand_dims(image, 0)
            image = tf.image.resize_bilinear(image, [height, width], align_corners=False)
            image = tf.squeeze(image, [0])

    return image


# def extract_feature(sess, preprocessed_img, scope=None):
#     """
#     :param sess:
#     :param preprocessed_img:
#     :param scope:
#     :return:
#     2-D float tensor containing extracted features
#     """
#
#     with tf.name_scope(scope, 'extract_feature'):
#         extracted_features = transfer_learning(sess, image=preprocessed_img)
#     return extracted_features


def generate_standard_dataset(dir_path, width, height, file_extension='*.jpg'):
    # '*.jpg'
    file_names = []

    for pathAndFileName in glob.iglob(os.path.join(dir_path, file_extension)):
        file_names.append(pathAndFileName)

    filename_queue = tf.train.string_input_producer(file_names, shuffle=None)

    reader = tf.WholeFileReader()

    _, value = reader.read(filename_queue)

    image = tf.image.decode_jpeg(value, channels=3)

    image = preprocess_image(image, width, height)

    return image, file_names


def _int64_feature(value):
    """
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    """
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def generate_tfrecords(tfrecords_dir, normal_images, labels, name_file):
    """
        Helper to saved preprocessed images into a tfrecord-file
    which is the standard format of tensorflow

    :param tfrecords_dir: directory where tfrecord-file is saved
    :param normal_images: the preprocessed images of size [#images, 299, 299]
    :param labels: the real label of images
    :param name_file: name of tfrecord-file
    :return:
    a tfrecord file
    """

    num_samples = normal_images.shape[0]

    file_name = os.path.join(tfrecords_dir, name_file + '.tfrecords')

    writer = tf.python_io.TFRecordWriter(file_name)

    for idx in range(num_samples):
        raw_image = normal_images[idx].tostring()
        example = tf.train.Example(features=tf.train.Features(
            feature={'label': _int64_feature(int(labels[idx])),
                     'image_raw': _bytes_feature(raw_image)}
        ))
        writer.write(example.SerializeToString())

    writer.close()


def read_and_decode(filename_queue, height, width, channels):
    """

    :param filename_queue:
    :param height
    :param width
    :param channels
    :return:
    """
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string)
        }
    )
    image = tf.decode_raw(features['image_raw'], tf.float32)
    image = tf.reshape(image, [height, width, channels])
    label = tf.cast(features['label'], tf.int32)
    return image, label


def extract_features(sess, expected_size=(299, 299), data_dir=None,
                     tfrecord_path=None, num_samples=None):
    """

    :param sess:
    :param expected_size:
    :param data_dir:
    :param tfrecord_path:
    :param num_samples:
    :return:
    """

    filenames = []
    if data_dir is not None:
        for pathAndFileName in glob.iglob(os.path.join(data_dir, '*.jpg')):
            filenames.append(pathAndFileName)

        filename_queue = tf.train.string_input_producer(filenames, shufflestring_input_producer=None)

        reader = tf.WholeFileReader()

        _, value = reader.read(filename_queue)

        image = tf.image.decode_jpeg(value, channels=3)

        image = preprocess_image(image, expected_size[0], expected_size[1])

        features = transfer_learning(sess, image=image)

        return features

    elif tfrecord_path is not None:
        print('Processsing...')
        filename_queue = tf.train.string_input_producer([tfrecord_path], shuffle=None)

        print('Reading processed images, and labels')
        image, label = read_and_decode(filename_queue, expected_size[0], expected_size[1], 3)

        print('Initialziing....')
        images_batch = [None] * num_samples
        label_batch = [None] * num_samples

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for idx in range(num_samples):

            img, lb = sess.run([image, label])

            if idx == 0:
                result = transfer_learning(sess, image=img)
                result = np.array(result)
                result = result.reshape((1, 1, 2048))
                images_batch[idx] = result

            else:
                result = transfer_learning(sess, image=img, fist_load=False)
                result = np.array(result)
                result = result.reshape((1, 1, 2048))
                images_batch[idx] = result

            label_batch[idx] = lb

        coord.request_stop()
        coord.join(threads)

        return images_batch, label_batch


# ############## Batch of Images ############# #

def input_data(data_dir, batch_size, is_training=True):

    filename_queue = tf.train.string_input_producer(data_dir)

    # sample, label = read_and_decode(filename_queue, 1, 1, 2048)

    sample, label = read_and_decode(filename_queue, 224, 224, 3)

    if is_training:
        sample_batch, label_batch = tf.train.shuffle_batch(
            [sample, label],
            batch_size=batch_size,
            capacity=10 + 3 * batch_size,
            min_after_dequeue=10,
            name='input_train_data'
        )

    else:
        sample_batch, label_batch = tf.train.batch(
            [sample, label],
            batch_size=batch_size,
            capacity= batch_size,
            name='input_val_data'
        )

    return sample_batch, label_batch
