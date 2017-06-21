"""
@File         : mnist_input.py
@Time         : 2017/6/20 
@Author       : Chen Huang
@Update       : 
@Discription  : Download and extract data;
                convert original data to standard tensorflow data '.tfrecords';
                generate image and label batch.
"""


import os
import gzip
import numpy
from tensorflow.python.platform import gfile
from tensorflow.contrib.learn.python.learn.datasets import base
import tensorflow as tf


# dirs
DATA_DIR = '../data'
LOG_DIR = '../logs'

# path of mnist
SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'

TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

# information of mnist
NUM_CLASSES = 10
HEIGHT = 28
WIDTH = 28
DEPTH = 1

TRAIN_DATA_NUM = 55000
VAL_DATA_NUM = 5000
TEST_DATA_NUM = 10000


def _read32(bytestream):
    dt = numpy.dtype(numpy.uint32).newbyteorder('>')
    return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def extract_images(f):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth].

    Args:
        :param f: file object. a file object that can be passed into a gzip reader.

    Returns:
        :return data: A 4D uint8 numpy array [index, y, x, depth].

    Raises:
        :exception ValueError: If the bytestream does not start with 2051.
    """
    print('Extracting', f.name)
    with gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                             (magic, f.name))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = numpy.frombuffer(buf, dtype=numpy.uint8)
        data = data.reshape(num_images, rows, cols, 1)

        return data


def extract_labels(f, one_hot=False, num_classes=10):
    """Extract the labels into a 1D uint8 numpy array [index].

    Args:
        :param f: file object. A file object that can be passed into a gzip reader.
        :param one_hot: bool. Does one hot encoding for the result.
        :param num_classes: int. Number of classes for the one hot encoding.

    Returns:
        :returns labels: ndarray. a 1D uint8 numpy array.

    Raises:
        :exception ValueError: If the bystream doesn't start with 2049.
    """
    print('Extracting', f.name)
    with gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError('Invalid magic number %d in MNIST label file: %s' %
                             (magic, f.name))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = numpy.frombuffer(buf, dtype=numpy.uint8)
        if one_hot:
            return dense_to_one_hot(labels, num_classes)

        return labels


def load_data():
    """Download and extract specific data.

     Returns:
        :return train_set, val_set, test_set: tuple - (images, labels).
    """
    one_hot = False

    local_file = base.maybe_download(filename=TRAIN_IMAGES, work_directory=DATA_DIR,
                                     source_url=SOURCE_URL + TRAIN_IMAGES)
    with open(local_file, 'rb') as f:
        train_images = extract_images(f)

    local_file = base.maybe_download(filename=TRAIN_LABELS, work_directory=DATA_DIR,
                                     source_url=SOURCE_URL + TRAIN_LABELS)
    with open(local_file, 'rb') as f:
        train_labels = extract_labels(f, one_hot=one_hot)

    local_file = base.maybe_download(filename=TEST_IMAGES, work_directory=DATA_DIR,
                                     source_url=SOURCE_URL + TEST_IMAGES)
    with open(local_file, 'rb') as f:
        test_images = extract_images(f)

    local_file = base.maybe_download(filename=TEST_LABELS, work_directory=DATA_DIR,
                                     source_url=SOURCE_URL + TEST_LABELS)
    with open(local_file, 'rb') as f:
        test_labels = extract_labels(f, one_hot=one_hot)

    validation_images = train_images[:VAL_DATA_NUM]
    validation_labels = train_labels[:VAL_DATA_NUM]
    train_images = train_images[VAL_DATA_NUM:]
    train_labels = train_labels[VAL_DATA_NUM:]

    train_set = (train_images, train_labels)
    val_set = (validation_images, validation_labels)
    test_set = (test_images, test_labels)

    return train_set, val_set, test_set


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(data_set, name):
    """Converts a dataset to tfrecords.

    Args:
      :param data_set: tuple - (images, labels).
      :param name: str - 'train'/'validation'/'test'.
    """
    images = data_set[0]
    labels = data_set[1]

    num_examples = images.shape[0]
    rows = images.shape[1]
    cols = images.shape[2]
    depth = images.shape[3]

    filename = os.path.join(DATA_DIR, name + '.tfrecords')
    if not gfile.Exists(filename):
        print('Writing', filename)
        writer = tf.python_io.TFRecordWriter(filename)
        for index in range(num_examples):
            image_raw = images[index].tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(rows),
                'width': _int64_feature(cols),
                'depth': _int64_feature(depth),
                'label': _int64_feature(int(labels[index])),
                'image_raw': _bytes_feature(image_raw)}))
            writer.write(example.SerializeToString())
        writer.close()


def read_and_decode(filename_queue):
    """Reads and parses examples from MNIST data files .tfrecords.
    
    Args:
        :param filename_queue: queue. A queue of strings with the filenames to read from. 
    
    Returns:
        :return result: DataRecord. An object representing a single example, with the following fields:
            key: a scalar string Tensor describing the filename & record number
            for this example.
            label: an int32 Tensor with the label in the range 0..9.
            image: a [height, width, depth] uint8 Tensor with the image data.
    """
    class DataRecord(object):
        pass

    result = DataRecord()
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })

    # Convert from a scalar string tensor (whose single string has
    # length mnist.IMAGE_PIXELS) to a uint8 tensor
    result.image = tf.decode_raw(features['image_raw'], tf.uint8)
    result.image.set_shape([HEIGHT * WIDTH * DEPTH])

    # Convert label from a scalar uint8 tensor to an int32 scalar.
    result.label = tf.cast(features['label'], tf.int32)

    return result


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
    """Construct a queued batch of images and labels.
  
    Args:
        :param image: 3-D Tensor of [height, width, 3] of type.float32.
        :param label: 1-D Tensor of type.int32
        :param min_queue_examples: int32, minimum number of samples to retain 
        in the queue that provides of batches of examples.
        :param batch_size: Number of images per batch.
        :param shuffle: boolean indicating whether to use a shuffling queue.
    
    Returns:
        :return images: Images. 4D tensor of [batch_size, height, width, 3] size.
        :return labels: Labels. 1D tensor of [batch_size] size.
    """
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    num_preprocess_threads = 8
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)

    # Display the training images in the visualizer.
    reshaped_images = tf.reshape(images, [-1, HEIGHT, WIDTH, DEPTH])
    tf.summary.image('images', reshaped_images)

    return images, tf.reshape(label_batch, [batch_size])


def distorted_inputs(filenames, batch_size):
    """Construct distorted input for MNIST training using the Reader ops.
    
    Args:
        :param filenames: list - [str1, str2, ...].
        :param batch_size: int. 
    
    Returns:
       :returns: tuple - (images, labels).
                images: tensors - [batch_size, height*width*depth].
                lables: tensors - [batch_size].
    """
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    with tf.name_scope('input'):
        # Create a queue that produces the filenames to read.
        filename_queue = tf.train.string_input_producer(string_tensor=filenames)

        # Even when reading in multiple threads, share the filename
        # queue.
        result = read_and_decode(filename_queue)

        # # OPTIONAL: Could reshape into a 28x28 image and apply distortions
        # # here.
        # reshaped_image = tf.reshape(image, [result.height, result.width, result.depth])
        #
        # # Randomly crop a [height, width] section of the image.
        # distorted_image = tf.random_crop(reshaped_image, [patch_size, patch_size, result.depth])
        #
        # # Randomly flip the image horizontally.
        # distorted_image = tf.image.random_flip_left_right(distorted_image)
        #
        # # Because these operations are not commutative, consider randomizing
        # # the order their operation.
        # distorted_image = tf.image.random_brightness(distorted_image,
        #                                              max_delta=63)
        # distorted_image = tf.image.random_contrast(distorted_image,
        #                                            lower=0.2, upper=1.8)
        #
        # # Subtract off the mean and divide by the variance of the pixels.
        # distorted_image = tf.image.per_image_standardization(distorted_image)
        #
        # # Set the shapes of tensors.
        # distorted_image.set_shape([patch_size, patch_size, result.depth])
        # result.label.set_shape([1])

        # Convert from [0, 255] -> [-0.5, 0.5] floats.
        image = tf.cast(result.image, tf.float32) * (1. / 255) - 0.5

        distorted_image = image
        label = result.label

        # Ensure that the random shuffling has good mixing properties.
        min_queue_examples = 1000
        print('Filling queue with %d mnist images before starting to train or validation. '
              'This will take a few minutes.' % min_queue_examples)

        # Generate a batch of images and labels by building up a queue of examples.
        return _generate_image_and_label_batch(image=distorted_image, label=label,
                                               min_queue_examples=min_queue_examples, batch_size=batch_size,
                                               shuffle=True)


def inputs(filenames, batch_size):
    """Construct input without distortion for MNIST using the Reader ops.

    Args:
        :param filenames: list - [str1, str2, ...].
        :param batch_size: int. 

    Returns:
       :returns: tuple - (images, labels).
                images: tensors - [batch_size, height*width*depth].
                lables: tensors - [batch_size].
    """
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    with tf.name_scope('input'):
        # Create a queue that produces the filenames to read.
        filename_queue = tf.train.string_input_producer(string_tensor=filenames)

        # Even when reading in multiple threads, share the filename
        # queue.
        result = read_and_decode(filename_queue)

        # Convert from [0, 255] -> [-0.5, 0.5] floats.
        image = tf.cast(result.image, tf.float32) * (1. / 255) - 0.5

        label = result.label

        # Ensure that the random shuffling has good mixing properties.
        min_queue_examples = 1000
        print('Filling queue with %d mnist images before starting to train or validation. '
              'This will take a few minutes.' % min_queue_examples)

        # Generate a batch of images and labels by building up a queue of examples.
        return _generate_image_and_label_batch(image=image, label=label,
                                               min_queue_examples=min_queue_examples, batch_size=batch_size,
                                               shuffle=True)