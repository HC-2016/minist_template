"""
@File         : mnist.py
@Time         : 2017/6/20 
@Author       : Chen Huang
@Update       : 
@Discription  : Builds the CIFAR-10 network.

Summary of available functions:

# Compute input images and labels for training. 
inputs, labels = distorted_inputs()

# Compute inference on the model inputs to make a prediction.
predictions = inference(images)

# Compute the total loss of the prediction with respect to the labels.
loss = loss_func(predictions, labels)

# Create a graph to run one step of training with respect to the loss.
train_op = train_func(loss, global_step)
"""

import os
import re

import numpy
import tensorflow as tf

from src import mnist_input

# training parameters
INI_LEARNING_RATE = 0.01
MAX_STEPS = 2000
BATCH_SIZE = 100
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.

DATA_TYPE = tf.float32

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'


def distorted_inputs(filenames):
    """Construct distorted input for MNIST training using the Reader ops.
    
    Args:
        :param filenames: list - [name1, name2, ...]
        
    Returns:
        :return images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
        :return labels: Labels. 1D tensor of [batch_size] size.
    
    Raises:
        :exception ValueError: If no data_dir
    """
    images, labels = mnist_input.distorted_inputs(filenames=filenames, batch_size=BATCH_SIZE)

    return images, labels


def inputs(filenames):
    """Construct input without distortion for MNIST using the Reader ops.
    
    Args:
        :param filenames: list - [name1, name2, ...]
        
    Returns:
        :return images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
        :return labels: Labels. 1D tensor of [batch_size] size.

    Raises:
        :exception ValueError: If no data_dir
    """
    images, labels = mnist_input.inputs(filenames=filenames, batch_size=BATCH_SIZE)

    return images, labels

def _activation_summary(x):
    """Helper to create summaries for activations.
    
    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.
    
    Args:
        :param x: Tensor
    """
    # # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # # session. This helps the clarity of presentation on tensorboard.
    # tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    # tf.summary.histogram(tensor_name + '/activations', x)
    # tf.summary.scalar(tensor_name + '/sparsity',
    #                                    tf.nn.zero_fraction(x))
    tf.summary.histogram('activations', x)
    tf.summary.scalar('sparsity', tf.nn.zero_fraction(x))


def inference(images, hidden1_units, hidden2_units, wd):
    """Build the MNIST model up to where it may be used for inference.
    
    Args:
        :param images: Images placeholder, from inputs().
        :param hidden1_units: Size of the first hidden layer.
        :param hidden2_units: Size of the second hidden layer.
        :param wd: float. add L2Loss weight decay multiplied by this float.
    
    Returns:
        :return logits: Output tensor with the computed logits.
    """
    # Hidden 1
    weights_initializer = tf.truncated_normal_initializer(
        stddev=1.0 / numpy.sqrt(float(mnist_input.HEIGHT * mnist_input.WIDTH * mnist_input.DEPTH)))
    with tf.variable_scope('Layer_hidden1'):
        weights = tf.get_variable(name='weights',
                                  shape=[mnist_input.HEIGHT * mnist_input.WIDTH * mnist_input.DEPTH, hidden1_units],
                                  initializer=weights_initializer, dtype=DATA_TYPE)
        biases = tf.get_variable(name='biases', shape=[hidden1_units],
                                 initializer=tf.constant_initializer(0.0), dtype=DATA_TYPE)
        hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
        _activation_summary(hidden1)

        if wd is not None:
            weight_decay = tf.multiply(tf.nn.l2_loss(weights), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)

    # Hidden 2
    weights_initializer = tf.truncated_normal_initializer(stddev=1.0 / numpy.sqrt(float(hidden1_units)))
    with tf.variable_scope('Layer_hidden2'):
        weights = tf.get_variable(name='weights', shape=[hidden1_units, hidden2_units],
                                  initializer=weights_initializer, dtype=DATA_TYPE)
        biases = tf.get_variable(name='biases', shape=[hidden2_units],
                                 initializer=tf.constant_initializer(0.0), dtype=DATA_TYPE)
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
        _activation_summary(hidden2)

        if wd is not None:
            weight_decay = tf.multiply(tf.nn.l2_loss(weights), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)

    # Linear
    weights_initializer = tf.truncated_normal_initializer(stddev=1.0 / numpy.sqrt(float(hidden2_units)))
    with tf.variable_scope('Layer_linear'):
        weights = tf.get_variable(name='weights', shape=[hidden2_units, mnist_input.NUM_CLASSES],
                                  initializer=weights_initializer, dtype=DATA_TYPE)
        biases = tf.get_variable(name='biases', shape=[mnist_input.NUM_CLASSES],
                                 initializer=tf.constant_initializer(0.0), dtype=DATA_TYPE)
        logits = tf.matmul(hidden2, weights) + biases
        _activation_summary(logits)

        if wd is not None:
            weight_decay = tf.multiply(tf.nn.l2_loss(weights), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)

    return logits


def loss_func(logits, labels):
    """Add L2Loss to all the trainable variables.
    
    Add summary for "Loss" and "Loss/avg".
    Args:
        :param logits: Logits from inference().
        :param labels: Labels from distorted_inputs or inputs(). 1-D tensor
                of shape [batch_size]
    
    Returns:
        :return Loss tensor of type float.
    """
    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int64)
    with tf.variable_scope('Loss'):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)

        # The total loss is defined as the cross entropy loss plus all of the weight
        # decay terms (L2 loss).
        return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
    """Add summaries for losses.
    
    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.
     
    Args:
        :param total_loss: Total loss from loss().
    
    Returns:
        :return loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name + '_raw', l)
        # tf.summary.scalar(l.op.name + '_average', loss_averages.average(l))

    return loss_averages_op


def train_func(data_num, total_loss, global_step):
    # Variables that affect learning rate.
    num_batches_per_epoch = data_num / BATCH_SIZE
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(INI_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
    tf.summary.scalar('learning_rate', lr)

    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
          tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op
