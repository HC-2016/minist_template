"""
@File         : minist_single_gpu_train.py
@Time         : 2017/6/20
@Author       : Chen Huang
@Update       : 
@Discription  : This is a template for tensorflow program using single GPU.

This program trains a fully-connection network on MNIST, It involves the following standard pipeline: 
    1. download and extract data. 
    2. convert original data to standard tensorflow data '.tfrecords'. 
    3. construct the graph and train the network:
        3.1 generate batch data.
        3.2 inference
        3.3 cal loss
        3.4 train
        3.5 evaluate
    4. save the model

The code is refered from the following codes:
    1. models-master\tutorials\image\cifar10.
    2. tensorflow-master\tensorflow\examples\tutorials\mnist\fully_connected_feed.py.
    
# visulization on TensorBoard 
$ tensorboard --logdir logs 
"""

import os
import time

from src import mnist
import numpy
import tensorflow as tf

from src import mnist_input


def evaluate(type):
    """ Eval MNIST for a number of steps."
    
    :param type: str - 'train'/'val'/'test'
    """
    assert type in ['train', 'val', 'test']

    graph = tf.Graph()
    with graph.as_default() as g:
        # Input images and labels.
        if type == 'train':
            filenames = [os.path.join(mnist_input.DATA_DIR, 'validation.tfrecords')]
            num_data = mnist_input.TRAIN_DATA_NUM
        elif type == 'val':
            filenames = [os.path.join(mnist_input.DATA_DIR, 'validation.tfrecords')]
            num_data = mnist_input.VAL_DATA_NUM
        elif type == 'test':
            filenames = [os.path.join(mnist_input.DATA_DIR, 'test.tfrecords')]
            num_data = mnist_input.TEST_DATA_NUM

        images, labels = mnist.inputs(filenames=filenames)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = mnist.inference(images, hidden1_units=128, hidden2_units=32, wd=0.0)

        # Calculate predictions.
        top_k_op = tf.nn.in_top_k(logits, labels, 1)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(
            mnist.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(os.path.join(mnist_input.LOG_DIR, type), g)

    with tf.Session(graph=graph) as sess:
        ckpt = tf.train.get_checkpoint_state(mnist_input.LOG_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/cifar10_train/model.ckpt-0,
            # extract global_step from it.
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            return

        # Start the queue runners.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        num_iter = int(numpy.ceil(num_data / mnist.BATCH_SIZE))
        true_count = 0  # Counts the number of correct predictions.
        total_sample_count = num_iter * mnist.BATCH_SIZE
        step = 0
        while step < num_iter and not coord.should_stop():
            predictions = sess.run([top_k_op])
            true_count += numpy.sum(predictions)
            step += 1

        # Compute precision @ 1.
        precision = true_count / total_sample_count
        print('%s:\t precision @ 1 = %.3f' % (type, precision))

        summary = tf.Summary()
        summary.ParseFromString(sess.run(summary_op))
        summary.value.add(tag='Precision_'+type+'@ 1', simple_value=precision)
        summary_writer.add_summary(summary, global_step)

        coord.request_stop()
        coord.join(threads)


def train():
    """Train MNIST for a number of steps."""
    graph = tf.Graph()
    with graph.as_default():
        # Create a variable to count the number of train() calls. For multi-GPU programs, this equals the
        # number of batches processed * num_gpus.
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)

        # Input images and labels.
        filenames = [os.path.join(mnist_input.DATA_DIR, 'train.tfrecords')]
        images, labels = mnist.distorted_inputs(filenames)

        # Build a Graph that computes predictions from the inference model.
        logits = mnist.inference(images, hidden1_units=128, hidden2_units=32, wd=0.0)

        # Add to the Graph the Ops for loss calculation.
        total_loss = mnist.loss_func(logits, labels)

        # Add to the Graph the Ops that calculate and apply gradients.
        train_op = mnist.train_func(data_num=mnist_input.TRAIN_DATA_NUM, total_loss=total_loss, global_step=global_step)

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()

        # Build the summary Tensor based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        # The op for initializing the variables.
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())

    with tf.Session(graph=graph) as sess:
        # Initialize the variables (the trained variables and the epoch counter).
        sess.run(init_op)

        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # Instantiate a SummaryWriter to output summaries and the Graph.
        summary_writer = tf.summary.FileWriter(mnist_input.LOG_DIR, sess.graph)

        # Start the training loop.
        for step in range(mnist.MAX_STEPS):
            start_time = time.time()
            _, loss_value = sess.run([train_op, total_loss])
            duration = time.time() - start_time
            assert not numpy.isnan(loss_value), 'Model diverged with loss = NaN'

            # Write the summaries and print an overview fairly often.
            if step % 100 == 0:
                # Print status to stdout.
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                # Update the events file.
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

            # Save a checkpoint and evaluate the model periodically.
            if step > 0 and (step % 500 == 0 or (step + 1) == mnist.MAX_STEPS):
                checkpoint_file = os.path.join(mnist_input.LOG_DIR, 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=step)

                evaluate(type='train')
                evaluate(type='val')


        coord.request_stop()
        coord.join(threads)

        evaluate(type='test')


if __name__ == '__main__':
    # Load data.
    train_set, val_set, test_set = mnist_input.load_data()

    # Convert to Examples and write the result to TFRecords.
    mnist_input.convert_to(train_set, 'train')
    mnist_input.convert_to(val_set, 'validation')
    mnist_input.convert_to(test_set, 'test')

    train()
