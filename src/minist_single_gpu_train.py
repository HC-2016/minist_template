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
"""

import os
import time

import mnist
import numpy
import tensorflow as tf

from src import mnist_input


def train():
    """Train MNIST for a number of steps."""
    with tf.Graph().as_default():
        # Create a variable to count the number of train() calls. For multi-GPU programs, this equals the
        # number of batches processed * num_gpus.
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)

        # Input images and labels.
        images, labels = mnist.distorted_inputs()

        # Build a Graph that computes predictions from the inference model.
        logits = mnist.inference(images, hidden1_units=128, hidden2_units=32, wd=0.0)

        # Add to the Graph the Ops for loss calculation.
        total_loss = mnist.loss_func(logits, labels)

        # Add to the Graph the Ops that calculate and apply gradients.
        train_op = mnist.train_func(data_num=mnist_input.TRAIN_DATA_NUM, total_loss=total_loss, global_step=global_step)

        # Add the Op to compare the logits to the labels during evaluation.
        # eval_correct =mnist.evaluation(logits, labels)

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()

        # Build the summary Tensor based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        # The op for initializing the variables.
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())

        with tf.Session() as sess:
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
                if (step + 1) % 1000 == 0 or (step + 1) == mnist.MAX_STEPS:
                    checkpoint_file = os.path.join(mnist_input.LOG_DIR, 'model.ckpt')
                    saver.save(sess, checkpoint_file, global_step=step)
                    # # Evaluate against the training set.
                    # print('Training Data Eval:')
                    # do_eval(sess,
                    #         eval_correct,
                    #         images_placeholder,
                    #         labels_placeholder,
                    #         data_sets.train)
                    # # Evaluate against the validation set.
                    # print('Validation Data Eval:')
                    # do_eval(sess,
                    #         eval_correct,
                    #         images_placeholder,
                    #         labels_placeholder,
                    #         data_sets.validation)
                    # # Evaluate against the test set.
                    # print('Test Data Eval:')
                    # do_eval(sess,
                    #         eval_correct,
                    #         images_placeholder,
                    #         labels_placeholder,
                    #         data_sets.test)

            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    # Load data.
    train_set, val_set, test_set = mnist_input.load_data()

    # Convert to Examples and write the result to TFRecords.
    mnist_input.convert_to(train_set, 'train')
    mnist_input.convert_to(val_set, 'validation')
    mnist_input.convert_to(test_set, 'test')

    train()
