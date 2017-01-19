from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import data as dt
import vng_model as md
import time
import csv
import math

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/home/taivu/workspace/Pycharm_Nudity_Detection/Dataset',
                           """Direction where the training set is""")

tf.app.flags.DEFINE_string('val_dir', '/home/taivu/workspace/Pycharm_Nudity_Detection/Dataset',
                           """Direction where the validation set is""")

tf.app.flags.DEFINE_integer('num_steps', 500000,
                            "The number of steps in updating the weights of models")

tf.app.flags.DEFINE_string('checkpoint_dir_resnet', '/home/taivu/workspace/Pycharm_Nudity_Detection/pretrain_weight',
                           """Direction where the checkpoint is""")

tf.app.flags.DEFINE_string('checkpoint_dir', '/home/taivu/workspace/Pycharm_Nudity_Detection/checkpoint_model',
                           """Direction where the checkpoint of model is saved""")

tf.app.flags.DEFINE_float('learning_rate', 1e-3,
                          """Learning rate for optimization""")

tf.app.flags.DEFINE_integer('num_train_sample', 8000,
                            """The number of training samples""")

tf.app.flags.DEFINE_integer('num_val_sample', 1156,
                            """The number of validate samples""")

tf.app.flags.DEFINE_integer('batch_size', 32,
                            "The size of a image batch")

tf.app.flags.DEFINE_float('weight_decay', 0.01,
                          """Weight decay""")

# Flags for validation process
tf.app.flags.DEFINE_boolean('use_val', True,
                            """Whether using the validation set in the training process""")

tf.app.flags.DEFINE_integer('val_batch_size', 128,
                            """The size of a validate data batch""")

# Logging the result
tf.app.flags.DEFINE_boolean('is_logging', True,
                            """Whether logging the result of training model""")

tf.app.flags.DEFINE_string('log_dir', '/home/taivu/workspace/Pycharm_Nudity_Detection/checkpoint_model',
                           """Direction where the log file is saved""")

tf.app.flags.DEFINE_string('summaries_dir', '/home/taivu/Dropbox/Pycharm_Nudity_Detection/log',
                           """Direction where the log tensorboard is saved""")


def set_flag(flag, value):
    flag.assign(value)


def train():
    """

    :return:
    """
    # Read data
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)
        train_flag = tf.Variable(True, trainable=False)

        train_path = os.path.join(FLAGS.train_dir, 'transfer_learning_train.tfrecords')
        val_path = os.path.join(FLAGS.val_dir, 'transfer_learning_val.tfrecords')

        tr_samples, tr_lb = dt.input_data(train_path, FLAGS.batch_size)
        val_samples, val_lb = dt.input_data(val_path, 1156, False)

        samples, labels = tf.cond(train_flag,
                                   lambda: (tr_samples, tr_lb),
                                   lambda: (val_samples, val_lb))

        samples = tf.squeeze(samples, [1, 2])

        logits = md.inference(samples)

        loss = md.loss(logits, labels)

        correct_predict = tf.equal(tf.cast(tf.arg_max(logits, 1), tf.int32), labels)

        val_acc = tf.reduce_mean(tf.cast(correct_predict, tf.float32))

        # train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
        train_step = tf.train.RMSPropOptimizer(1e-5).minimize(loss)

        coord = tf.train.Coordinator()

        format_str = ('%d step: %.2f (%.1f examples/sec; %0.3f sec/batch)')

        with tf.Session() as sess:
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            sess.run(tf.global_variables_initializer())

            # samp_batch, lb_batch = sess.run([tr_samples, tr_lb])
            # size: samp_batch [#batch_size, 1, 1, 2048]
            #        tr_lb [#batch_size,]

            for idx in range(FLAGS.num_epochs):
                start_time = time.time()
                _, loss_value = sess.run([train_step, loss])
                duration = time.time() - start_time
                examples_per_sec = FLAGS.batch_size / float(duration)
                sec_per_batch = float(duration)

                if idx % 10 == 0:
                    set_flag(train_flag, False)
                    acc = sess.run([val_acc])
                    set_flag(train_flag, True)
                    print('Validation accuracy: %.2f'%acc[0])

                print(format_str %(idx, loss_value, examples_per_sec, sec_per_batch))

            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=120)
            sess.close()


def train_resnet():
    """
        The function is used  trainto the model. We use the 'Stochastic Gradient Descent' algorithm to
    optimize the weights of model. More detail, we initialize them by using pre-trained weights of Resnet model.
    To train model, we freeze the first block of model and train the other. The learning rate in layers that belong to
    Resnet model is set 5 times larger than in additional layers.

    :return:
    """

    batch_ls = []
    for batch in range(2, 8):
        name_batch = '4000x224x224_batch_' + str(batch) + '.tfrecords'
        train_batch = os.path.join(FLAGS.train_dir, name_batch)
        batch_ls.append(train_batch)

    val_path = os.path.join(FLAGS.train_dir, '4000x224x224_batch_1.tfrecords')

    with tf.Graph().as_default() as g:
        # ------------------------- BUILD THE GRAPH OF MODEL ---------------------------- #
        x = tf.placeholder(tf.float32, (None, 224, 224, 3), name='input_features')
        y_ = tf.placeholder(tf.int32, (None,), name='labels')

        val_x = tf.placeholder(tf.float32, (None, 224, 224, 3), name='val_input_features')

        val_y = tf.placeholder(tf.int32, (None,), name='val_labels')

        tr_samples, tr_labels = dt.input_data(batch_ls, FLAGS.batch_size)

        val_samples, val_labels = dt.input_data([val_path], FLAGS.val_batch_size, False)

        logit = md.inference_resnet(x, is_log=FLAGS.is_logging)

        val_logit = md.inference_resnet(val_x, False, reuse=True, is_log=FLAGS.is_logging)

        # Define variables to output the predict of model and to evaluate one
        resnet_var_ls = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='resnet_v1_50')
        resnet_weight_ls = []
        for idx in range(0, 159, 3):
            resnet_weight_ls.append(resnet_var_ls[idx])

        loss = md.loss(logit, y_, resnet_weight_ls)

        v_loss = md.loss(val_logit, val_y, resnet_weight_ls)

        hat_y = tf.arg_max(val_logit, 1, name='predict_label')

        correct_pre = tf.equal(tf.cast(hat_y, tf.int32), val_y)

        accuracy = tf.reduce_mean(tf.cast(correct_pre, tf.float32))

        tf.summary.scalar('train_loss', loss) # Log the value of the train loss
        tf.summary.scalar('validate_loss', v_loss) # Log the value of validate loss
        tf.summary.scalar('validate_accuracy', accuracy) # Log the accuracy
        # ------------------------------------- END -------------------------------------- #

        # -------------------------------Optimizing process ------------------------------ #
        resnet_var_ls = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='resnet_v1_50')

        add_var_ls = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='additional_layers')

        opt_1 = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)

        opt_2 = tf.train.GradientDescentOptimizer(5*FLAGS.learning_rate)

        # Freeze the weights of from first to third blocks
        grads = tf.gradients(loss, resnet_var_ls[153:] + add_var_ls)

        # Do gradient descent only on a particular weight set
        num_opt_resnet_layers = len(resnet_var_ls[153:])

        grads_1 = grads[:num_opt_resnet_layers]  # Do gradient for Resnet's layers

        grads_2 = grads[num_opt_resnet_layers:]  # Do gradient for Additional layers

        train_opt_1 = opt_1.apply_gradients(zip(grads_1, resnet_var_ls[153:]))

        train_opt_2 = opt_2.apply_gradients(zip(grads_2, add_var_ls))

        train_opt = tf.group(train_opt_1, train_opt_2)
        # ------------------------------------- END -------------------------------------- #

        saver_my_model = tf.train.Saver(tf.all_variables(), max_to_keep=50)

        # ------------------ Support for loading the trained weights of Resnet ----------- #
        saver_resnet = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                                 scope='resnet_v1_50'))

        ckpt_resnet = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir_resnet)
        ###################################################################################

        # ------------------ TENSORBOARD --------------------------------
        merged = tf.summary.merge_all()
        train_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/train', graph=g)
        # test_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/test')

        coord = tf.train.Coordinator()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # ------ Load pre-trained weights of the Resnet model -------------------------- #
            if ckpt_resnet and ckpt_resnet.model_checkpoint_path:
                saver_resnet.restore(sess, ckpt_resnet.model_checkpoint_path)
                print('Load pre-trained weights of Resnet successfully!')

            else:
                print('Checkpoint of Resnet not found!')
            ####################################################################################

            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            format_str=('Step %d: %0.2f (%0.1f samples/sec; %0.3f secs/batch)')

            steps_per_epoch = int(math.ceil(float(FLAGS.num_train_sample)/FLAGS.batch_size))

            for idx in range(FLAGS.num_steps):

                tr_x, tr_y = sess.run([tr_samples, tr_labels])

                start_time = time.time()

                _, loss_value = sess.run([train_opt, loss], feed_dict={x: tr_x, y_: tr_y})

                duration = time.time() - start_time

                examples_per_sec = FLAGS.batch_size / float(duration)

                sec_per_batch = float(duration)

                print(format_str % (idx, loss_value, examples_per_sec, sec_per_batch))

                mean_val_acc = 0
                mean_tr_loss = 0
                if (idx + 1) % steps_per_epoch == 0 or idx == 0:
                    # Logging the performance of model in training process
                    if FLAGS.use_val and FLAGS.is_logging:
                        val_iter = int(math.ceil(FLAGS.num_val_sample)/FLAGS.val_batch_size)

                        for i in range(val_iter):
                            v_x, v_y = sess.run([val_samples, val_labels])

                            val_acc, val_loss, summary = sess.run([accuracy, v_loss, merged], feed_dict={x:tr_x,
                                                                                        y_:tr_y,
                                                                                        val_x: v_x,
                                                                                        val_y: v_y})
                            train_writer.add_summary(summary, idx)  # Log
                            if i == 0:
                                mean_val_acc = val_acc
                                mean_val_loss = val_loss

                            else:
                                mean_val_acc = 1.0/(i + 1)*(val_acc + i*mean_val_acc)
                                mean_val_loss = 1.0/(i + 1)*(val_loss + i*mean_val_loss)

                        print('Validation accuracy: %0.2f'%mean_val_acc)

                        for i in range(steps_per_epoch):
                            eval_tr_x, eval_tr_y = sess.run([tr_samples, tr_labels])

                            loss_value = sess.run(loss, feed_dict={x:eval_tr_x, y_:eval_tr_y})

                            if i == 0:
                                mean_tr_loss = loss_value
                            else:
                                mean_tr_loss = 1.0/(i+1)*(loss_value + i*mean_tr_loss)

                        # -------------------- Writing log-file ------------------------------
                        log_path = os.path.join(FLAGS.log_dir, 'result.csv')

                        if os.path.isfile(log_path) and idx == 0:
                            os.remove(log_path)

                        with open(log_path, 'a') as csvfile:
                            print('Writing data into csv file ...')

                            csv_writer = csv.writer(csvfile, delimiter=',',
                                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)

                            csv_writer.writerow([idx, mean_tr_loss, mean_val_loss, mean_val_acc])

                            print('Finish writing!')
                        # ---------------------------- END ------------------------------------

                    elif FLAGS.use_val:
                        val_acc, val_loss = sess.run([accuracy, loss], feed_dict={x: val_x, y_: val_y})
                        print('Validation accuracy: %0.2f' % val_acc)

                    else:
                        print('Set True for use_val flag to log the performance of model in training process!')

                    checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt')

                    saver_my_model.save(sess, checkpoint_path, global_step=idx)

            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=120)
            sess.close()


def main(argv=None):
    train_resnet()

if __name__ == '__main__':
    tf.app.run()
