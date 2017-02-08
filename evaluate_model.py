import tensorflow as tf
import os
import glob
import vng_model as md
import numpy as np
import csv

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('checkpoint_dir', '',
                           """Direction where the trained weights of model is save""")

tf.app.flags.DEFINE_string('eval_data_path', '',
                           """The direction of folder that contains evaluation data-set""")

tf.app.flags.DEFINE_integer('batch_size', 100,
                            """The size of a data batch""")

tf.app.flags.DEFINE_string('output_path', '',
                           """The direction of folder that contains the output of model""")


def read_data(filename_queue, height, width, channels):
    reader = tf.WholeFileReader()

    filename, value = reader.read(filename_queue)

    image_sample = tf.image.decode_jpeg(value, channels=channels)

    image_sample = tf.expand_dims(image_sample, 0)
    image_sample = tf.image.resize_bilinear(image_sample, [height, width])
    image_sample = tf.squeeze(image_sample, [0])

    return image_sample, filename


def eval_model():
    """
    The function evaluate the model
    :return:
    """

    eval_data_path = []
    for path in glob.iglob(os.path.join(FLAGS.eval_data_path, '*.jpeg')):
        eval_data_path.append(path)

    with tf.Graph().as_default() as g:

        filename_queue = tf.train.string_input_producer(eval_data_path)

        sample, filename = read_data(filename_queue, 224, 224, 3)

        batch_samples, batch_filename = tf.train.batch([sample, filename],
                                                       batch_size=FLAGS.batch_size,
                                                       capacity=FLAGS.batch_size,
                                                       name='input_test_data')

        # Build the VNG model
        x = tf.placeholder(tf.float32, (None, 224, 224, 3), name='input_features')
        y_ = tf.placeholder(tf.int32, (None,), name='labels')

        logit = md.inference_resnet(x, False)

        hat_y = tf.arg_max(logit, 1, name='predict_label')

        # Load trained weights into model
        ckpt_model = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)

        saver_model = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

        # Run
        coord = tf.train.Coordinator()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            if ckpt_model and ckpt_model.model_checkpoint_path:
                saver_model.restore(sess, ckpt_model.model_checkpoint_path)
                print('Load trained weights successfully!')

            else:
                print('No checkpoint found!')

            num_iter = len(eval_data_path) / FLAGS.batch_size

            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            with open(FLAGS.output_path, "wb") as f:
                writer = csv.writer(f)

                for _ in range(num_iter):
                    batch_images, batch_name = sess.run([batch_samples, batch_filename])

                    predicted_lb = sess.run(hat_y, feed_dict={x:batch_images})

                    result_model = np.column_stack((np.array(batch_name),
                                                    np.array(predicted_lb)))

                    writer.writerows(result_model)

            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=5)
            sess.close()
