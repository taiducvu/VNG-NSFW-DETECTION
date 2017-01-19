from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib.slim.python.slim.nets import resnet_v1

# ------ Import for resnet ------ #
slim = tf.contrib.slim

INITIAL_LEARNING_RATE = 0.1
LEARNING_RATE_DECAY_FACTOR = 0.9
NUM_CLASSES = 2

trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)

# ------- THE NAME OF LAYERS IN THE INCEPTION V3 ------- #
tensor_name_input_jpeg = "DecodeJpeg/contents:0"
tensor_name_input_image = "DecodeJpeg:0"
tensor_name_transfer_layer = "pool_3:0"

# ------- FLAGS ------- #
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('saved_graph_dir',
                           '/home/taivu/workspace/NudityDetection/Trained_weight/tensorflow_inception_graph.pb',
                           "The direction where the graph file is saved")

tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using Float-16""")

tf.app.flags.DEFINE_boolean('using_gpu', False,
                            """whether the model is run by GPU""")

tf.app.flags.DEFINE_integer('num_test_sample', 2000,
                            """the number of test samples""")

tf.app.flags.DEFINE_integer('num_epochs_per_decay', 2000,
                            """The numper of epochs in each decay""")


def _create_feed_dict(image_path=None, image=None):
    if image is not None:
        feed_dict = {tensor_name_input_image: image}

    elif image_path is not None:
        image_data = tf.gfile.FastGFile(image_path, 'rb').read()

        feed_dict = {tensor_name_input_jpeg: image_data}

    else:
        raise ValueError("Either image or image_path must be set")

    return feed_dict


def transfer_learning(sess, image_path = None, image = None, fist_load=True):
    """
        The raw images will be passed through the Inception V3 model as a feature extraction
    process.

    :param sess:
    :param image_path:
    :param image: a tensor of size [299, 299, 3]
    :return:
    """
    # Load the graph of Inception V3 from file

    if fist_load:
        with tf.gfile.FastGFile(FLAGS.saved_graph_dir, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')

    feed_dict = _create_feed_dict(image_path, image)

    transfer_layer = sess.graph.get_tensor_by_name(tensor_name_transfer_layer)

    tf_value = sess.run(transfer_layer, feed_dict=feed_dict)

    return tf_value


def _initialize_variable(name, shape, initializer):
    """
    Helper to create a variable on CPU memory or GPU memory
    :param name: name of the variable
    :param shape: shape of the variable
    :param initializer: initializer for variable
    :return:
    a variable tensor
    """

    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    if FLAGS.using_gpu:
        with tf.device('/gpu:0'):
            var = tf.get_variable(name, shape, dtype, initializer)

    else:
        with tf.device('/cpu:0'):
            var = tf.get_variable(name, shape, dtype, initializer)

    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    """
    Helper to create an initialized variable with weight decay
    :param name:
    :param shape:
    :param stddev:
    :param wd:
    :return:
    """

    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = _initialize_variable(name, shape, tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))

    if wd is not None:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)

    return var


def inference(features):
    """
    In this
    :param features: The extracted features of size [None, 2048]
    :return:
    """

    with tf.variable_scope('FC_1') as scope:
        weights = _variable_with_weight_decay('weights',
                                              [2048, 1024],
                                              0.04, 0.004)

        biases = _initialize_variable('biases',
                                      [1024],
                                      tf.constant_initializer(0.1))

        active_1 = tf.nn.relu(tf.matmul(features, weights) + biases, name=scope.name)

    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay('weights',
                                              [1024, NUM_CLASSES],
                                              stddev= 1/1024.0, wd=0.0)
        biases = _initialize_variable('biases',
                                       [NUM_CLASSES],
                                       tf.constant_initializer(0.0))

        softmax_linear = tf.add(tf.matmul(active_1, weights), biases, name=scope.name)

    return softmax_linear


def loss(logits, labels, resnet_var_ls=None):
    """
        The function compute the cross-entropy function between the output of model and
    the real label of training sample.

    :param logits: A tensor of size [None, 2]
    :param labels: A tensor of size [None,]
    :return:
    a tensor of size [None,]
    """
    if resnet_var_ls is not None:
        for var in resnet_var_ls:
            weight_decay = tf.mul(tf.nn.l2_loss(var), 0.0)
            tf.add_to_collection('losses', weight_decay)

    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def inference_resnet(features, is_training=True, reuse=None, is_log=None):
    """
        The function generates the model for classifying a image into the 'Nudity' class or
     the 'Normal' class. We replace the last 1000-class layer of the resnet model with two fully
     collected layers.

    :param features: a tensor of size [None, 224, 224, 3]
    :param is_training: whether being training model
    :param reuse:
    :return:
    a tensor of size [None, 2]
    """
    with slim.arg_scope(resnet_v1.resnet_arg_scope(is_training)):
        net, end_points = resnet_v1.resnet_v1_50(inputs=features, reuse=reuse)

    net = tf.squeeze(net, [1, 2])

    # ---- ADD a new fully-connected layer ----- #
    with tf.variable_scope('additional_layers', reuse=reuse):
        # with tf.variable_scope('FC_1') as scope:
        #     weights = _variable_with_weight_decay('weights',
        #                                           [2048, 1024],
        #                                           0.04, 0.004)
        #
        #     biases = _initialize_variable('biases',
        #                                   [1024],
        #                                   tf.constant_initializer(0.1))
        #
        #     activate_1 = tf.nn.relu(tf.matmul(net, weights) + biases, name=scope.name)

        with tf.variable_scope('softmax', reuse=reuse) as scope:
            weights = _variable_with_weight_decay('weights',
                                                  [2048, 2],
                                                  stddev=1 / 2048.0, wd=0.0)

            biases = _initialize_variable('biases',
                                          [2],
                                          tf.constant_initializer(0.0))

           # softmax_classifier = tf.add(tf.matmul(activate_1, weights), biases, name=scope.name)
            softmax_classifier = tf.add(tf.matmul(net, weights), biases, name=scope.name)

    if is_log is not None:
        var_ls = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        log_variables(var_ls)

    return softmax_classifier

def log_variables(var_ls):
    for var in var_ls:
        tf.summary.histogram(var.name, var)
