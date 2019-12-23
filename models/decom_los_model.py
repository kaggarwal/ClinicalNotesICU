'''
make sure you comment out while True loop for multitask reader in
multitask model util's generator code. Else it will consume all you RAM. :)

compute scp mimic3-benchmarks/mimic3models/multitask/utils.py
tensorflow-machine-vm:~/mimic3-text/mimic3-benchmarks/mimic3models/multitask/ --project mimic3

'''
import numpy as np
import pickle
from tensorflow.contrib import rnn
import tensorflow as tf
import utils
import random
import os
import sys
import math

from mimic3models.multitask import utils as mt_utils
from mimic3benchmark.readers import MultitaskReader
from mimic3models.preprocessing import Discretizer, Normalizer

from text_utils import TextReader, merge_text_events_with_timeseries

tf.logging.set_verbosity(tf.logging.INFO)
tf.logging.info("*** Loaded Data ***")

conf = utils.get_config()
args = utils.get_args()
log = utils.get_logger(args['log_file'])
vectors, word2index_lookup = utils.get_embedding_dict(conf)
lookup = utils.lookup

model_name = args['model_name']
assert model_name in ['baseline', 'text_cnn', 'text_only']

problem_type = args['problem_type']
assert problem_type in ['los', 'decom']

# let's set pad token to zero padding instead of random padding.
# might be better for attention as it will give minimum value.
if conf.padding_type == 'Zero':
    tf.logging.info("Zero Padding..")
    vectors[lookup(word2index_lookup, '<pad>')] = 0

train_reader = MultitaskReader(dataset_dir=os.path.join(
    conf.multitask_path, 'train'), listfile=os.path.join(conf.multitask_path, 'train_listfile.csv'))
val_reader = MultitaskReader(dataset_dir=os.path.join(
    conf.multitask_path, 'train'), listfile=os.path.join(conf.multitask_path, 'val_listfile.csv'))
test_reader = MultitaskReader(dataset_dir=os.path.join(
    conf.multitask_path, 'test'), listfile=os.path.join(conf.multitask_path, 'test_listfile.csv'))

discretizer = Discretizer(timestep=conf.timestep,
                          store_masks=True,
                          impute_strategy='previous',
                          start_time='zero')

discretizer_header = discretizer.transform(
    train_reader.read_example(0)["X"])[1].split(',')
cont_channels = [i for (i, x) in enumerate(
    discretizer_header) if x.find("->") == -1]

# choose here which columns to standardize
normalizer = Normalizer(fields=cont_channels)
normalizer_state = conf.normalizer_state
if normalizer_state is None:
    normalizer_state = 'los_ts{}.input_str:previous.start_time:zero.n5e4.normalizer'.format(
        conf.timestep)
    normalizer_state = os.path.join(
        os.path.dirname(__file__), normalizer_state)
normalizer.load_params(normalizer_state)

tf.logging.info(str(vars(conf)))
tf.logging.info(str(args))

number_epoch = int(args['number_epoch'])
batch_size = int(args['batch_size'])

if args['mode'] in['train', 'eval']:
    sp = True if args['mode'] == 'eval' else conf.small_part
    train_data_gen = mt_utils.BatchGen(reader=train_reader, discretizer=discretizer,
                                       normalizer=normalizer, batch_size=batch_size,
                                       shuffle=True, return_names=True, ihm_pos=48, partition='custom',
                                       target_repl=False, small_part=sp)
    eval_data_gen = mt_utils.BatchGen(reader=val_reader, discretizer=discretizer,
                                      normalizer=normalizer, batch_size=batch_size,
                                      shuffle=True, return_names=True, ihm_pos=48, partition='custom',
                                      target_repl=False, small_part=conf.small_part)

text_reader = TextReader(
    conf.textdata_fixed, conf.starttime_path, conf.maximum_number_events)
text_reader_test = TextReader(
    conf.test_textdata_fixed, conf.test_starttime_path)


def get_data(train_data_gen_, eval_data_gen_, text_reader_):
    data = []
    try:
        while True:
            d = train_data_gen_.next(return_y_true=True)
            data.append(d)
    except StopIteration:
        pass

    eval_data = []
    try:
        while True:
            d = eval_data_gen_.next(return_y_true=True)
            eval_data.append(d)
    except StopIteration:
        pass

    data, data_event_lens = merge_text_events_with_timeseries(
        problem_type, data, text_reader_, word2index_lookup, conf.max_len)
    eval_data, eval_data_event_lens = merge_text_events_with_timeseries(
        problem_type, eval_data, text_reader_, word2index_lookup, conf.max_len)

    tf.logging.info("Training data length: {}, Eval data length: {}".format(
        len(data), len(eval_data)))

    return data, eval_data


if args['mode'] in['train', 'eval']:
    data, eval_data = get_data(train_data_gen, eval_data_gen, text_reader)

if args['mode'] == 'test':
    test_data_gen = mt_utils.BatchGen(reader=test_reader, discretizer=discretizer,
                                      normalizer=normalizer, batch_size=batch_size,
                                      shuffle=False, return_names=True, ihm_pos=48, partition='custom',
                                      target_repl=False, small_part=conf.small_part)
    test_data = []
    try:
        while True:
            d = test_data_gen.next(return_y_true=True)
            test_data.append(d)
    except StopIteration:
        tf.logging.info("Generated test data!")
        pass
    test_data, _ = merge_text_events_with_timeseries(
        problem_type, test_data, text_reader_test, word2index_lookup, conf.max_len, False, 'Test_Text.pkl')


def TimeSpreadConv1D(sentence_features_i, time_mask_i):
    A = tf.transpose(sentence_features_i)
    B = tf.expand_dims(time_mask_i, axis=1)
    C = tf.multiply(A, B)
    #C = tf.math.reduce_max(C, axis=2, keep_dims=False)
    C = tf.math.reduce_sum(C, axis=2, keepdims=False) / \
        (tf.math.reduce_sum(time_mask_i, axis=1, keepdims=True) + 1e-8)
    return C


def get_text_features(embeds, dropout_keep_prob, time_mask, is_train):
    shapes = tf.shape(embeds)  # [None, None, None, 300]
    # [None, None, 300]
    embeds = tf.reshape(embeds, (-1, shapes[2], vectors.shape[1]))
    sizes = range(2, 5)
    result_tensors = []
    for ngram_size in sizes:
        # 256 -> 2,3 best yet.
        text_conv1d = tf.layers.conv1d(inputs=embeds, filters=conf.conv1d_channel_size, kernel_size=ngram_size,
                                       strides=1, padding='same', dilation_rate=1,
                                       activation='relu', name='Text_Conv_1D_N{}'.format(ngram_size),
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01))

        text_conv1d = tf.reduce_max(text_conv1d, axis=1, keepdims=False)
        result_tensors.append(text_conv1d)
    text_embeddings = tf.concat(result_tensors, axis=1)
    # text_embeddings = tf.nn.leaky_relu(text_embeddings, alpha=0.1)
    # text_embeddings = tf.layers.batch_normalization(
    #    text_embeddings, axis=1, training=is_train)
    # text_embeddings = tf.nn.dropout(
    #    text_embeddings, keep_prob=dropout_keep_prob)
    sentence_features = tf.reshape(
        text_embeddings, (shapes[0], shapes[1], conf.conv1d_channel_size*len(sizes)))

    # using tf map_fn multiply sentence_features to time_mask and take maximum for each row in batch.
    map_fn_elems = (sentence_features, time_mask)
    final_text_features = tf.map_fn(lambda x: TimeSpreadConv1D(
        x[0], x[1]), map_fn_elems, dtype=tf.float32)

    # text_lstm = rnn.LSTMCell(num_units=128, name='text_rnn')
    # final_text_features, _ = tf.nn.dynamic_rnn(text_lstm, final_text_features, time_major=False, dtype=tf.float32)
    text_feature_dim = conf.conv1d_channel_size*len(sizes)
    return final_text_features, text_feature_dim


tf.reset_default_graph()
# define placeholders
X = tf.placeholder(shape=(None, None, 76),
                   dtype=tf.float32, name='X')  # B*48*76
y = tf.placeholder(shape=(None, None, 1), dtype=tf.int32, name='y')
mask = tf.placeholder(shape=(None, None), dtype=tf.float32, name='mask')
is_training = tf.placeholder(dtype=tf.bool)
dropout_keep_prob = tf.placeholder(dtype=tf.float32, name='dropout_kp')

rnn_cell = rnn.LSTMCell(num_units=conf.rnn_hidden_units)
rnn_outputs, _ = tf.nn.dynamic_rnn(rnn_cell, X,
                                   time_major=False,
                                   dtype=tf.float32)
if model_name == 'baseline':
    output_features_reshaped = tf.reshape(
        rnn_outputs, [-1, conf.rnn_hidden_units])

if model_name in ['text_cnn', 'text_only']:
    # text side
    tf.logging.info("Adding graph for text_features!")
    time_mask = tf.placeholder(
        shape=(None, None, None), dtype=tf.float32, name='time_mask')  # B*D*S
    text = tf.placeholder(shape=(None, None, None),
                          dtype=tf.int32, name='text_ts')  # B*D*S
    # clip time mask for equal weightage
    tf.logging.info("Using decay with lambda = %f" % args['decay'])
    time_mask_decay = tf.math.exp(-args['decay']*time_mask)
    time_mask_clipped = tf.clip_by_value(
        time_mask, 0, 1, name='UniformWeigthTM')
    time_mask_final = time_mask_decay * time_mask_clipped

    # define variables
    W = tf.get_variable(name="W", shape=vectors.shape,
                        initializer=tf.constant_initializer(vectors), trainable=False)
    embeds = tf.nn.embedding_lookup(W, text)  # B*D*S*E
    text_features, text_feature_dim = get_text_features(
        embeds, dropout_keep_prob, time_mask_final, is_training)
    tf.logging.info(
        "Total dimenstion of text feature: {}".format(text_feature_dim))

    if model_name == 'text_cnn':
        tf.logging.info("Text Convolution Model for Decompensation")
        output_features = tf.concat(
            [rnn_outputs, text_features], axis=2, name='rnn_text_concat')
        output_features_reshaped = tf.reshape(
            output_features, [-1, conf.rnn_hidden_units + text_feature_dim])
    elif model_name == 'text_only':
        tf.logging.info("Training text only Model for Decompensation")
        output_features_reshaped = tf.reshape(
            text_features, [-1, text_feature_dim])

if problem_type == 'decom':
    output_units = 1
elif problem_type == 'los':
    output_features_reshaped = tf.nn.dropout(
        output_features_reshaped, keep_prob=dropout_keep_prob)
    output_units = 10
logits = tf.layers.dense(inputs=output_features_reshaped,
                         units=output_units, activation=None, use_bias=True,
                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                         kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01))
logits = tf.reshape(logits, [tf.shape(X)[0], tf.shape(X)[1], -1])

if problem_type == 'decom':
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
    probs = tf.math.sigmoid(logits)
elif problem_type == 'los':
    # dropping last dimension of 1 by squeezing.
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=tf.squeeze(y, axis=2), logits=logits)
    probs = tf.math.softmax(logits)
# sigmoid_loss = tf.nn.weighted_cross_entropy_with_logits(
#    targets=y, logits=logits, pos_weight=conf.mortality_class_ce_weigth)
loss = tf.squeeze(loss)
loss = tf.multiply(loss, mask)
loss = tf.math.reduce_sum(loss) / tf.math.reduce_sum(mask) + \
    tf.reduce_mean(tf.losses.get_regularization_loss())

with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate=conf.learning_rate)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
if problem_type == 'decom':
    with tf.name_scope('train_metric'):
        aucroc, update_aucroc_op = tf.metrics.auc(
            labels=y, predictions=probs, weights=mask)
        aucpr, update_aucpr_op = tf.metrics.auc(labels=y, predictions=probs, weights=mask,
                                                curve="PR")

    with tf.name_scope('valid_metric'):
        val_aucroc, update_val_aucroc_op = tf.metrics.auc(
            labels=y, predictions=probs, weights=mask)
        val_aucpr, update_val_aucpr_op = tf.metrics.auc(labels=y, predictions=probs, weights=mask,
                                                        curve="PR")
else:
    val_aucpr = None
    val_aucroc = None
    update_val_aucpr_op = None
    update_val_aucroc_op = None

init = tf.group(tf.global_variables_initializer(),
                tf.local_variables_initializer())
gpu_config = tf.ConfigProto(device_count={'GPU': 1})

saver = tf.train.Saver()


def validate(eval_data, sess, loss, val_aucpr, val_aucroc, update_val_aucpr_op, update_val_aucroc_op, save, last_best, saver):
    loss_list = []
    # sess.run(tf.variables_initializer([v for v in tf.local_variables() if 'valid_metric' in v.name]))
    sess.run(tf.local_variables_initializer())

    runnable_nodes = []
    if problem_type == 'decom':
        runnable_nodes.extend(
            [loss, probs, update_val_aucpr_op, update_val_aucroc_op])
        metric_obj = utils.AUCPRperHour()
    elif problem_type == 'los':
        runnable_nodes.extend([loss, probs])
        metric_obj = utils.KappaPerHour()

    for batch in eval_data:
        fd = get_feed_dict_from_batch(batch, model_name, False, 1.0)
        node_outputs = sess.run(runnable_nodes, fd)
        loss_value = node_outputs[0]
        probablities = node_outputs[1]
        loss_list.append(loss_value)
        if problem_type == 'decom':
            metric_obj.add(probablities, fd[y], fd[mask])
        elif problem_type == 'los':
            predictions = np.argmax(probablities, axis=2)
            predictions = np.expand_dims(predictions, axis=2)
            metric_obj.add(predictions, fd[y], fd[mask])

    if problem_type == 'decom':
        final_aucpr = sess.run(val_aucpr)
        final_aucroc = sess.run(val_aucroc)
        _, _, _, sklean_aucpr = metric_obj.get()
        if args['mode'] == 'test':
            tf.logging.info(
                "Saving predictions in file : {}".format("decome_"+model_name))
            metric_obj.save("decome_"+model_name)

        tf.logging.info("Problem: %s Validation Loss: %f - AUCPR: %f - AUCPR-SKLEARN: %f - AUCROC: %f" %
                        (problem_type, np.mean(loss_list), final_aucpr, sklean_aucpr, final_aucroc))

        changed = False
        if final_aucpr > last_best:
            changed = True
            if save:
                save_path = saver.save(sess, args['checkpoint_path'])
                tf.logging.info(
                    "Best Model saved in path: %s" % save_path)
        return max(last_best, final_aucpr), changed
    elif problem_type == 'los':
        _, _, _, kappa_score = metric_obj.get()
        if args['mode'] == 'test':
            tf.logging.info(
                "Saving predictions in file : {}".format("decome_"+model_name))
            metric_obj.save("decome_"+model_name)

        tf.logging.info("Problem: %s Validation Loss: %f - Sklearn-Kappa: %f" %
                        (problem_type, np.mean(loss_list), kappa_score))

        changed = False
        if kappa_score > last_best:
            changed = True
            if save:
                save_path = saver.save(sess, args['checkpoint_path'])
                tf.logging.info(
                    "Best Model saved in path: %s" % save_path)
        return max(last_best, kappa_score), changed


def get_feed_dict_from_batch(batch, model_name, training, keep_prob=1):
    fd = {X: batch['X'],
          y: batch['Output'],
          mask: batch['Mask'],
          is_training: training,
          dropout_keep_prob: keep_prob}
    if model_name in ['text_cnn', 'text_only']:
        fd[time_mask] = batch['TimeMask']
        fd[text] = batch['Texts']
    return fd


last_best = -1
with tf.Session(config=gpu_config) as sess:
    sess.run(init)

    if bool(int(args['load_model'])):
        saver.restore(sess, args['checkpoint_path'])
        if args['mode'] == 'train':
            tf.logging.info('Evaluating model to get the best yet.')
            last_best = validate(
                eval_data, sess, loss, val_aucpr, val_aucroc, update_val_aucpr_op,
                update_val_aucroc_op, False, last_best, saver)

    if args['mode'] == 'eval':
        assert bool(int(args['load_model']))
        last_best, _ = validate(
            eval_data, sess, loss, val_aucpr, val_aucroc, update_val_aucpr_op,
            update_val_aucroc_op, False, last_best, saver)
        tf.logging.info('Evaluation completed.')
        sys.exit(0)

    if args['mode'] == 'test':
        assert bool(int(args['load_model']))
        val_aucpr, val_aucroc, update_val_aucpr_op, update_val_aucroc_op = [
            None]*4
        last_best, _ = validate(
            test_data, sess, loss, val_aucpr, val_aucroc, update_val_aucpr_op,
            update_val_aucroc_op, False, last_best, saver)
        tf.logging.info('Testing completed.')
        sys.exit(0)

    early_stopping_count = 0
    runnable_nodes = []
    if problem_type == 'decom':
        runnable_nodes.extend(
            [train_op, loss, update_aucpr_op, update_aucroc_op])
        metric_obj = utils.AUCPRperHour()
    elif problem_type == 'los':
        runnable_nodes.extend([train_op, loss, probs])

    print(runnable_nodes)
    for epoch in range(number_epoch):
        if problem_type == 'decom':
            metric_obj = utils.AUCPRperHour()
        elif problem_type == 'los':
            metric_obj = utils.KappaPerHour()

        loss_list = []
        for batch in data:
            fd = get_feed_dict_from_batch(
                batch, model_name, True, conf.dropout)
            node_outputs = sess.run(
                runnable_nodes, fd)
            loss_value = node_outputs[1]
            loss_list.append(loss_value)

            if problem_type == 'los':
                probablities = node_outputs[2]
                predictions = np.argmax(probablities, axis=2)
                predictions = np.expand_dims(predictions, axis=2)
                metric_obj.add(predictions, fd[y], fd[mask])

        if problem_type == 'decom':
            aucpr_value = node_outputs[2]
            aucroc_value = node_outputs[3]
            current_aucroc = sess.run(aucroc)
            current_aucpr = sess.run(aucpr)

            tf.logging.info("Epoch %d Loss: %f - AUCPR: %f - AUCROC: %f" %
                            (epoch, np.mean(loss_list), current_aucpr, current_aucroc))
        elif problem_type == 'los':
            _, _, _, train_kappa = metric_obj.get()
            tf.logging.info("Epoch %d Loss: %f Kappa: %f" %
                            (epoch, np.mean(loss_list), train_kappa))

        # reset aucroc and aucpr local variables
        sess.run(tf.local_variables_initializer())
        last_best, changed = validate(eval_data, sess, loss, val_aucpr, val_aucroc,
                                      update_val_aucpr_op, update_val_aucroc_op, True,
                                      last_best, saver)
        if changed == False:
            early_stopping_count += 1
            tf.logging.info("Didn't improve! : Count: " +
                            str(early_stopping_count))
        else:
            early_stopping_count = 0

        if early_stopping_count >= 10:
            tf.logging.info(
                "Value didn't change from last 10 epochs, early stopping: " + str(early_stopping_count))
            break
