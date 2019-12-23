from config import Config
from mimic3models import common_utils
from mimic3models.preprocessing import Discretizer, Normalizer
from mimic3benchmark.readers import InHospitalMortalityReader
from mimic3models.in_hospital_mortality import utils as ihm_utils
'''
In intensive care units, where patients come in with a wide range of health conditions, 
triaging relies heavily on clinical judgment. ICU staff run numerous physiological tests, 
such as bloodwork and checking vital signs, 
to determine if patients are at immediate risk of dying if not treated aggressively.
'''

import utils
import tensorflow as tf
from tensorflow.contrib import rnn
import pickle
import numpy as np
import sys
import random
import os

tf.logging.set_verbosity(tf.logging.INFO)
tf.logging.info("*** Loaded Data ***")

conf = utils.get_config()
args = utils.get_args()
log = utils.get_logger(args['log_file'])
vectors, word2index_lookup = utils.get_embedding_dict(conf)
lookup = utils.lookup
# let's set pad token to zero padding instead of random padding.
# might be better for attention as it will give minimum value.
if conf.padding_type == 'Zero':
    tf.logging.info("Zero Padding..")
    vectors[lookup(word2index_lookup, '<pad>')] = 0

tf.logging.info(str(vars(conf)))
tf.logging.info(str(args))

number_epoch = int(args['number_epoch'])
batch_size = int(args['batch_size'])

X = tf.placeholder(shape=(None, 48, 76), dtype=tf.float32, name='X')  # B*48*76
y = tf.placeholder(shape=(None), dtype=tf.float32, name='y')
text = tf.placeholder(shape=(None, None), dtype=tf.int32, name='text')  # B*L
dropout_keep_prob = tf.placeholder(dtype=tf.float32, name='dropout_kp')
T = tf.placeholder(dtype=tf.int32)
N = tf.placeholder(dtype=tf.int32)

W = tf.get_variable(name="W", shape=vectors.shape,
                    initializer=tf.constant_initializer(vectors), trainable=False)
embeds = tf.nn.embedding_lookup(W, text)

hidden_units = vectors.shape[1]

#avg_word_embedding_mode = bool(int(args['avg_we_model']))
#baseline = bool(int(args['baseline']))

model_name = args['model_name']
assert model_name in ['baseline', 'avg_we', 'transformer', 'cnn', 'text_only']
if model_name != 'baseline':
    if model_name == 'avg_we':
        tf.logging.info("Average Word Embeddings Model!")
        text_embeddings = tf.math.reduce_mean(embeds, axis=1, keepdims=False)
    elif model_name == 'transformer':
        tf.logging.info("Transformer Encoder Based Model!")
        key_masks = tf.expand_dims(
            tf.sign(tf.reduce_sum(tf.abs(embeds), axis=-1)), -1)
        embeds += utils.positional_encoding(text, T, N, num_units=hidden_units,
                                            zero_pad=False, scale=False, scope="enc_pe")
        embeds *= key_masks

        # Dropout
        embeds = tf.nn.dropout(embeds, keep_prob=dropout_keep_prob)
        enc = embeds
        # Blocks
        for i in range(conf.num_blocks):
            with tf.variable_scope("num_blocks_{}".format(i)):
                # Multihead Attention
                enc = utils.multihead_attention(queries=enc,
                                                keys=embeds,
                                                num_units=hidden_units,
                                                num_heads=10,
                                                dropout_rate=dropout_keep_prob,
                                                causality=False)

                # Feed Forward
                enc = utils.feedforward(enc, num_units=[
                    4*hidden_units, hidden_units])
        text_embeddings = tf.reduce_mean(enc, axis=1)
    else:
        tf.logging.info("1D Convolution Model")
        sizes = range(2, 5)
        result_tensors = []
        for ngram_size in sizes:
            # 256 -> 2,3 best yet.
            text_conv1d = tf.layers.conv1d(inputs=embeds, filters=256, kernel_size=ngram_size,
                                           strides=1, padding='same', dilation_rate=1,
                                           activation='relu', name='Text_Conv_1D_N{}'.format(ngram_size),
                                           kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01))
            text_conv1d = tf.reduce_max(text_conv1d, axis=1, keepdims=False)
            result_tensors.append(text_conv1d)
        text_embeddings = tf.concat(result_tensors, axis=1)
        text_embeddings = tf.nn.dropout(
            text_embeddings, keep_prob=dropout_keep_prob)

rnn_cell = rnn.LSTMCell(num_units=256)
rnn_outputs, _ = tf.nn.dynamic_rnn(rnn_cell, X,
                                   time_major=False,
                                   dtype=tf.float32)

#mean_rnn_outputs = tf.math.reduce_mean(rnn_outputs, axis=1, keepdims=False)
mean_rnn_outputs = rnn_outputs[:, -1, :]
if model_name == 'baseline':
    logit_X = mean_rnn_outputs
elif model_name == 'text_only':
    logit_X = text_embeddings
else:
    logit_X = tf.concat([text_embeddings, mean_rnn_outputs], axis=1)

logits_regularizer = tf.contrib.layers.l2_regularizer(scale=0.01)
logits = tf.layers.dense(inputs=logit_X, units=1, activation=None, use_bias=False,
                         kernel_initializer=tf.contrib.layers.xavier_initializer(), kernel_regularizer=logits_regularizer)

# logits = tf.layers.dense(inputs=mean_rnn_outputs,
#                         units=1, activation=None, use_bias=False)
logits = tf.squeeze(logits)
loss = tf.math.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    labels=y, logits=logits) + tf.losses.get_regularization_loss(), name='celoss')
probs = tf.math.sigmoid(logits)

with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate=conf.learning_rate)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())

with tf.name_scope('train_metric'):
    aucroc, update_aucroc_op = tf.metrics.auc(labels=y, predictions=probs)
    aucpr, update_aucpr_op = tf.metrics.auc(labels=y, predictions=probs,
                                            curve="PR")

with tf.name_scope('valid_metric'):
    val_aucroc, update_val_aucroc_op = tf.metrics.auc(
        labels=y, predictions=probs)
    val_aucpr, update_val_aucpr_op = tf.metrics.auc(labels=y, predictions=probs,
                                                    curve="PR")

# Build readers, discretizers, normalizers
train_reader = InHospitalMortalityReader(dataset_dir=os.path.join(conf.ihm_path, 'train'),
                                         listfile=os.path.join(conf.ihm_path, 'train_listfile.csv'), period_length=48.0)

discretizer = Discretizer(timestep=float(conf.timestep),
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
    normalizer_state = 'ihm_ts{}.input_str:{}.start_time:zero.normalizer'.format(
        conf.timestep, conf.imputation)
    normalizer_state = os.path.join(
        os.path.dirname(__file__), normalizer_state)
normalizer.load_params(normalizer_state)

normalizer = None
train_raw = ihm_utils.load_data(
    train_reader, discretizer, normalizer, conf.small_part, return_names=True)

print("Number of train_raw_names: ", len(train_raw['names']))

text_reader = utils.TextReader(conf.textdata_fixed, conf.starttime_path)

train_text = text_reader.read_all_text_concat_json(train_raw['names'], 48)
data = utils.merge_text_raw(train_text, train_raw)

data_X = data[0]
data_y = data[1]
data_text = data[2]
del data
del train_raw
del train_text

eval_reader = InHospitalMortalityReader(dataset_dir=os.path.join(conf.ihm_path, 'train'),
                                        listfile=os.path.join(
                                            conf.ihm_path, 'val_listfile.csv'),
                                        period_length=48.0)
eval_raw = ihm_utils.load_data(eval_reader, discretizer,
                               normalizer, conf.small_part, return_names=True)

eval_text = text_reader.read_all_text_concat_json(eval_raw['names'], 48)
data = utils.merge_text_raw(eval_text, eval_raw)

eval_data_X = data[0]
eval_data_y = data[1]
eval_data_text = data[2]
del data
del eval_raw
del eval_text

if args['mode'] == 'test':
    test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(conf.ihm_path, 'test'),
                                            listfile=os.path.join(
                                                conf.ihm_path, 'test_listfile.csv'),
                                            period_length=48.0)
    test_raw = ihm_utils.load_data(test_reader, discretizer,
                                   normalizer, conf.small_part, return_names=True)
    text_reader_test = utils.TextReader(
        conf.test_textdata_fixed, conf.test_starttime_path)
    test_text = text_reader_test.read_all_text_concat_json(
        test_raw['names'], 48)
    data = utils.merge_text_raw(test_text, test_raw)

    test_data_X = data[0]
    test_data_y = data[1]
    test_data_text = data[2]

    del data
    del test_raw
    del test_text


def generate_tensor_text(t, w2i_lookup):
    t_new = []
    max_len = -1
    for text in t:
        tokens = list(map(lambda x: lookup(w2i_lookup, x), str(text).split()))
        if conf.max_len > 0:
            tokens = tokens[:conf.max_len]
        t_new.append(tokens)
        max_len = max(max_len, len(tokens))
    pad_token = w2i_lookup['<pad>']
    for i in range(len(t_new)):
        if len(t_new[i]) < max_len:
            t_new[i] += [pad_token] * (max_len - len(t_new[i]))
    return np.array(t_new)


def generate_padded_batches(x, y, t, bs, w2i_lookup):
    batches = []
    begin = 0
    while begin < len(t):
        end = min(begin+batch_size, len(t))
        x_slice = np.stack(x[begin:end])
        y_slice = np.stack(y[begin:end])
        t_slice = generate_tensor_text(t[begin:end], w2i_lookup)
        batches.append((x_slice, y_slice, t_slice))
        begin += batch_size
    return batches


def validate(data_X_val, data_y_val, data_text_val, batch_size, word2index_lookup,
             sess, saver, last_best_val_aucpr, loss, val_aucpr, val_aucroc,
             update_val_aucpr_op, update_val_aucroc_op, save):
    val_batches = generate_padded_batches(
        data_X_val, data_y_val, data_text_val, batch_size, word2index_lookup)
    loss_list = []
    aucpr_obj = utils.AUCPR()
    #sess.run(tf.variables_initializer([v for v in tf.local_variables() if 'valid_metric' in v.name]))
    sess.run(tf.local_variables_initializer())
    for v_batch in val_batches:
        fd = {X: v_batch[0], y: v_batch[1],
              text: v_batch[2], dropout_keep_prob: 1,
              T: v_batch[2].shape[1],
              N: v_batch[2].shape[0]}
        loss_value, _, _, probablities = sess.run(
            [loss, update_val_aucpr_op, update_val_aucroc_op, probs], fd)
        loss_list.append(loss_value)
        aucpr_obj.add(probablities, v_batch[1])

    final_aucpr = sess.run(val_aucpr)
    final_aucroc = sess.run(val_aucroc)

    tf.logging.info("Validation Loss: %f - AUCPR: %f - AUCPR-SKLEARN: %f - AUCROC: %f" %
                    (np.mean(loss_list), final_aucpr, aucpr_obj.get(), final_aucroc))

    # aucpr_obj.save()
    changed = False
    if final_aucpr > last_best_val_aucpr:
        changed = True
        if save:
            save_path = saver.save(sess, args['checkpoint_path'])
            tf.logging.info(
                "Best Model saved in path: %s" % save_path)
    return max(last_best_val_aucpr, final_aucpr), changed


init = tf.group(tf.global_variables_initializer(),
                tf.local_variables_initializer())
print(len(tf.local_variables()))
saver = tf.train.Saver()
last_best_val_aucpr = -1


gpu_config = tf.ConfigProto(device_count={'GPU': 1})

with tf.Session(config=gpu_config) as sess:
    sess.run(init)

    if bool(int(args['load_model'])):
        saver.restore(sess, args['checkpoint_path'])
        last_best_val_aucpr, _ = validate(eval_data_X, eval_data_y, eval_data_text,
                                          batch_size, word2index_lookup, sess, saver, last_best_val_aucpr,
                                          loss, val_aucpr, val_aucroc, update_val_aucpr_op, update_val_aucroc_op, False)

    if args['mode'] == 'eval':
        assert bool(int(args['load_model']))
        tf.logging.info('Just Evaluating Mode.')
        last_best_val_aucpr, _ = validate(eval_data_X, eval_data_y, eval_data_text,
                                          batch_size, word2index_lookup, sess, saver, last_best_val_aucpr,
                                          loss, val_aucpr, val_aucroc, update_val_aucpr_op, update_val_aucroc_op, False)
        sys.exit(0)

    if args['mode'] == 'test':
        assert bool(int(args['load_model']))
        tf.logging.info('Testing Mode.')
        last_best_val_aucpr, _ = validate(test_data_X, test_data_y, test_data_text,
                                          batch_size, word2index_lookup, sess, saver, last_best_val_aucpr,
                                          loss, val_aucpr, val_aucroc, update_val_aucpr_op, update_val_aucroc_op, False)
        sys.exit(0)

    early_stopping = 0
    for epoch in range(number_epoch):
        tf.logging.info("Started training for epoch: %d" % epoch)
        data = list(zip(data_X, data_y, data_text))
        random.shuffle(data)
        data_X, data_y, data_text = zip(*data)

        loss_list = []

        del data

        batches = generate_padded_batches(
            data_X, data_y, data_text, batch_size, word2index_lookup)

        tf.logging.info("Generated batches for the epoch!")

        for ind, batch in enumerate(batches):
            # train the batch and update parameters.
            fd = {X: batch[0], y: batch[1],
                  text: batch[2], dropout_keep_prob: conf.dropout,
                  T: batch[2].shape[1],
                  N: batch[2].shape[0]}
            _, loss_value, aucpr_value, aucroc_value = sess.run(
                [train_op, loss, update_aucpr_op, update_aucroc_op], fd)
            loss_list.append(loss_value)

            # if ind % 200 == 0:
            # tf.logging.info(
            #   "Processed Batch: %d for Epoch: %d" % (ind, epoch))

        current_aucroc = sess.run(aucroc)
        current_aucpr = sess.run(aucpr)

        tf.logging.info("Loss: %f - AUCPR: %f - AUCROC: %f" %
                        (np.mean(loss_list), current_aucpr, current_aucroc))

        # reset aucroc and aucpr local variables
        # sess.run(tf.variables_initializer(
        #    [v for v in tf.local_variables() if 'train_metric' in v.name]))
        sess.run(tf.local_variables_initializer())
        loss_list = []

        del batches
        tf.logging.info("Started Evaluation After Epoch : %d" % epoch)
        last_best_val_aucpr, changed = validate(eval_data_X, eval_data_y, eval_data_text,
                                                batch_size, word2index_lookup, sess, saver, last_best_val_aucpr,
                                                loss, val_aucpr, val_aucroc, update_val_aucpr_op, update_val_aucroc_op, True)
        if changed == False:
            early_stopping += 1
            tf.logging.info("Didn't improve!: " + str(early_stopping))
        else:
            early_stopping = 0

        if early_stopping >= 15:
            tf.logging.info(
                "AUCPR didn't change from last 15 epochs, early stopping")
            break
        tf.logging.info("*End of Epoch.*\n")
