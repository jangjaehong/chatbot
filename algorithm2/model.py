import matplotlib.pyplot as plt
# for pretty print
from pprint import pprint
# for tokenizer
import re
# for word counter in vocabulary dictionary
from collections import Counter
# TensorFlow of Course :)
import tensorflow as tf

# The paths of RNNCell or rnn functions are too long.
from tensorflow.contrib.legacy_seq2seq.python.ops import *
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import *
from tensorflow.contrib.legacy_seq2seq.python.ops.seq2seq import *
from tqdm import tqdm

from algorithm2.util import DataUtil


class Config:
    datautil = DataUtil()
    input_batches = datautil.input_batches
    target_batches = datautil.target_batches

    enc_vocab = datautil.enc_vocab
    dec_vocab = datautil.dec_vocab

    enc_reverse_vocab = datautil.enc_reverse_vocab
    dec_reverse_vocab = datautil.dec_reverse_vocab

    enc_vocab_size = datautil.enc_vocab_size
    dec_vocab_size = datautil.dec_vocab_size

    enc_sentence_length = datautil.enc_sentence_length
    dec_sentence_length = datautil.dec_sentence_length

    batch_size = datautil.batch_size
    embedding_size = enc_emb_size = dec_emb_size = 100
    hidden_size = 100
    n_epoch = 2000
    learning_rate = 0.0001


class Seq2Seq:
    config = Config()
    tf.reset_default_graph()

    enc_inputs = tf.placeholder(
        tf.int32,
        shape=[None, config.enc_sentence_length],
        name='input_sentences')

    sequence_lengths = tf.placeholder(
        tf.int32,
        shape=[None],
        name='sentences_length')

    dec_inputs = tf.placeholder(
        tf.int32,
        shape=[None, config.dec_sentence_length + 2],
        name='output_sentences')

    # batch_major => time_major
    enc_inputs_t = tf.transpose(enc_inputs, [1, 0])
    dec_inputs_t = tf.transpose(dec_inputs, [1, 0])

    cell = tf.nn.rnn_cell.BasicRNNCell(config.hidden_size)

    with tf.variable_scope("embedding_attention_seq2seq"):
        # dec_outputs: [dec_sent_len+1 x batch_size x hidden_size]
        dec_outputs, dec_last_state = embedding_attention_seq2seq(
            encoder_inputs=tf.unstack(enc_inputs_t),
            decoder_inputs=tf.unstack(dec_inputs_t),
            cell=cell,
            num_encoder_symbols=config.enc_vocab_size + 1,
            num_decoder_symbols=config.dec_vocab_size + 3,
            embedding_size=config.embedding_size,
            feed_previous=True)

    # predictions: [batch_size x dec_sentence_lengths+1]
    predictions = tf.transpose(tf.argmax(tf.stack(dec_outputs), axis=-1), [1, 0])

    # labels & logits: [dec_sentence_length+1 x batch_size x dec_vocab_size+2]
    labels = tf.one_hot(dec_inputs_t, config.dec_vocab_size + 3)
    logits = tf.stack(dec_outputs)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=labels, logits=logits))

    # training_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)
    training_op = tf.train.RMSPropOptimizer(learning_rate=config.learning_rate).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        loss_history = []
        for epoch in tqdm(range(config.n_epoch)):
            all_preds = []
            epoch_loss = 0
            for input_batch, target_batch in zip(config.input_batches, config.target_batches):
                input_token_indices = []
                target_token_indices = []
                sentence_lengths = []

                for input_sent in input_batch:
                    input_sent, sent_len = config.datautil.sent2idx(input_sent,
                                                                    vocab=config.enc_vocab,
                                                                    max_sentence_length=config.enc_sentence_length)
                    input_token_indices.append(input_sent)
                    sentence_lengths.append(sent_len)

                for target_sent in target_batch:
                    target_token_indices.append(
                        config.datautil.sent2idx(target_sent,
                                                 vocab=config.dec_vocab,
                                                 max_sentence_length=config.dec_sentence_length,
                                                 is_target=True))

                # Evaluate three operations in the graph
                # => predictions, loss, training_op(optimzier)
                batch_preds, batch_loss, _ = sess.run(
                    [predictions, loss, training_op],
                    feed_dict={
                        enc_inputs: input_token_indices,
                        sequence_lengths: sentence_lengths,
                        dec_inputs: target_token_indices
                    })
                loss_history.append(batch_loss)
                epoch_loss += batch_loss
                all_preds.append(batch_preds)

            # Logging every 400 epochs
            if epoch % 50 == 0:
                print('Train Epoch =========', epoch)
                for input_batch, target_batch, batch_preds in zip(config.input_batches, config.target_batches, all_preds):
                    for input_sent, target_sent, pred in zip(input_batch, target_batch, batch_preds):
                        print('\t', input_sent)
                        print('\t => ', config.datautil.idx2sent(pred, reverse_vocab=config.dec_reverse_vocab))
                        print('\tCorrent answer:', target_sent)
                print('\tepoch loss: {:.2f}\n'.format(epoch_loss))

        plt.figure(figsize=(20, 10))
        plt.scatter(range(config.n_epoch * config.batch_size), loss_history)
        plt.title('Learning Curve')
        plt.xlabel('Global step')
        plt.ylabel('Loss')
        plt.show()
