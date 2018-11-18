# for checkpoint paths
import os
# for fancy progress bar
from tqdm import tqdm
# TensorFlow
import tensorflow as tf

from tensorflow.python.layers.core import Dense
from algorithm4.util import DataUtil


class DemoConfig:
    def __init__(self, util):
        self.util = util

        # Model
        self.hidden_size = 100
        self.embedding_size = 100
        self.cell = tf.nn.rnn_cell.BasicLSTMCell

        # Training
        self.optimizer = tf.train.AdamOptimizer
        self.n_epoch = 800
        self.learning_rate = 0.001

        self.enc_vocab = util.enc_vocab
        self.dec_vocab = util.dec_vocab

        self.enc_reverse_vocab = util.enc_reverse_vocab
        self.dec_reverse_vocab = util.dec_reverse_vocab

        self.enc_vocab_size = util.enc_vocab_size
        self.dec_vocab_size = util.dec_vocab_size

        self.enc_sentence_length = util.enc_sentence_length
        self.dec_sentence_length = util.dec_sentence_length

        # Tokens
        self.start_token = util.start_token  # start_token = 0
        self.end_token = util.end_token  # end_token = 1
        self.unk_token = util.unk_token

        # Checkpoint Path
        self.ckpt_dir = './model/'


class Seq2SeqModel(object):
    def __init__(self, config, mode='training'):
        assert mode in ['training', 'evaluation', 'inference']
        self.mode = mode

        self.util = config.util

        self.enc_vocab_size = config.enc_vocab_size
        self.dec_vocab_size = config.dec_vocab_size

        self.enc_vocab = config.enc_vocab
        self.dec_vocab = config.dec_vocab

        self.enc_reverse_vocab = config.enc_reverse_vocab
        self.dec_reverse_vocab = config.dec_reverse_vocab

        self.enc_sentence_length = config.enc_sentence_length
        self.dec_sentence_length = config.dec_sentence_length

        # Model
        self.hidden_size = config.hidden_size
        self.embedding_size = config.embedding_size
        self.cell = config.cell

        # Training
        self.optimizer = config.optimizer
        self.n_epoch = config.n_epoch
        self.learning_rate = config.learning_rate

        # Tokens
        self.start_token = config.start_token  # start_token = 0
        self.end_token = config.end_token  # end_token = 1
        self.unk_token = config.unk_token

        # Checkpoint Path
        self.ckpt_dir = config.ckpt_dir

    def add_placeholders(self):
        self.enc_inputs = tf.placeholder(
            tf.int32,
            shape=[None, self.enc_sentence_length],
            name='input_sentences')

        self.enc_sequence_length = tf.placeholder(
            tf.int32,
            shape=[None, ],
            name='input_sequence_length')

        self.batch_size = tf.shape(self.enc_inputs)[0]

        if self.mode == 'training':
            self.dec_inputs = tf.placeholder(
                tf.int32,
                shape=[None, self.dec_sentence_length + 2],
                name='target_sentences')

            self.dec_sequence_length = tf.placeholder(
                tf.int32,
                shape=[None, ],
                name='target_sequence_length')

    def add_encoder(self):
        with tf.variable_scope('Encoder') as scope:
            with tf.device('/cpu:0'):
                self.enc_embedding = tf.get_variable(
                    name='embedding',
                    initializer=tf.random_uniform([self.enc_vocab_size + 1, self.embedding_size]),
                    dtype=tf.float32)

            # [Batch_size x enc_sent_len x embedding_size]
            enc_emb_inputs = tf.nn.embedding_lookup(
                self.enc_embedding,
                self.enc_inputs,
                name='emb_inputs')
            enc_cell = self.cell(self.hidden_size)
            # enc_cell = tf.nn.rnn_cell.DropoutWrapper(enc_cell, output_keep_prob=0.7)
            # enc_outputs: [batch_size x enc_sent_len x embedding_size]
            # enc_last_state: [batch_size x embedding_size]
            self.enc_outputs, self.enc_last_state = tf.nn.dynamic_rnn(
                cell=enc_cell,
                inputs=enc_emb_inputs,
                sequence_length=self.enc_sequence_length,
                time_major=False,
                dtype=tf.float32)

    def add_decoder(self):
        with tf.variable_scope('Decoder') as scope:
            with tf.device('/cpu:0'):
                self.dec_embedding = tf.get_variable(
                    name='embedding',
                    initializer=tf.random_uniform([self.dec_vocab_size + 3, self.embedding_size]),
                    dtype=tf.float32)

            dec_cell = self.cell(self.hidden_size)
            # dec_cell = tf.nn.rnn_cell.DropoutWrapper(dec_cell, output_keep_prob=0.7)

            # output projection (replacing `OutputProjectionWrapper`)
            output_layer = Dense(self.dec_vocab_size + 3, name='output_projection')

            if self.mode == 'training':
                # maxium unrollings in current batch = max(dec_sent_len) + 1(GO symbol)
                max_dec_len = tf.reduce_max(self.dec_sequence_length + 2, name='max_dec_len')

                dec_emb_inputs = tf.nn.embedding_lookup(
                    self.dec_embedding, self.dec_inputs, name='emb_inputs')

                training_helper = tf.contrib.seq2seq.TrainingHelper(
                    inputs=dec_emb_inputs,
                    sequence_length=self.dec_sequence_length+1,
                    time_major=False,
                    name='training_helper')

                training_decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell=dec_cell,
                    helper=training_helper,
                    initial_state=self.enc_last_state,
                    output_layer=output_layer)

                train_dec_outputs, train_dec_last_state, train_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
                    decoder=training_decoder,
                    output_time_major=False,
                    impute_finished=True,
                    maximum_iterations=max_dec_len)

                # dec_outputs: collections.namedtuple(rnn_outputs, sample_id)
                # dec_outputs.rnn_output: [batch_size x max(dec_sequence_len) x dec_vocab_size+3], tf.float32
                # dec_outputs.sample_id [batch_size], tf.int32

                # logits: [batch_size x max_dec_len x dec_vocab_size+3]
                logits = tf.identity(train_dec_outputs.rnn_output, name='logits')

                # prediction sample for validation
                # List of training variables
                # self.training_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
                self.valid_predictions = tf.identity(train_dec_outputs.sample_id, name='valid_preds')

                # masks: [batch_size x max_dec_len]
                # => ignore outputs after `dec_senquence_length+2` when calculating loss
                masks = tf.sequence_mask(self.dec_sequence_length + 1, max_dec_len, dtype=tf.float32, name='masks')

                targets = tf.slice(self.dec_inputs, [0, 0], [-1, max_dec_len], 'targets')

                # Control loss dimensions with `average_across_timesteps` and `average_across_batch`
                # internal: `tf.nn.sparse_softmax_cross_entropy_with_logits`
                self.batch_loss = tf.contrib.seq2seq.sequence_loss(
                    logits=logits,
                    targets=targets,
                    weights=masks,
                    average_across_timesteps=True,
                    average_across_batch=True,
                    name='batch_loss')

            elif self.mode == 'inference':
                start_tokens = tf.ones([self.batch_size], dtype=tf.int32) * self.start_token
                end_token = self.end_token

                inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                    embedding=self.dec_embedding,
                    start_tokens=start_tokens,
                    end_token=end_token)

                inference_decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell=dec_cell,
                    helper=inference_helper,
                    initial_state=self.enc_last_state,
                    output_layer=output_layer)

                infer_dec_outputs, infer_dec_last_state, _ = tf.contrib.seq2seq.dynamic_decode(
                    inference_decoder,
                    output_time_major=False,
                    impute_finished=True,
                    maximum_iterations=self.dec_sentence_length + 1)

                # [batch_size x dec_sentence_length], tf.int32
                self.predictions = tf.identity(infer_dec_outputs.sample_id, name='predictions')
                # equivalent to tf.argmax(infer_dec_outputs.rnn_output, axis=2, name='predictions')

                # List of training variables
                # self.training_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

    def add_training_op(self):
        self.training_op = self.optimizer(self.learning_rate, name='training_op').minimize(self.batch_loss)

    def save(self, sess, var_list=None, save_path=None):
        print(f'Saving model at {save_path}')
        if hasattr(self, 'training_variables'):
            var_list = self.training_variables
        saver = tf.train.Saver(var_list)
        saver.save(sess, save_path, write_meta_graph=False)

    def restore(self, sess, var_list=None, ckpt_path=None):
        if hasattr(self, 'training_variables'):
            var_list = self.training_variables
        self.restorer = tf.train.Saver(var_list)
        self.restorer.restore(sess, ckpt_path)
        print('Restore Finished!')

    def summary(self):
        summary_writer = tf.summary.FileWriter(
            logdir=self.ckpt_dir,
            graph=tf.get_default_graph())

    def build(self):
        self.add_placeholders()
        self.add_encoder()
        self.add_decoder()

    def train(self, sess, data, from_scratch=False, load_ckpt=None, save_path=None):
        # Restore Checkpoint
        if from_scratch is False and os.path.isfile(load_ckpt):
            self.restore(sess, load_ckpt)

        # Add Optimizer to current graph
        self.add_training_op()

        sess.run(tf.global_variables_initializer())

        input_batches, target_batches = data
        loss_history = []

        for epoch in tqdm(range(self.n_epoch)):
            all_preds = []
            epoch_loss = 0
            for input_batch, target_batch in zip(input_batches, target_batches):
                input_batch_tokens = []
                target_batch_tokens = []
                enc_sentence_lengths = []
                dec_sentence_lengths = []

                for input_sent in input_batch:
                    tokens, sent_len = self.util.sent2idx(
                        sent=input_sent,
                        vocab=self.enc_vocab,
                        max_sentence_length=self.enc_sentence_length)
                    input_batch_tokens.append(tokens)
                    enc_sentence_lengths.append(sent_len)

                for target_sent in target_batch:
                    tokens, sent_len = self.util.sent2idx(
                        sent=target_sent,
                        vocab=self.dec_vocab,
                        max_sentence_length=self.dec_sentence_length,
                        is_target=True)
                    target_batch_tokens.append(tokens)
                    dec_sentence_lengths.append(sent_len)
                # Evaluate 3 ops in the graph
                # => valid_predictions, loss, training_op(optimzier)
                batch_preds, batch_loss, _ = sess.run(
                    [self.valid_predictions, self.batch_loss, self.training_op],
                    feed_dict={
                        self.enc_inputs: input_batch_tokens,
                        self.enc_sequence_length: enc_sentence_lengths,
                        self.dec_inputs: target_batch_tokens,
                        self.dec_sequence_length: dec_sentence_lengths,
                    })
                # loss_history.append(batch_loss)
                epoch_loss += batch_loss
                all_preds.append(batch_preds)

            loss_history.append(epoch_loss)

            # Logging every 400 epochs
            if epoch % 100 == 0:
                print('Epoch', epoch)
                for input_batch, target_batch, batch_preds in zip(input_batches, target_batches, all_preds):
                    for input_sent, target_sent, pred in zip(input_batch, target_batch, batch_preds):
                        print(f'\tInput: {input_sent}')
                        print(f'\tPrediction:', self.util.idx2sent(pred, reverse_vocab=self.dec_reverse_vocab))
                        print(f'\tTarget:, {target_sent}')
                print(f'\tepoch loss: {epoch_loss:.2f}\n')

        if save_path:
            self.save(sess, save_path=save_path)

        return loss_history

    def inference(self, sess, data, load_ckpt):

        self.restore(sess, ckpt_path=load_ckpt)

        input_batch, target_batch = data

        batch_preds = []
        batch_tokens = []
        batch_sent_lens = []

        for input_sent in input_batch:
            tokens, sent_len = self.util.sent2idx(
                sent=input_sent,
                vocab=self.enc_vocab,
                max_sentence_length=self.enc_sentence_length)
            batch_tokens.append(tokens)
            batch_sent_lens.append(sent_len)

        batch_preds = sess.run(
            self.predictions,
            feed_dict={
                self.enc_inputs: batch_tokens,
                self.enc_sequence_length: batch_sent_lens,
            })

        for input_sent, target_sent, pred in zip(input_batch, target_batch, batch_preds):
            print('Input:', input_sent)
            print('Prediction:', self.util.idx2sent(pred, reverse_vocab=self.dec_reverse_vocab))
            print('Target:', target_sent, '\n')

