import numpy as np
import algorithm3.dialog as dialog

import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq

from tensorflow.python.ops.rnn_cell import GRUCell
from tensorflow.python.ops.rnn_cell import LSTMCell
from tensorflow.python.ops.rnn_cell import MultiRNNCell
from tensorflow.python.ops.rnn_cell import DropoutWrapper, ResidualWrapper

from tensorflow.python.ops import array_ops
from tensorflow.python.layers.core import Dense
from tensorflow.python.util import nest

from tensorflow.contrib.seq2seq.python.ops import attention_wrapper
from tensorflow.contrib.seq2seq.python.ops import beam_search_decoder


class Seq2Seq(object):

    def __init__(self, FLAGS):
        self.train_mode = FLAGS.train_mode
        self.decode_mode = FLAGS.decode_mode
        self.cell_type = FLAGS.cell_type
        self.hidden_dim = FLAGS.hidden_dim
        self.num_layer = FLAGS.num_layer
        self.attention_type = FLAGS.attention_type
        self.embedding_dim = FLAGS.embedding_dim
        self.src_vocab_size = FLAGS.src_vocab_size
        self.tgt_vocab_size = FLAGS.tgt_vocab_size
        self.use_residual = False
        self.attn_input_feeding = True
        self.use_dropout = True
        self.keep_prob = 0.7
        self.learning_rate = 0.001  # or 0.0001
        self.state_tuple_mode = True
        self.init_state_flag = 0
        self.max_gradient_norm = 1.0
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.global_epoch_step = tf.Variable(0, trainable=False, name='global_epoch_step')
        self.global_epoch_step_op = tf.assign(self.global_epoch_step, self.global_epoch_step + 1)
        self.keep_prob_placeholder = tf.placeholder(tf.float32, shape=[], name='keep_prob')
        self.dtype = tf.float32
        self.beamsearch_decode = FLAGS.beamsearch_decode
        if self.decode_mode:
            self.beam_width = FLAGS.beam_width
            self.beamsearch_decode = True if self.beam_width > 1 else False
            self.max_decode_step = FLAGS.max_decode_step
        self.saver = tf.train.Saver(tf.global_variables())
        self.build_model()

    def build_model(self):
        print("모델 생성중...")

        # Building encoder and decoder networks
        self.init_placeholders()
        self.build_encoder()
        self.build_decoder()

        # Merge all the training summaries
        self.summary_op = tf.summary.merge_all()

    def init_placeholders(self):
        # encoder_inputs: [batch_size, max_time_steps]
        self.encoder_inputs = tf.placeholder(
            dtype=tf.int32,
            shape=[None, None],
            name='encoder_inputs')

        # encoder_inputs_length: [batch_size]
        self.encoder_inputs_length = tf.placeholder(
            dtype=tf.int32,
            shape=[None, ],
            name='encoder_inputs_length')

        # get dynamic batch_size
        self.batch_size = tf.shape(self.encoder_inputs)[0]

        if self.train_mode:
            self.decoder_inputs = tf.placeholder(
                dtype=tf.int32,
                shape=[None, None],
                name='decoder_inputs')

            self.decoder_inputs_length = tf.placeholder(
                dtype=tf.int32,
                shape=[None, ],
                name='decoder_inputs_length')

            decoder_start_token = tf.ones(
                shape=[self.batch_size, 1],
                dtype=tf.int32) * dialog.start_token

            decoder_end_token = tf.ones(
                shape=[self.batch_size, 1],
                dtype=tf.int32) * dialog.pad_token

            # decoder_inputs_train: [batch_size , max_time_steps + 1]
            # insert _GO symbol in front of each decoder input
            self.decoder_inputs_train = tf.concat([decoder_start_token,
                                                   self.decoder_inputs], axis=1)

            # decoder_inputs_length_train: [batch_size]
            self.decoder_inputs_length_train = self.decoder_inputs_length + 1

            # decoder_targets_train: [batch_size, max_time_steps + 1]
            # insert EOS symbol at the end of each decoder input
            self.decoder_targets_train = tf.concat([self.decoder_inputs,
                                                    decoder_end_token], axis=1)

    def build_single_cell(self):
        cell_type = LSTMCell
        if self.cell_type == 'gru':
            cell_type = GRUCell
        cell = cell_type(self.hidden_dim)

        if self.use_dropout:
            cell = DropoutWrapper(cell, dtype=self.dtype, output_keep_prob=self.keep_prob_placeholder, )
        if self.use_residual:
            cell = ResidualWrapper(cell)

        return cell

    def build_encoder_cell(self):
        return MultiRNNCell([self.build_single_cell() for _ in range(self.num_layer)])

    def build_decoder_cell(self):
        encoder_outputs = self.encoder_outputs
        encoder_last_state = self.encoder_state
        encoder_inputs_length = self.encoder_inputs_length

        # To use BeamSearchDecoder, encoder_outputs, encoder_last_state, encoder_inputs_length
        # needs to be tiled so that: [batch_size, .., ..] -> [batch_size x beam_width, .., ..]
        if self.beamsearch_decode:
            print("use beamsearch decoding..")
            encoder_outputs = seq2seq.tile_batch(
                self.encoder_outputs, multiplier=self.beam_width)
            encoder_last_state = nest.map_structure(
                lambda s: seq2seq.tile_batch(s, self.beam_width), self.encoder_state)
            encoder_inputs_length = seq2seq.tile_batch(
                self.encoder_inputs_length, multiplier=self.beam_width)

        # Building attention mechanism: Default Bahdanau
        # 'Bahdanau' style attention: https://arxiv.org/abs/1409.0473
        self.attention_mechanism = attention_wrapper.BahdanauAttention(
            num_units=self.hidden_dim, memory=encoder_outputs,
            memory_sequence_length=encoder_inputs_length, )
        # 'Luong' style attention: https://arxiv.org/abs/1508.04025
        if self.attention_type.lower() == 'luong':
            self.attention_mechanism = attention_wrapper.LuongAttention(
                num_units=self.hidden_dim, memory=encoder_outputs,
                memory_sequence_length=encoder_inputs_length, )

        # Building decoder_cell
        self.decoder_cell_list = [
            self.build_single_cell() for _ in range(self.num_layer)]
        decoder_initial_state = encoder_last_state

        def attn_decoder_input_fn(inputs, attention):
            if not self.attn_input_feeding:
                return inputs

            # Essential when use_residual=True
            _input_layer = Dense(self.hidden_dim, dtype=self.dtype,
                                 name='attn_input_feeding')
            return _input_layer(array_ops.concat([inputs, attention], -1))

        # AttentionWrapper wraps RNNCell with the attention_mechanism
        # Note: We implement Attention mechanism only on the top decoder layer
        self.decoder_cell_list[-1] = attention_wrapper.AttentionWrapper(
            cell=self.decoder_cell_list[-1],
            attention_mechanism=self.attention_mechanism,
            attention_layer_size=self.hidden_dim,
            cell_input_fn=attn_decoder_input_fn,
            initial_cell_state=encoder_last_state[-1],
            alignment_history=False,
            name='Attention_Wrapper')

        # To be compatible with AttentionWrapper, the encoder last state
        # of the top layer should be converted into the AttentionWrapperState form
        # We can easily do this by calling AttentionWrapper.zero_state

        # Also if beamsearch decoding is used, the batch_size argument in .zero_state
        # should be ${decoder_beam_width} times to the origianl batch_size
        batch_size = self.batch_size if not self.beamsearch_decode \
            else self.batch_size * self.beam_width
        initial_state = [state for state in encoder_last_state]

        initial_state[-1] = self.decoder_cell_list[-1].zero_state(
            batch_size=batch_size, dtype=self.dtype)
        decoder_initial_state = tuple(initial_state)

        return MultiRNNCell(self.decoder_cell_list), decoder_initial_state

    def build_encoder(self):
        print("build encoder...")
        with tf.variable_scope('encoder'):
            with tf.device('/cpu:0'):
                self.encoder_cell = self.build_encoder_cell()

                self.embedding_encoder = tf.get_variable(name="embedding",
                                                         initializer=[self.src_vocab_size + 1, self.embedding_dim],
                                                         dtype=self.dtype)

                self.encoder_embedding_input = tf.nn.embedding_lookup(self.embedding_encoder, self.encoder_inputs)

                input_layer = tf.layers.Dense(units=self.hidden_dim, dtype=self.dtype)

                self.encoder_embedding_input = input_layer(self.encoder_embedding_input)

                self.encoder_outputs, self.encoder_state = tf.nn.dynamic_rnn(
                    cell=self.encoder_cell,
                    inputs=self.encoder_embedding_input,
                    sequence_length=self.encoder_inputs_length,
                    time_major=False,
                    dtype=self.dtype)

    def build_decoder(self):
        print("build decoder...")
        with tf.variable_scope('decoder'):
            with tf.device('/cpu:0'):
                self.embedding_decoder = tf.get_variable("embedding",
                                                         initializer=[self.tgt_vocab_size + 2, self.embedding_dim],
                                                         dtype=self.dtype)

            self.decoder_cell, self.decoder_state = self.build_decoder_cell()

            input_layer = tf.layers.Dense(units=self.hidden_dim, dtype=self.dtype)
            output_layer = tf.layers.Dense(units=self.tgt_vocab_size, use_bias=False)

            if self.train_mode:
                print("setting_train")

                max_decoder_length = tf.reduce_max(self.decoder_inputs_length_train + 1)

                self.decoder_embedding_input = tf.nn.embedding_lookup(
                    params=self.embedding_decoder,
                    ids=self.decoder_inputs_train)

                self.decoder_embedding_input = input_layer(self.decoder_embedding_input)

                training_helper = tf.contrib.seq2seq.TrainingHelper(
                    inputs=self.decoder_embedding_input,
                    sequence_length=self.decoder_inputs_length_train + 1,
                    time_major=False)

                training_decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell=self.decoder_cell,
                    helper=training_helper,
                    initial_state=self.decoder_state,
                    output_layer=output_layer)

                self.decoder_outputs_train, self.decoder_last_state_train, self.decoder_outputs_length_train = seq2seq.dynamic_decode(
                    decoder=training_decoder,
                    output_time_major=False,
                    impute_finished=True,
                    maximum_iterations=max_decoder_length)

                self.decoder_logits_train = tf.identity(self.decoder_outputs_train.rnn_output)

                self.decoder_pred_train = tf.argmax(self.decoder_logits_train, axis=-1)

                masks = tf.sequence_mask(lengths=self.decoder_inputs_length_train + 1,
                                         maxlen=max_decoder_length,
                                         dtype=self.dtype)

                self.loss = seq2seq.sequence_loss(
                    logits=self.decoder_logits_train,
                    targets=self.decoder_targets_train,
                    weights=masks,
                    average_across_timesteps=True,
                    average_across_batch=True, )
                # Training summary for the current batch_loss
                tf.summary.scalar('loss', self.loss)

                self.init_optimizer()

            elif self.decode_mode:
                print("build decode")
                # Start_tokens: [batch_size,] `int32` vector
                start_tokens = tf.ones([self.batch_size, ], tf.int32) * dialog.start_token
                end_token = dialog.pad_token

                def embed_and_input_proj(inputs):
                    return input_layer(tf.nn.embedding_lookup(self.embedding_decoder, inputs))

                if not self.beamsearch_decode:
                    decoding_helper = seq2seq.GreedyEmbeddingHelper(start_tokens=start_tokens,
                                                                    end_token=end_token,
                                                                    embedding=embed_and_input_proj)
                    print("building greedy decoder..")
                    inference_decoder = seq2seq.BasicDecoder(cell=self.decoder_cell,
                                                             helper=decoding_helper,
                                                             initial_state=self.decoder_state,
                                                             output_layer=output_layer)
                else:
                    print("building beamsearch decoder..")
                    inference_decoder = beam_search_decoder.BeamSearchDecoder(cell=self.decoder_cell,
                                                                              embedding=embed_and_input_proj,
                                                                              start_tokens=start_tokens,
                                                                              end_token=end_token,
                                                                              initial_state=self.decoder_state,
                                                                              beam_width=self.beam_width,
                                                                              output_layer=output_layer, )

                self.decoder_outputs_decode, self.decoder_last_state_decode, self.decoder_outputs_length_decode = seq2seq.dynamic_decode(
                    decoder=inference_decoder,
                    output_time_major=False,
                    #impute_finished=True,
                    maximum_iterations=self.max_decode_step)

                if not self.beamsearch_decode:
                    self.decoder_pred_decode = tf.expand_dims(self.decoder_outputs_decode.sample_id, -1)
                else:
                    self.decoder_pred_decode = self.decoder_outputs_decode.predicted_ids

    def init_optimizer(self):
        print("setting optimizer..")
        trainable_params = tf.trainable_variables()
        self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        gradients = tf.gradients(self.loss, trainable_params)
        clip_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
        self.updates = self.opt.apply_gradients(zip(clip_gradients, trainable_params), global_step=self.global_step)

    def save(self, sess, path, var_list=None, global_step=None):
        # var_list = None returns the list of all saveable variables
        saver = tf.train.Saver(var_list)

        # temporary code
        # del tf.get_collection_ref('LAYER_NAME_UIDS')[0]
        save_path = saver.save(sess, save_path=path, global_step=global_step)
        print('model saved at %s' % save_path)

    def restore(self, sess, path, var_list=None):
        # var_list = None returns the list of all saveable variables
        saver = tf.train.Saver(var_list)
        saver.restore(sess, save_path=path)
        print('model restored from %s' % path)

    def train(self, sess, encoder_inputs, encoder_inputs_length, decoder_inputs, decoder_inputs_length):
        if not self.train_mode:
            raise ValueError("train() 함수는 train 모드에서만 실행가능합니다.")

        input_feed = self.check_feeds(encoder_inputs, encoder_inputs_length,
                                      decoder_inputs, decoder_inputs_length, False)
        # Input feeds for dropout
        input_feed[self.keep_prob_placeholder.name] = self.keep_prob

        output_feed = [self.updates,  # Update Op that does optimization
                       self.loss,  # Loss for current batch
                       self.summary_op]  # Training summary

        outputs = sess.run(output_feed, input_feed)
        return outputs[1], outputs[2]  # loss, summary

    def eval(self, sess, encoder_inputs, encoder_inputs_length, decoder_inputs, decoder_inputs_length):
        """아래 매겨변수를 통해 모델 평가 진행.
        Args:
          session: tensorflow session to use.
          encoder_inputs: 정수형 numpy 매트릭스 [batch_size, max_source_time_steps]
          encoder_inputs_length: numpy 1D tensor [batch_size]
          decoder_inputs: 정수형 numpy 매트릭스 [batch_size, max_target_time_steps]
          decoder_inputs_length: numpy 1D tensor [batch_size]
        Returns:
          A triple consisting of gradient norm (or None if we did not do backward),
          average perplexity, and the outputs.
        """
        input_feed = self.check_feeds(encoder_inputs, encoder_inputs_length,
                                      decoder_inputs, decoder_inputs_length, False)
        # Input feeds for dropout
        input_feed[self.keep_prob_placeholder.name] = 1.0

        output_feed = [self.loss,  # Loss for current batch
                       self.summary_op]  # Training summary
        outputs = sess.run(output_feed, input_feed)
        return outputs[0], outputs[1]  # loss

    def predict(self, sess, encoder_inputs, encoder_inputs_length):
        input_feed = self.check_feeds(encoder_inputs, encoder_inputs_length,
                                      decoder_inputs=None, decoder_inputs_length=None,
                                      decode=True)

        # Input feeds for dropout
        input_feed[self.keep_prob_placeholder.name] = 1.0

        output_feed = [self.decoder_pred_decode]
        outputs = sess.run(output_feed, input_feed)

        # GreedyDecoder: [batch_size, max_time_step]
        return outputs[0]  # BeamSearchDecoder: [batch_size, max_time_step, beam_width]

    def check_feeds(self, encoder_inputs, encoder_inputs_length, decoder_inputs, decoder_inputs_length, decode):
        """
        Args:
          encoder_inputs: a numpy int matrix of [batch_size, max_source_time_steps]
              to feed as encoder inputs
          encoder_inputs_length: a numpy int vector of [batch_size]
              to feed as sequence lengths for each element in the given batch
          decoder_inputs: a numpy int matrix of [batch_size, max_target_time_steps]
              to feed as decoder inputs
          decoder_inputs_length: a numpy int vector of [batch_size]
              to feed as sequence lengths for each element in the given batch
          decode: a scalar boolean that indicates decode mode
        Returns:
          A feed for the model that consists of encoder_inputs, encoder_inputs_length,
          decoder_inputs, decoder_inputs_length
        """

        input_batch_size = encoder_inputs.shape[0]
        if input_batch_size != encoder_inputs_length.shape[0]:
            raise ValueError("Encoder inputs and their lengths must be equal in their "
                             "batch_size, %d != %d" % (input_batch_size, encoder_inputs_length.shape[0]))

        if not decode:
            target_batch_size = decoder_inputs.shape[0]
            if target_batch_size != input_batch_size:
                raise ValueError("Encoder inputs and Decoder inputs must be equal in their "
                                 "batch_size, %d != %d" % (input_batch_size, target_batch_size))
            if target_batch_size != decoder_inputs_length.shape[0]:
                raise ValueError("Decoder targets and their lengths must be equal in their "
                                 "batch_size, %d != %d" % (target_batch_size, decoder_inputs_length.shape[0]))

        input_feed = {}

        input_feed[self.encoder_inputs.name] = encoder_inputs
        input_feed[self.encoder_inputs_length.name] = encoder_inputs_length

        if not decode:
            input_feed[self.decoder_inputs.name] = decoder_inputs
            input_feed[self.decoder_inputs_length.name] = decoder_inputs_length

        return input_feed