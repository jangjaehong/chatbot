import tensorflow as tf
import numpy as np

class seq2seq(object):

    def __init__(self, multi, hidden_size, num_layers, learning_rate, batch_size,
                 vocab_size, encoder_size, decoder_size, forward_only):

        # variables
        self.source_vocab_size = vocab_size
        self.target_vocab_size = vocab_size
        self.batch_size = batch_size
        self.encoder_size = encoder_size
        self.decoder_size = decoder_size
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.global_step = tf.Variable(0, trainable=False)

        # networks
        W = tf.Variable(tf.random_normal([hidden_size, vocab_size]))
        b = tf.Variable(tf.random_normal([vocab_size]))
        output_projection = (W, b)
        self.encoder_inputs = [tf.placeholder(tf.int32, [batch_size]) for _ in range(encoder_size)]  # 인덱스만 있는 데이터 (원핫 인코딩 미시행)
        self.decoder_inputs = [tf.placeholder(tf.int32, [batch_size]) for _ in range(decoder_size)]
        self.targets = [tf.placeholder(tf.int32, [batch_size]) for _ in range(decoder_size)]
        self.target_weights = [tf.placeholder(tf.float32, [batch_size]) for _ in range(decoder_size)]

        self.saver = tf.train.Saver(tf.global_variables())

        # models
        if multi:
            single_cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_size)
            cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * num_layers)
        else:
            cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_size)
            #cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size)
        # 학습
        if not forward_only:
            self.outputs, self.states = tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                self.encoder_inputs, self.decoder_inputs, cell,
                num_encoder_symbols=vocab_size,
                num_decoder_symbols=vocab_size,
                embedding_size=hidden_size,
                output_projection=output_projection,
                feed_previous=False)

            self.logits = [tf.matmul(output, output_projection[0]) + output_projection[1] for output in self.outputs]
            loss_list = []
            for logit, target, target_weight in zip(self.logits, self.targets, self.target_weights):
                crossentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=target)
                loss_list.append(crossentropy * target_weight)
            self.cost = tf.add_n(loss_list)
            self.loss = tf.reduce_mean(self.cost)
            self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss, global_step=self.global_step)
        else: #추론
            self.outputs, self.states = tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                self.encoder_inputs, self.decoder_inputs, cell,
                num_encoder_symbols=vocab_size,
                num_decoder_symbols=vocab_size,
                embedding_size=hidden_size,
                output_projection=output_projection,
                feed_previous=True)
            self.logits = [tf.matmul(output, output_projection[0]) + output_projection[1] for output in self.outputs]

    def step(self, session, encoderinputs, decoderinputs, targets, targetweights, forward_only):
        input_feed = {}
        for l in range(len(self.encoder_inputs)):
            input_feed[self.encoder_inputs[l].name] = encoderinputs[l]
        for l in range(len(self.decoder_inputs)):
            input_feed[self.decoder_inputs[l].name] = decoderinputs[l]
            input_feed[self.targets[l].name] = targets[l]
            input_feed[self.target_weights[l].name] = targetweights[l]
        #학습
        if not forward_only:
            print("not forward_only")
            output_feed = [self.train_op, self.loss]
        #추론
        else:
            print("forward_only")
            output_feed = []
            for l in range(len(self.decoder_inputs)):
                output_feed.append(self.logits[l])

        output = session.run(output_feed, input_feed)
        # 학습
        if not forward_only:
            print("not forward_only")
            return output[1] # loss
        # 추론
        else:
            print("forward_only")
            return output[0:] # outputs

