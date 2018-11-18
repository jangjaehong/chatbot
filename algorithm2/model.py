import matplotlib.pyplot as plt
import tensorflow as tf
import os

from tensorflow.python.layers.core import Dense
from algorithm2.util import DataUtil
from tqdm import tqdm



class Config:
    datautil = DataUtil()
    input_batches = datautil.input_batches
    target_batches = datautil.target_batches

    enc_vocab = datautil.enc_vocab
    dec_vocab = target_vocab = datautil.dec_vocab

    enc_reverse_vocab = datautil.enc_reverse_vocab
    dec_reverse_vocab = target_reverse_vocab = datautil.dec_reverse_vocab

    enc_vocab_size = datautil.enc_vocab_size
    dec_vocab_size = target_vocab_size = datautil.dec_vocab_size

    enc_sentence_length = datautil.enc_sentence_length
    dec_sentence_length = target_sentence_length = datautil.dec_sentence_length

    embedding_size = enc_emb_size = dec_emb_size = 100
    hidden_size = 100
    attn_size = 100
    n_epoch = 2000
    learning_rate = 0.0001

    start_token = datautil.start_token
    end_token = datautil.end_token

    ckpt_dir = './model/'


class Seq2Seq:
    def __init__(self, mode):
        config = Config()
        self.util = config.datautil

        self.mode = mode
        self.cell_type = "LSTM"
        self.cell_multi = True

        self.enc_vocab = config.enc_vocab
        self.enc_reverse_vocab = config.enc_reverse_vocab
        self.enc_vocab_size = config.enc_vocab_size
        self.enc_sentence_length = config.enc_sentence_length

        self.dec_vocab = config.dec_vocab
        self.dec_reverse_vocab = config.dec_reverse_vocab
        self.dec_vocab_size = config.dec_vocab_size
        self.dec_sentence_length = config.dec_sentence_length

        self.target_vocab = config.target_vocab
        self.target_reverse_vocab = config.target_reverse_vocab
        self.target_vocab_size = config.target_vocab_size
        self.target_sentence_length = config.target_sentence_length

        # RNN Cell의 출력
        self.enc_output_dim = self.enc_vocab_size
        self.dec_output_dim = self.dec_vocab_size
        # RNN Cell hidden layer size
        self.hidden_size = 100
        # Cell depth
        self.num_layer = 3
        # embedding vector
        self.embedding_size = self.enc_emb_size = self.dec_emb_size = 100
        self.attn_size= config.attn_size
        # train
        self.n_epoch = 2000
        self.learning_rate = 0.0001
        #token
        self.start_token = config.start_token
        self.end_token = config.end_token

        # Checkpoint Path
        self.ckpt_dir = config.ckpt_dir
        self.state_tuple_mode = False

    def _init_placeholder(self):
        self.enc_inputs = tf.placeholder(
            tf.int32,
            shape=[None, self.enc_sentence_length],
            name='input_sentences')

        self.enc_inputs_length = tf.placeholder(
            tf.int32,
            shape=[None,],
            name='sentences_length')

        self.batch_size = tf.shape(self.enc_inputs)[0]

        if self.mode == "training":
            self.dec_inputs = tf.placeholder(
                tf.int32,
                shape=[None, self.dec_sentence_length + 1],
                name='output_sentences')

            self.dec_inputs_length = tf.placeholder(
                tf.int32,
                shape=[None, ],
                name='sentences_length')

            self.target_inputs = tf.placeholder(
                tf.int32,
                shape=[None, self.dec_sentence_length + 1],
                name='output_sentences')

            # decoder_inputs_length_train: [batch_size]
            self.train_dec_inputs_length = self.dec_inputs_length + 1


    def _make_cell(self, hidden_size, cell_multi=False):
        single_cell = None
        if self.cell_type == "RNN":
            single_cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_size)
        elif self.cell_type == "LSTM":
            single_cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, name='basic_lstm_cell')
        elif self.cell_type == "GRU":
            single_cell = tf.contrib.rnn.GRUCell(num_units=hidden_size)

        if cell_multi:
            cells = [single_cell] * self.num_layer
            return tf.contrib.rnn.MultiRNNCell(cells)
        else:
            return single_cell

    def _build_encoder(self):
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE) as scope:
            with tf.device('/cpu:0'):
                init = tf.contrib.layers.xavier_initializer()
                self.enc_embedding = \
                    tf.get_variable('embeeding',
                                    #shape=[self.enc_vocab_size + 1, self.enc_emb_size],
                                    initializer=tf.random_uniform([self.enc_vocab_size + 1, self.enc_emb_size]),
                                    dtype=tf.float32)
            # sentence -> vetor
            enc_emb_inputs = tf.nn.embedding_lookup(
                params=self.enc_embedding,
                ids=self.enc_inputs,
                name="emb_inputs")

            # make multi cell
            ''' [Batch_size x enc_sent_len x embedding_size]'''
            enc_cell = self._make_cell(hidden_size=self.hidden_size,
                                       cell_multi=False)

            ''' enc_outputs: [batch_size x enc_sent_len x embedding_size] '''
            ''' enc_last_state: [batch_size x embedding_size] '''
            self.enc_outputs, self.enc_last_state = tf.nn.dynamic_rnn(
                cell=enc_cell,
                inputs=enc_emb_inputs,
                sequence_length=self.enc_inputs_length,
                time_major=False,
                dtype=tf.float32)

    def _build_decoder(self):
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE) as scope:
            with tf.device('/cpu:0'):
                init = tf.contrib.layers.xavier_initializer()
                self.dec_embedding = \
                    tf.get_variable('embeeding',
                                    #shape=[self.dec_vocab_size + 1, self.dec_emb_size],
                                    initializer=tf.random_uniform([self.dec_vocab_size + 3, self.dec_emb_size]),
                                    dtype=tf.float32)

            dec_cell = self._make_cell(hidden_size=self.hidden_size, cell_multi=False)

            attn_mech = tf.contrib.seq2seq.LuongAttention(
                num_units=self.attn_size,
                memory=self.enc_outputs,
                memory_sequence_length=self.enc_inputs_length,
                name='LuongAttention')

            dec_cell = tf.contrib.seq2seq.AttentionWrapper(
                cell=dec_cell,
                attention_mechanism=attn_mech,
                attention_layer_size=self.attn_size,
                name='Attention_Wrapper')

            initial_state = dec_cell.zero_state(dtype=tf.float32, batch_size=self.batch_size)

            output_layer = Dense(self.dec_vocab_size + 3, name='output_projection')

            if self.mode == "training":
                # dec_emb_inputs: [batch_size, max_time_step + 1, embedding_size]
                dec_emb_inputs = tf.nn.embedding_lookup(
                    self.dec_embedding, self.dec_inputs, name='emb_inputs')

                training_helper = tf.contrib.seq2seq.TrainingHelper(
                    inputs=dec_emb_inputs,
                    sequence_length=self.train_dec_inputs_length,
                    time_major=False,
                    name="training_helper")
                training_decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell=dec_cell,
                    helper=training_helper,
                    initial_state=initial_state,
                    output_layer=output_layer)

                max_dec_len = tf.reduce_max(self.train_dec_inputs_length, name='max_dec_len')

                train_dec_outputs, train_dec_last_state, train_dec_output_length = tf.contrib.seq2seq.dynamic_decode(
                    decoder=training_decoder,
                    output_time_major=False,
                    impute_finished=True,
                    maximum_iterations=max_dec_len)

                ''' dec_outputs: collections.namedtuple(rnn_outputs, sample_id) '''
                ''' dec_outputs.rnn_output: [batch_size x max(dec_sequence_len) x dec_vocab_size+2], tf.float32 '''
                ''' dec_outputs.sample_id [batch_size], tf.int32 '''
                logits = tf.identity(train_dec_outputs.rnn_output, name='logits')

                targets = tf.slice(self.target_inputs, [0, 0], [-1, max_dec_len], 'targets')

                masks = tf.sequence_mask(self.train_dec_inputs_length, max_dec_len, dtype=tf.float32, name='masks')
                self.batch_loss = tf.contrib.seq2seq.sequence_loss(
                    logits=logits,
                    targets=targets,
                    weights=masks,
                    name='batch_loss')
                self.valid_predictions = tf.identity(train_dec_outputs.sample_id, name='valid_preds')

            elif self.mode == "inference":

                start_tokens = tf.ones([self.batch_size, ], tf.int32) * self.start_token
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

                infer_dec_outputs, infer_dec_last_state, infer_dec_output_length = tf.contrib.seq2seq.dynamic_decode(
                    inference_decoder,
                    output_time_major=False,
                    impute_finished=True,
                    maximum_iterations=500)

                # [batch_size x dec_sentence_length], tf.int32
                self.predictions = tf.identity(infer_dec_outputs.sample_id, name='predictions')
                # equivalent to tf.argmax(infer_dec_outputs.rnn_output, axis=2, name='predictions')

                # List of training variables
                # self.training_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

    def add_training_op(self):
        self.training_op = tf.train.AdamOptimizer(self.learning_rate, name='training_op').minimize(self.batch_loss)

    def save(self, sess, var_list=None, save_path=None):
        print('Saving model at {save_path}')
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
        self._init_placeholder()
        self._build_encoder()
        self._build_decoder()

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
                enc_batch_tokens = []
                dec_batch_tokens = []
                target_batch_tokens = []

                enc_batch_sent_lens = []
                dec_batch_sent_lens = []

                for enc_sent in input_batch:
                    tokens, sent_len = self.util.sent2idx(enc_sent,
                                                          vocab=self.enc_vocab,
                                                          max_sentence_length=self.enc_sentence_length)
                    enc_batch_tokens.append(tokens)
                    enc_batch_sent_lens.append(sent_len)

                for dec_sent in target_batch:
                    tokens, sent_len = self.util.sent2idx(dec_sent,
                                                          vocab=self.dec_vocab,
                                                          max_sentence_length=self.dec_sentence_length,
                                                          is_dec=True)
                    dec_batch_tokens.append(tokens)
                    dec_batch_sent_lens.append(sent_len)

                for dec_sent in target_batch:
                    tokens = self.util.sent2idx(dec_sent,
                                                vocab=self.dec_vocab,
                                                max_sentence_length=self.dec_sentence_length,
                                                is_target=True)
                    target_batch_tokens.append(tokens)
                # Evaluate 3 ops in the graph
                # => valid_predictions, loss, training_op(optimzier)
                batch_valid_preds, batch_loss, _ = sess.run(
                    [self.valid_predictions, self.batch_loss, self.training_op],
                    feed_dict={
                        self.enc_inputs: enc_batch_tokens,
                        self.enc_inputs_length: enc_batch_sent_lens,
                        self.dec_inputs: dec_batch_tokens,
                        self.dec_inputs_length: dec_batch_sent_lens,
                        self.target_inputs: target_batch_tokens
                    }
                )
                # loss_history.append(batch_loss)
                epoch_loss += batch_loss
                all_preds.append(batch_valid_preds)

            loss_history.append(epoch_loss)

            # Logging every 400 epochs
            if epoch % 100 == 0:
                print('Epoch', epoch)
                for input_batch, target_batch, batch_preds in zip(input_batches, target_batches, all_preds):
                    for input_sent, target_sent, pred in zip(input_batch, target_batch, batch_preds):
                        print(f'\tInput: {input_sent}')
                        print(f'\tPrediction:', self.util.idx2sent(pred, reverse_vocab=self.dec_reverse_vocab))
                        print(f'\tTarget: {target_sent}\n')
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
            tokens, sent_len = self.util.sent2idx(input_sent)
            batch_tokens.append(tokens)
            batch_sent_lens.append(sent_len)

        batch_preds = sess.run(
            self.predictions,
            feed_dict={
                self.enc_inputs: batch_tokens,
                self.enc_inputs_length: batch_sent_lens,
            })

        for input_sent, target_sent, pred in zip(input_batch, target_batch, batch_preds):
            print('Input:', input_sent)
            print('Prediction:', self.util.idx2sent(pred, reverse_vocab=self.dec_reverse_vocab))
            print('Target:', target_sent, '\n')

def main(_):
    util = DataUtil()
    input_batches, target_batches = util.load_data()
    config = Config()

    # 트레이닝
    tf.reset_default_graph()
    with tf.Session() as sess:
        model = Seq2Seq(mode="training")
        model.build()
        data = (input_batches, target_batches)
        loss_history = model.train(sess, data, from_scratch=True, save_path=model.ckpt_dir + f'epoch_{model.n_epoch}')

    plt.figure(figsize=(20, 10))
    plt.scatter(range(model.n_epoch), loss_history)
    plt.title('Learning Curve')
    plt.xlabel('Global step')
    plt.ylabel('Loss')
    plt.show()

    tf.reset_default_graph()
    with tf.Session() as sess:
        model = Seq2Seq(mode='inference')
        model.build()
        for input_batch, target_batch in zip(input_batches, target_batches):
            data = (input_batch, target_batch)
            model.inference(sess, data, load_ckpt=model.ckpt_dir + f'epoch_{model.n_epoch}_attention')

if __name__ == "__main__":
    tf.app.run()