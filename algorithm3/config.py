import tensorflow as tf

tf.app.flags.DEFINE_boolean('train_mode', True, 'Train 여부 설정')
tf.app.flags.DEFINE_boolean('decode_mode', False, 'Train 여부 설정')
tf.app.flags.DEFINE_string('model_dir', 'model/', 'Path to save model checkpoints')
tf.app.flags.DEFINE_string('model_path', 'model/', 'Path to a specific model checkpoint.')
tf.app.flags.DEFINE_string('model_name', 'translate.ckpt', 'File name used for model checkpoints')

# Network parameters
tf.app.flags.DEFINE_string('cell_type', 'lstm', 'RNN cell for encoder and decoder, default: lstm')
tf.app.flags.DEFINE_string('attention_type', 'bahdanau', 'Attention mechanism: (bahdanau, luong), default: bahdanau')
tf.app.flags.DEFINE_integer('hidden_dim', 1024, 'Number of hidden units in each layer')
tf.app.flags.DEFINE_integer('num_layer', 2, 'Number of layers in each encoder and decoder')
tf.app.flags.DEFINE_integer('embedding_dim', 500, 'Embedding dimensions of encoder and decoder inputs')
tf.app.flags.DEFINE_integer('src_vocab_size', 1000, 'Source vocabulary size')
tf.app.flags.DEFINE_integer('tgt_vocab_size', 1000, 'Target vocabulary size')

tf.app.flags.DEFINE_boolean('use_residual', True, 'Use residual connection between layers')
tf.app.flags.DEFINE_boolean('attn_input_feeding', False, 'Use input feeding method in attentional decoder')
tf.app.flags.DEFINE_boolean('use_dropout', True, 'Use dropout in each rnn cell')
tf.app.flags.DEFINE_float('dropout_rate', 0.3, 'Dropout probability for input/output/state units (0.0: no dropout)')

# Training parameters
tf.app.flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate')
tf.app.flags.DEFINE_float('max_gradient_norm', 1.0, 'Clip gradients to this norm')
tf.app.flags.DEFINE_integer('batch_size', 128, 'Batch size')
tf.app.flags.DEFINE_integer('epochs', 1000, 'Maximum # of training epochs')
tf.app.flags.DEFINE_integer('max_load_batches', 20, 'Maximum # of batches to load at one time')
tf.app.flags.DEFINE_integer('max_seq_length', 50, 'Maximum sequence length')
tf.app.flags.DEFINE_integer('display_freq', 100, 'Display training status every this iteration')
tf.app.flags.DEFINE_integer('save_freq', 11500, 'Save model checkpoint every this iteration')
tf.app.flags.DEFINE_integer('valid_freq', 1150000, 'Evaluate model every this iteration: valid_data needed')
tf.app.flags.DEFINE_boolean('beamsearch_decode', False, '')
tf.app.flags.DEFINE_boolean('shuffle_each_epoch', True, 'Shuffle training dataset for each epoch')
tf.app.flags.DEFINE_boolean('sort_by_length', True, 'Sort pre-fetched minibatches by their target sequence lengths')

# Decoding parameters
tf.app.flags.DEFINE_integer('beam_width', 12, 'Beam width used in beamsearch')
tf.app.flags.DEFINE_integer('decode_batch_size', 80, 'Batch size used for decoding')
tf.app.flags.DEFINE_integer('max_decode_step', 500, 'Maximum time step limit to decode')
tf.app.flags.DEFINE_boolean('write_n_best', False, 'Write n-best list (n=beam_width)')

# Runtime parameters
tf.app.flags.DEFINE_boolean('allow_soft_placement', True, 'Allow device soft placement')
tf.app.flags.DEFINE_boolean('log_device_placement', False, 'Log placement of ops on devices')

FLAGS = tf.app.flags.FLAGS
