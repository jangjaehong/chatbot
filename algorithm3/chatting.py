import tensorflow as tf
from algorithm3.model import Seq2Seq
from algorithm3.config import FLAGS
from algorithm3.dialog import Dialog


def load_model(session):
    model = Seq2Seq(FLAGS)
    # if tf.train.checkpoint_exists(FLAGS.model_path):
    #     print('Reloading model parameters..')
    #     model.saver.restore(session, FLAGS.model_path)
    # else:
    #     raise ValueError('No such file:[{}]'.format(FLAGS.model_path))
    ckpt = tf.train.get_checkpoint_state(FLAGS.model_path)
    print("다음 파일에서 모델을 읽는 중 입니다..", ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
    return model


def decode():
    # Load parallel data to train
    print('트레이닝 데이터 읽어오는중..')
    dialog = Dialog()
    question, answer = dialog.load_data()
    word_to_idx, idx_to_word = dialog.load_vocab()
    vocab_size = len(idx_to_word)
    encoder_size = dialog.max_sequence_len(question)
    decoder_size = dialog.max_sequence_len(answer)
    enc_input, dec_input, target = dialog.make_input(question, answer, word_to_idx, encoder_size, decoder_size)

    # Initiate TF session
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                          log_device_placement=FLAGS.log_device_placement)) as sess:
                                         # gpu_options=tf.GPUOptions(allow_growth=True))) as sess:

        # # Reload existing checkpoint
        # try:
        #     model = load_model(sess)
        #     #print(idx, source_seq)
        #
        source, source_len = dialog.prepare_batch(enc_input)
        print(source)
        print(source_len)
        #     # predicted_ids: GreedyDecoder; [batch_size, max_time_step, 1]
        #     # BeamSearchDecoder; [batch_size, max_time_step, beam_width]
        #     predicted_ids = model.predict(sess, encoder_inputs=source,
        #                                   encoder_inputs_length=source_len)
        #
        #     for seq in predicted_ids:
        #         print(dialog.seq2words(seq, idx_to_word))
        # except ValueError:
        #     print('오류')


def main(_):
    decode()


if __name__ == '__main__':
    tf.app.run()