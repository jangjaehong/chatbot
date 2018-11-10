import os as os
import time as time
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import math as math

from algorithm3.model import Seq2Seq
from algorithm3.config import FLAGS
from algorithm3.dialog import Dialog


def create_model(session, enc_vocab_size, dec_vocab_size):
    FLAGS.src_vocab_size = enc_vocab_size
    FLAGS.tgt_vocab_size = dec_vocab_size

    model = Seq2Seq(FLAGS)

    ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print('모델 읽어오는중...')
        model.restore(session, ckpt.model_checkpoint_path)

    else:
        if not os.path.exists(FLAGS.model_dir):
            os.makedirs(FLAGS.model_dir)
        print('새로운 모델 생성중...')
        session.run(tf.global_variables_initializer())

    return model


def train():
    # Load parallel data to train
    print('트레이닝 데이터 읽어오는중..')
    enc_sentence_length = 100
    dec_sentence_length = 100
    dialog = Dialog()
    question, answer = dialog.load_data()

    all_input_sentence = []
    for input in question:
        all_input_sentence.extend(input)

    all_target_sentence = []
    for target in answer:
        all_target_sentence.extend(target)

    enc_vocab, enc_reverse_vocab, enc_vocab_size = dialog.build_vocab(all_input_sentence)
    dec_vocab, dec_reverse_vocab, dec_vocab_size = dialog.build_vocab(all_target_sentence, is_target=True)

    input_tokens = []
    target_tokens = []
    enc_sentence_lengths = []
    dec_sentence_lengths = []

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                          log_device_placement=FLAGS.log_device_placement)) as sess:
                                          #gpu_options=tf.GPUOptions(allow_growth=True))) as sess:

        # Create a new model or reload existing checkpoint
        model = create_model(sess, enc_vocab_size, dec_vocab_size)

        # Create a log writer object
        log_writer = tf.summary.FileWriter(FLAGS.model_dir, graph=sess.graph)

        step_time, loss = 0.0, 0.0
        words_seen, sents_seen = 0, 0
        start_time = time.time()

        # Training loop
        print('훈련시작...')
        for epoch_idx in tqdm(range(FLAGS.epochs)):
            if model.global_epoch_step.eval() >= FLAGS.epochs:
                print('Training is already complete.', 'current epoch:{}, max epoch:{}'.format(model.global_epoch_step.eval(), FLAGS.epochs))
                break

            # Get a batch from training parallel data
            for input_batch, target_batch in zip(question, answer):
                input_batch_tokens = []
                target_batch_tokens = []
                enc_sentence_lengths = []
                dec_sentence_lengths = []

                for input_sent in input_batch:
                    print("input_sent:", input_sent)
                    tokens, sent_len = dialog.sent2idx(input_sent)
                    input_batch_tokens.append(tokens)
                    enc_sentence_lengths.append(sent_len)

                for target_sent in target_batch:
                    tokens, sent_len = dialog.sent2idx(target_sent,
                                                vocab=dec_vocab,
                                                max_sentence_length=dec_sentence_length,
                                                is_target=True)
                    target_batch_tokens.append(tokens)
                    dec_sentence_lengths.append(sent_len)


                # Execute a single training step
                step_loss, summary = model.train(sess, encoder_inputs=input_batch_tokens, encoder_inputs_length=enc_sentence_lengths,
                                                 decoder_inputs=target_batch_tokens, decoder_inputs_length=dec_sentence_lengths)
                loss += float(step_loss) / FLAGS.display_freq
                words_seen += float(np.sum(enc_sentence_lengths + dec_sentence_lengths))
                sents_seen += float(input_batch_tokens.shape[0])  # batch_size

            if model.global_step.eval() % FLAGS.display_freq == 0:
                avg_perplexity = math.exp(float(loss)) if loss < 300 else float("inf")

                time_elapsed = time.time() - start_time
                step_time = time_elapsed / FLAGS.display_freq

                words_per_sec = words_seen / time_elapsed
                sents_per_sec = sents_seen / time_elapsed

                print('Epoch ', model.global_epoch_step.eval(), 'Step ', model.global_step.eval(),
                      'Perplexity {0:.2f}'.format(avg_perplexity), 'Step-time ', step_time,
                      '{0:.2f} sents/s'.format(sents_per_sec), '{0:.2f} words/s'.format(words_per_sec))

                loss = 0
                words_seen = 0
                sents_seen = 0
                start_time = time.time()

                # Record training summary for the current batch
                log_writer.add_summary(summary, model.global_step.eval())

                # Increase the epoch index of the model
                model.global_epoch_step_op.eval()
                print('Epoch {0:} DONE'.format(model.global_epoch_step.eval()))
        print('Saving the last model..')
        checkpoint_path = os.path.join(FLAGS.model_dir, FLAGS.model_name)
        model.save(sess, checkpoint_path, global_step=model.global_step)
    print('훈련 완료!!!')


def main(_):
    train()


if __name__ == '__main__':
    tf.app.run()