import tensorflow as tf
import random
import os

from algorithm2.model import seq2seq
from algorithm2.dialog import Dialog
from algorithm2.config import FLAGS
import numpy as np


def train(epoch=FLAGS.epoch):
    tool = Dialog()
    # data loading
    question, answer = tool.loading_data()
    word_to_ix, ix_to_word = tool.make_dict_all_cut(question + answer, minlength=0, maxlength=3, jamo_delete=True,
                                                    special_delete=True)

    # parameters
    multi = True
    forward_only = False
    hidden_size = 300
    vocab_size = len(ix_to_word)
    num_layers = 3
    learning_rate = 0.001
    batch_size = 11
    encoder_size = tool.check_doclength(question, sep=True)
    decoder_size = tool.check_doclength(answer, sep=True)  # (Maximum) number of time steps in this batch

    # transform data
    encoderinputs, decoderinputs, targets_, targetweights = \
        tool.make_inputs(question, answer, word_to_ix, encoder_size=encoder_size, decoder_size=decoder_size)

    model = seq2seq(multi=multi, hidden_size=hidden_size, num_layers=num_layers,
                    learning_rate=learning_rate, batch_size=batch_size, vocab_size=vocab_size,
                    encoder_size=encoder_size, decoder_size=decoder_size, forward_only=forward_only)
    
    with tf.Session() as sess:
        # TODO: 세션을 로드하고 로그를 위한 summary 저장등의 로직을 Seq2Seq 모델로 넣을 필요가 있음
        ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("다음 파일에서 모델을 읽는 중 입니다..", ckpt.model_checkpoint_path)
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("새로운 모델을 생성하는 중 입니다.")
            sess.run(tf.global_variables_initializer())
        
        # 학습 변수
        start = 0
        end = batch_size
        for current_step in range(epoch):
            if end > len(answer):
                start = 0
                end = batch_size
    
            #Get a batch and make a step
            encoder_inputs, decoder_inputs, targets, target_weights = tool.make_batch(encoderinputs[start:end],
                                                                                      decoderinputs[start:end],
                                                                                      targets_[start:end],
                                                                                      targetweights[start:end])
            # encoder_inputs, decoder_inputs, targets, target_weights = tool.make_batch(encoderinputs,
            #                                                                           decoderinputs,
            #                                                                           targets_,
            #                                                                           targetweights)
            # 학습
            if current_step % 10 == 0:
                for i in range(decoder_size - 2):
                    decoder_inputs[i + 1] = np.array([word_to_ix['<PAD>']] * batch_size)
                loss = model.step(sess, encoder_inputs, decoder_inputs, targets, target_weights, forward_only)
                print('Step:', '%06d' % model.global_step.eval(), 'cost =', '{:.6f}'.format(loss))

            start += batch_size
            end += batch_size

        checkpoint_path = os.path.join(FLAGS.train_dir, FLAGS.ckpt_name)
        model.saver.save(sess, checkpoint_path, global_step=model.global_step)

    print('최적화 완료!')

#
# def test():
#     print("\n=== 예측 테스트 ===")
#     tool = Dialog()
#     question, answer = tool.loading_data()
#     word_to_ix, ix_to_word = tool.make_dict_all_cut(question + answer, minlength=0, maxlength=3, jamo_delete=True,
#                                                     special_delete=True)
#
#     # parameters
#     multi = True
#     forward_only = False
#     hidden_size = 300
#     vocab_size = len(ix_to_word)
#     num_layers = 3
#     learning_rate = 0.001
#     batch_size = 11
#     encoder_size = tool.check_doclength(question, sep=True)
#     decoder_size = tool.check_doclength(answer, sep=True)  # (Maximum) number of time steps in this batch
#
#     # transform data
#     encoderinputs, decoderinputs, targets_, targetweights = \
#         tool.make_inputs(question, answer, word_to_ix, encoder_size=encoder_size, decoder_size=decoder_size)
#
#     model = seq2seq(multi=multi, hidden_size=hidden_size, num_layers=num_layers,
#                     learning_rate=learning_rate, batch_size=batch_size, vocab_size=vocab_size,
#                     encoder_size=encoder_size, decoder_size=decoder_size, forward_only=forward_only)
#
#     with tf.Session() as sess:
#         ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
#         print("다음 파일에서 모델을 읽는 중 입니다..", ckpt.model_checkpoint_path)
#         model.saver.restore(sess, ckpt.model_checkpoint_path)
#
#         # Get a batch and make a step
#         encoder_inputs, decoder_inputs, targets, target_weights = tool.make_batch(encoderinputs,
#                                                                                   decoderinputs,
#                                                                                   targets_,
#                                                                                   targetweights)
#         pick = random.randrange(0, len(encoder_inputs))
#         #
#         # # 추측
#         # if (current_step % 10) == 0:
#         #     outputs_logits = model.step(sess, encoder_inputs, decoder_inputs, targets, target_weights, True)
#         #     predict = [np.argmax(output, axis=1)[0] for output in outputs_logits]
#         #     predict = ' '.join(ix_to_word[ix][0] for ix in predict)
#         #
#         #     real = [word[0] for word in targets]
#         #     real = ' '.join(ix_to_word[ix][0] for ix in real)
#         #     print('---- 타겟값 : %s \n 예측값 : %s ----\n' % (real, predict))
#         #
#         # expect, outputs, accuracy = model.test(sess, [encoder_inputs[pick]], [decoder_inputs[pick]], [targets[pick]])
#         #
#         # expect = dialog.decode(expect)
#         # outputs = dialog.decode(outputs)
#         #
#         # input = dialog.decode([dialog.examples[pick]], True)
#         # expect = dialog.decode([dialog.examples[pick]], True)
#         #outputs = dialog.cut_eos(outputs[0])
#         #
#         # print("\n정확도:", accuracy)
#         # print("랜덤 결과\n")
#         # print("    입력값:", input)
#         # print("    실제값:", expect)
#         # print("    예측값:", outputs)


def main(_):

    if FLAGS.train:
        train(epoch=FLAGS.epoch)
    # elif FLAGS.test:
    #     test()


if __name__ == "__main__":
    tf.app.run()