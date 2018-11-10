import tensorflow as tf
import random
import math
import os

from algorithm.config import FLAGS
from algorithm.model import Seq2Seq
from algorithm.dialog import Dialog
from algorithm.config import FLAGS
import numpy as np


def train(dialog, epoch=FLAGS.epoch):
    model = Seq2Seq(dialog.vocab_size)

    with tf.Session() as sess:
        # TODO: 세션을 로드하고 로그를 위한 summary 저장등의 로직을 Seq2Seq 모델로 넣을 필요가 있음
        ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("다음 파일에서 모델을 읽는 중 입니다..", ckpt.model_checkpoint_path)
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("새로운 모델을 생성하는 중 입니다.")
            sess.run(tf.global_variables_initializer())

        writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

        enc_input, dec_input, targets = dialog.make_batch()

        for step in range(epoch):
            _, loss = model.train(sess, enc_input, dec_input, targets)

            if (step + 1) % 10 == 0:
                model.write_logs(sess, writer, enc_input, dec_input, targets)

                print('Step:', '%06d' % model.global_step.eval(),
                      'cost =', '{:.6f}'.format(loss))

        checkpoint_path = os.path.join(FLAGS.train_dir, FLAGS.ckpt_name)
        model.saver.save(sess, checkpoint_path, global_step=model.global_step)

    print('최적화 완료!')


def test(dialog):
    print("\n=== 예측 테스트 ===")

    model = Seq2Seq(dialog.vocab_size)

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
        print("다음 파일에서 모델을 읽는 중 입니다..", ckpt.model_checkpoint_path)
        model.saver.restore(sess, ckpt.model_checkpoint_path)

        enc_input, dec_input, targets = dialog.make_batch()

        pick = random.randrange(0, len(enc_input))

        expect, outputs, accuracy = model.test(sess, [enc_input[pick]], [dec_input[pick]], [targets[pick]])

        expect = dialog.decode(expect)
        outputs = dialog.decode(outputs)

        input = dialog.decode([dialog.examples[pick]], True)
        expect = dialog.decode([dialog.examples[pick]], True)
        #outputs = dialog.cut_eos(outputs[0])

        print("\n정확도:", accuracy)
        print("랜덤 결과\n")
        print("    입력값:", input)
        print("    실제값:", expect)
        print("    예측값:", outputs)


def main(_):
    dialog = Dialog()

    dialog.load_vocab()
    dialog.load_examples()

    if FLAGS.train:
        train(dialog, epoch=FLAGS.epoch)
    elif FLAGS.test:
        test(dialog)


if __name__ == "__main__":
    tf.app.run()