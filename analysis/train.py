import tensorflow as tf
import os

from analysis.config import FLAGS
from analysis.model import LogisticRegression
from analysis.physical import Physical
from analysis.config import FLAGS


def train(physical, epoch=FLAGS.epoch):
    model = LogisticRegression(physical.data_size)

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

        x_data = physical.physical_data
        y_data = physical.bmi_data
        for step in range(epoch):
            _, cost = model.train(sess, x_data, y_data)
            if step % 10 == 0:
                model.write_logs(sess, writer, x_data, y_data)
                print('Step:', '%06d' % model.global_step.eval(), 'cost =', '{:.6f}'.format(cost))

        checkpoint_path = os.path.join(FLAGS.train_dir, FLAGS.ckpt_name)
        model.saver.save(sess, checkpoint_path, global_step=model.global_step)
    print('최적화 완료!')


def test(physical):
    model = LogisticRegression(physical.data_size)
    with tf.Session() as sess:

        x_data = physical.physical_data
        y_data = physical.bmi_data
        for step in range(5000):
            _, cost = model.train(sess, x_data, y_data)
            if step % 10 == 0:
                print('Step:', step, 'cost =', '{:.6f}'.format(cost))

        hypothesis, accuracy = model.test(sess, x_data, y_data)

        print("\n정확도:", accuracy)
        print("랜덤 결과\n")
        print("    입력값:", input)
        print("    실제값:", x_data)
        print("    예측값:", y_data)


def main(_):
    physical = Physical()
    physical.load_data("2018-09-01", "2018-09-20")

    if FLAGS.train:
        train(physical, epoch=FLAGS.epoch)
    elif FLAGS.test:
        test(physical)


if __name__ == "__main__":
    tf.app.run()