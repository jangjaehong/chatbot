import tensorflow as tf


# tf.app.flags.DEFINE_string("train_dir", "/home/jjh/pyproject/chatbot/analysis/model", "학습한 신경망을 저장할 폴더")
tf.app.flags.DEFINE_string("train_dir", "D:/chatbot/analysis/model", "학습한 신경망을 저장할 폴더")
tf.app.flags.DEFINE_string("log_dir", "./logs", "로그를 저장할 폴더")
tf.app.flags.DEFINE_string("ckpt_name", "conversation.ckpt", "체크포인트 파일명")

tf.app.flags.DEFINE_boolean("train", False, "학습을 진행합니다.")
tf.app.flags.DEFINE_boolean("test", True, "테스트를 합니다.")
tf.app.flags.DEFINE_integer("epoch", 100, "총 학습 반복 횟수")

FLAGS = tf.app.flags.FLAGS