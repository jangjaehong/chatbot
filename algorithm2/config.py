import tensorflow as tf
flags = tf.app.flags
FLAGS = tf.flags.FLAGS

flags.DEFINE_string("win_train_dir", "D:/chatbot/algorithm2/model", "학습한 신경망을 저장할 폴더")
flags.DEFINE_string("train_dir", "/home/jjh/pyproject/chatbot/algorithm2/model", "학습한 신경망을 저장할 폴더")
flags.DEFINE_string("log_dir", "./logs", "로그를 저장할 폴더")
flags.DEFINE_string("ckpt_name", "conversation.ckpt", "체크포인트 파일명")
flags.DEFINE_boolean("train", True, "학습을 진행합니다.")
flags.DEFINE_boolean("test", False, "테스트를 합니다.")
flags.DEFINE_integer("epoch", 10000, "총 학습 반복 횟수")
flags.DEFINE_boolean("voc_test", False, "어휘 사전을 테스트합니다.")
flags.DEFINE_boolean("voc_build", True, "주어진 대화 파일을 이용해 어휘 사전을 작성합니다.")


