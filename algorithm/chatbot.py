import tensorflow as tf
from algorithm.model import Seq2Seq

def _get_answer(question_msg):
    tf.reset_default_graph()
    with tf.Session() as sess:
        model = Seq2Seq(mode='inference')
        model.build()
        predict = model.prediction(sess, question_msg, load_ckpt=model.ckpt_dir + f'epoch_{model.n_epoch}')
    print(predict)
    return predict


