import tensorflow as tf
from algorithm.model import Seq2Seq
from algorithm.util import DataUtil

def _get_answer(question_msg):
    datautil = DataUtil()

    tf.reset_default_graph()
    with tf.Session() as sess:
        model = Seq2Seq(mode='inference')
        model.build()
        predict = model.prediction(sess, question_msg, load_ckpt=model.ckpt_dir + f'epoch_{model.n_epoch}')

    print(datautil.idx2sent(predict, datautil.dec_reverse_vocab))
    return predict


_get_answer("일일 체크")
