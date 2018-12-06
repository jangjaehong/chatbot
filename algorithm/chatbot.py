import tensorflow as tf
from algorithm.model import Seq2Seq
from algorithm.util import DataUtil

ckpt_dir = './algorithm/model'
datautil = DataUtil()
vocab = datautil.dec_reverse_vocab


def _get_answer(question_msg):
    with tf.Session() as sess:
        model = Seq2Seq(mode='inference')
        model.build()
        model.restore(sess, ckpt_path=f'./algorithm/model/chat_model_{model.n_epoch}.ckpt')
        predict = model.prediction(sess, question_msg)

    answer = datautil.idx2sent_pad_removce(predict[0], vocab)
    return answer
#
# chatbot = ChatBot()
# answer = chatbot._get_answer("고지혈증")
# print(answer)