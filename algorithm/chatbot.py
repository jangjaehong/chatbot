import tensorflow as tf
from algorithm.model import Seq2Seq
from algorithm.util import DataUtil

def _get_answer(self, question_msg):
    ckpt_dir = './algorithm/model'

    datautil = DataUtil()
    vocab = self.datautil.dec_reverse_vocab

    sess = tf.Session()
    model = Seq2Seq(mode='inference')
    model.build()
    model.restore(sess, ckpt_path=ckpt_dir + f'/chat_model_{model.n_epoch}.ckpt')
    predict = model.prediction(sess, question_msg)
    answer = datautil.idx2sent_pad_removce(predict[0], vocab)
    return answer
#
# chatbot = ChatBot()
# answer = chatbot._get_answer("고지혈증")
# print(answer)