import tensorflow as tf
import numpy as np
import math
import sys

from algorithm.config import FLAGS
from algorithm.model import Seq2Seq
from algorithm.dialog import Dialog


class ChatBot:

    def __init__(self, train_dir):
        self.dialog = Dialog()
        self.dialog.load_vocab()

        self.model = Seq2Seq(self.dialog.vocab_size)

        self.sess = tf.Session()
        ckpt = tf.train.get_checkpoint_state(train_dir)
        print("다음 파일에서 모델을 읽는 중 입니다..", ckpt.model_checkpoint_path)
        self.model.saver.restore(self.sess, ckpt.model_checkpoint_path)

    def run(self):
        sys.stdout.write("> ")
        sys.stdout.flush()
        line = sys.stdin.readline()

        while line:
            print(self._get_replay(line.strip()))

            sys.stdout.write("\n> ")
            sys.stdout.flush()

            line = sys.stdin.readline()

    def _decode(self, enc_input, dec_input):
        if type(dec_input) is np.ndarray:
            dec_input = dec_input.tolist()

        # TODO: ???? ??? ???? ?? ??? ??? ????? ???? ?????
        input_len = int(math.ceil((len(enc_input) + 1) * 1.5))

        enc_input, dec_input, _ = self.dialog.transform(enc_input, dec_input,
                                                        input_len,
                                                        FLAGS.max_decode_len)

        return self.model.predict(self.sess, enc_input, dec_input)

    def _get_replay(self, msg):
        enc_input = self.dialog.tokenizer([[msg]], apprun=True)
        enc_input = sum(enc_input, [])
        enc_input = self.dialog.tokens_to_ids(enc_input)
        dec_input = []

        curr_seq = 0
        for i in range(FLAGS.max_decode_len):
            outputs = self._decode([enc_input], [dec_input])
            if self.dialog.is_eos(outputs[0][curr_seq]):
                break
            elif self.dialog.is_defined(outputs[0][curr_seq]) is not True:
                dec_input.append(outputs[0][curr_seq])
                curr_seq += 1

        reply = self.dialog.decode([dec_input], True)
        print("ai answer: ", reply)
        return reply


def main(_):
    print("메디봇 가동중입니다...\n")

    chatbot = ChatBot(FLAGS.train_dir)
    chatbot.run()


if __name__ == "__main__":
    tf.app.run()