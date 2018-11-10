# 어휘 사전과 워드 임베딩을 만들고, 학습을 위해 대화 데이터를 읽어들이는 유틸리티들의 모음
import tensorflow as tf
import numpy as np
import re
import algorithm.db as db
from algorithm.config import FLAGS

class Dialog():

    _PAD_ = "<PAD>"  # 빈칸 채우는 심볼
    _STA_ = "<S>"  # 디코드 입력 시퀀스의 시작 심볼
    _EOS_ = "<E>"  # 디코드 입출력 시퀀스의 종료 심볼
    _UNK_ = "<UNK>"  # 사전에 없는 단어를 나타내는 심볼

    _PAD_ID_ = 0
    _STA_ID_ = 1
    _EOS_ID_ = 2
    _UNK_ID_ = 3
    _PRE_DEFINED_ = [_PAD_, _STA_, _EOS_, _UNK_]

    def __init__(self):
        self.vocab_list = []
        self.vocab_dict = {}
        self.vocab_size = 0
        self.examples = []

        self._index_in_epoch = 0

    def decode(self, indices, string=False):
        tokens = [[self.vocab_list[i] for i in dec] for dec in indices]

        if string:
            return self._decode_to_string(tokens[0])
        else:
            return tokens

    def _decode_to_string(self, tokens):
        text = ' '.join(tokens)
        return text.strip()

    def cut_eos(self, indices):
        eos_idx = indices.index(self._EOS_)
        return indices[:eos_idx]

    def is_eos(self, voc_id):
        return voc_id == self._EOS_ID_

    def is_defined(self, voc_id):
        return voc_id in self._PRE_DEFINED_

    def _max_len(self, batch_set):
        max_size = 0
        for sentence in batch_set:
            if len(sentence) > max_size:
                max_size = len(sentence)
        return max_size

    def _pad(self, seq, max_len, start=None, eos=None):
        if start:
            padded_seq = [self._STA_ID_] + seq
        elif eos:
            padded_seq = seq + [self._EOS_ID_]
        else:
            padded_seq = seq

        if len(padded_seq) < max_len:
            return padded_seq + ([self._PAD_ID_] * (max_len - len(padded_seq)))
        else:
            return padded_seq

    def _pad_left(self, seq, max_len):
        if len(seq) < max_len:
            return ([self._PAD_ID_] * (max_len - len(seq))) + seq
        else:
            return seq

    def transform(self, input, output, input_max, output_max):
        enc_input = []
        dec_input = []
        target = []
        for i, o in zip(input, output):
            # 구글 방식으로 입력을 인코더에 역순으로 입력한다.
            tmp_enc = self._pad(i, input_max)
            tmp_enc.reverse()
            enc_input.append(np.eye(self.vocab_size)[tmp_enc])

            tmp_dec = self._pad(o, output_max + 1, start=True)
            dec_input.append(np.eye(self.vocab_size)[tmp_dec])

            tmp_tar = self._pad(o, output_max + 1, eos=True)
            target.append(tmp_tar)

        return enc_input, dec_input, target

    def make_batch(self):
        enc_input = []
        dec_input = []

        for n in range(0, len(self.examples), 2):
            enc_input.append(self.examples[n])
            dec_input.append(self.examples[n + 1])

        # TODO: 구글처럼 버킷을 이용한 방식으로 변경
        # 간단하게 만들기 위해 구글처럼 버킷을 쓰지 않고 같은 배치는 같은 사이즈를 사용하도록 만듬
        max_len_input = self._max_len(enc_input)
        max_len_output = self._max_len(dec_input)

        enc, dec, tar = self.transform(enc_input, dec_input, max_len_input, max_len_output)
        return enc, dec, tar

    def tokens_to_ids(self, tokens):
        ids = []
        for t in tokens:
            if t in self.vocab_dict:
                ids.append(self.vocab_dict[t])
            else:
                ids.append(self._UNK_ID_)

        return ids

    def ids_to_tokens(self, ids):
        tokens = []
        for i in ids:
            tokens.append(self.vocab_list[i])
        return tokens

    def tokenizer(self, sentence_list, build=None, load=None, apprun=None):
        words = []
        for sentence in sentence_list:
            for fragment in sentence:
                words.extend([fragment.strip().split()])
        if build:
            build_words = []
            for word_list in words:
                for word in word_list:
                    build_words.append(word)
            return build_words
        if load or apprun:
            return words

    # 어휘 사전 제작 및 디비에 저장
    def build_vocab(self):
        sequence_data = db.select_chat_sequence()
        words = self.tokenizer(sequence_data, build=True)

        # 어휘를 디비에 저장
        words_dic = [{'vocab': r, 'morpheme': ''} for r in list(set(words))]
        db.delete_in_chat_vocab(words_dic)

    # 어휘 사전 로드
    def load_vocab(self):
        self.vocab_list = self._PRE_DEFINED_ + []

        vocab_list = db.select_chat_vocab()
        for row in vocab_list:
            self.vocab_list.append(row[1])

        # {'_PAD_': 0, '_STA_': 1, '_EOS_': 2, '_UNK_': 3, 'Hello': 4, 'World': 5, ...}
        self.vocab_dict = {n: i for i, n in enumerate(self.vocab_list)}
        self.vocab_size = len(self.vocab_list)
        FLAGS.max_decode_len = self.vocab_size
        print('어휘 사전을 불러왔습니다.')

    # 예제 로드
    def load_examples(self):
        self.examples = []
        sequence_data = db.select_chat_sequence()

        tokens = self.tokenizer(sequence_data, load=True)
        for sentence in tokens:
            ids = self.tokens_to_ids(sentence)
            self.examples.append(ids)


def main(_):
    dialog = Dialog()

    if FLAGS.voc_test:
        print("데이터베이스 데이터를 통해 어휘 사전을 테스트합니다.")
        dialog.load_vocab()
        dialog.load_examples()

        enc, dec, target = dialog.make_batch()

    elif FLAGS.voc_build:
        dialog.build_vocab()

    elif FLAGS.voc_test:
        dialog.load_vocab()
        print(dialog.vocab_dict)


if __name__ == "__main__":
    tf.app.run()