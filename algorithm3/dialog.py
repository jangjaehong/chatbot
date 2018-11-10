import algorithm3.db as db
import re
import tensorflow as tf
import numpy as np
from algorithm3.config import FLAGS
from collections import Counter
S = '<S>'
PAD = '<PAD>'
UNK = '<UNK>'

Token = [S, PAD, UNK]
start_token = Token.index(S)
pad_token = Token.index(PAD)
unk_token = Token.index(UNK)


class Dialog:
    # 질의 응답 데이터 로드
    def load_data(self):
        contents = db.select_chat_sequence()
        question = []
        answer = []
        if contents:
            for sequence in contents:
                # 목적이 같은 질문을 ,를 통해 분리
                docs = [sequence[0].split(',')]
                for doc in docs:
                    # 목적이 같은 여러 형태의 질문에 같은 답변 저장
                    question.append(doc)
                    answer.append([sequence[1]] * len(doc))
        return question, answer

    # 어휘 분리
    def tokenizer(self, sentence):
        tokens = re.findall(r"[\w]+|[^\s\w]", sentence)
        return tokens

    # 어휘 사전 제작 및 디비에 저장
    def build_vocab(self, sentences, is_target=False, max_vocab_size=None):
        # 사전 제작
        word_counter = Counter()
        vocab = dict()
        reverse_vocab = dict()

        for sentence in sentences:
            tokens = self.tokenizer(sentence)
            word_counter.update(tokens)

        if max_vocab_size is None:
            max_vocab_size = len(word_counter)

        if is_target:
            vocab['_GO'] = 0
            vocab['_PAD'] = 1
            vocab_idx = 2
            for key, value in word_counter.most_common(max_vocab_size):
                vocab[key] = vocab_idx
                vocab_idx += 1
        else:
            vocab['_PAD'] = 0
            vocab_idx = 1
            for key, value in word_counter.most_common(max_vocab_size):
                vocab[key] = vocab_idx
                vocab_idx += 1

        for key, value in vocab.items():
            reverse_vocab[value] = key

        return vocab, reverse_vocab, max_vocab_size

    def token2idx(self, word, vocab):
        return vocab[word]

    def sent2idx(self, sent, vocab, max_sentence_length, is_target=False):
        tokens = self.tokenizer(sent)
        current_length = len(tokens)
        pad_length = max_sentence_length - current_length
        if is_target:
            return [0] + [self.token2idx(token, vocab) for token in tokens] + [1] * pad_length, current_length
        else:
            return [self.token2idx(token, vocab) for token in tokens] + [0] * pad_length, current_length

def main(_):
    dialog = Dialog()
    question, answer = dialog.load_data()

    all_input_sentence = []
    for input in question:
        all_input_sentence.extend(input)

    all_target_sentence = []
    for target in answer:
        all_target_sentence.extend(target)

    enc_vocab, enc_reverse_vocab, enc_vocab_size = dialog.build_vocab(all_input_sentence)
    dec_vocab, dec_reverse_vocab, dec_vocab_size = dialog.build_vocab(all_target_sentence, is_target=True)

    input_tokens = []
    target_tokens = []
    enc_sentence_lengths = []
    dec_sentence_lengths = []

    for input_sent, target_sent in zip(question, answer):
        for input in input_sent:
            token, token_len = dialog.sent2idx(input, vocab=enc_vocab, max_sentence_length=enc_vocab_size)
            input_tokens.append(token)
            enc_sentence_lengths.append(token_len)

        for target in target_sent:
            token, token_len = dialog.sent2idx(target, vocab=dec_vocab, max_sentence_length=dec_vocab_size, is_target=True)
            target_tokens.append(token)
            dec_sentence_lengths.append(token_len)


if __name__ == "__main__":
    tf.app.run()

