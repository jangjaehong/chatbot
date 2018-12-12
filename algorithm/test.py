import re
from collections import Counter
import algorithm.db as db
import numpy as np
from konlpy.tag import Mecab


class DataUtil:

    def __init__(self):
        self._GO = '_GO'
        self.EOS = '_EOS'  # also function as PAD
        self.PAD = '_PAD'
        self.extra_tokens = [self._GO, self.EOS, self.PAD]

        self.start_token = self.extra_tokens.index(self._GO)  # start_token = 0
        self.end_token = self.extra_tokens.index(self.EOS)  # end_token = 1
        self.pad_token = self.extra_tokens.index(self.PAD)

    def load_data(self):
        contents = db.select_chat_sequence()
        question = []
        answer = []
        if contents:
            for sequence in contents:
                question.append([sequence[0]])
                answer.append([sequence[1]])
        return question, answer

    def tokenizer(self, sentence, flag=1):
        # flag = 1 : word token
        # flag = 2 : word & tag token
        mecab = Mecab()
        tokens = []
        if flag == 1:
            tokens = mecab.morphs(sentence)

        return tokens

    def max_sequence_len(self, input):
        max_seq_len = 0
        for seq in input:
            if len(seq) > max_seq_len:
                max_seq_len = len(seq)
        return max_seq_len


    def build_vocab(self, tonkens, is_target=False, max_vocab_size=None):
        word_counter = Counter()
        vocab = dict()
        reverse_vocab = dict()
        word_counter.update(tokens)

        if max_vocab_size is None:
            max_vocab_size = len(word_counter)

        if is_target:
            vocab[self.PAD] = self.pad_token
            vocab[self._GO] = self.start_token
            vocab[self.EOS] = self.end_token
            vocab_idx = 3
            for key, value in word_counter.most_common(max_vocab_size):
                vocab[key] = vocab_idx
                vocab_idx += 1
        else:
            vocab[self.PAD] = self.pad_token
            vocab_idx = 1
            for key, value in word_counter.most_common(max_vocab_size):
                vocab[key] = vocab_idx
                vocab_idx += 1

        for key, value in vocab.items():
            reverse_vocab[value] = key

        return vocab, reverse_vocab, max_vocab_size

    def token2idx(self, word, vocab):
        return vocab[word]

    def sent2idx(self, tokens, vocab, max_sentence_length, is_dec=False, is_target=False):
        current_length = len(tokens)
        pad_length = max_sentence_length - current_length
        if is_dec:
            return [self.start_token] + [self.token2idx(token, vocab) for token in tokens] + ([self.pad_token] * pad_length), current_length
        elif is_target:
            return [self.token2idx(token, vocab) for token in tokens] + ([self.pad_token] * pad_length) + [self.end_token]
        else:
            return [self.token2idx(token, vocab) for token in tokens] + ([self.pad_token] * pad_length), current_length

    def idx2token(self, idx, reverse_vocab):
        return reverse_vocab[idx]

    def idx2sent(self, indices, reverse_vocab):
        return " ".join([self.idx2token(idx, reverse_vocab) for idx in indices])

    def idx2sent_pad_removce(self, indices, reverse_vocab):
        in_index = np.where(indices == self.pad_token)
        return " ".join([self.idx2token(idx, reverse_vocab) for idx in indices[0:in_index[0][0]]])


datautil = DataUtil()
mecab = Mecab()
tokens = []
# 데이터 로드
question = "당뇨병이란"
answer = "당뇨병이란 혈액중의 포도당(혈당)이 높아서 소변으로 포도당이 넘쳐 나오는데서 지어진 이름입니다."
print("질문 :", question)
print("답변 :", answer)
print()
# 데이터 토큰화
question_tokens = mecab.morphs(question)
answer_tokens = mecab.morphs(answer)
print("질문 토큰화:", question)
print(" -> ", question_tokens)
print("답변 토큰화:", answer)
print(" -> ", answer_tokens)
print()
#데이터 최대길이 체크
q_max_sequence_len =  datautil.max_sequence_len(question)
a_max_sequence_len =  datautil.max_sequence_len(answer)

# 데이터 사전제작
q_vocab, q_reverse_vocab, q_max_vocab_size = datautil.build_vocab(question_tokens)
a_vocab, a_reverse_vocab, a_max_vocab_size = datautil.build_vocab(answer_tokens, is_target=True)
print("질문 사전:", q_vocab)
print("답변 사전:", a_vocab)
print()
q_sent2idx = datautil.sent2idx(tokens, q_vocab, q_max_sequence_len)
ta_sent2idx = datautil.sent2idx(tokens, a_vocab, a_max_vocab_size, is_target=True)
da_sent2idx = datautil.sent2idx(tokens, a_vocab, a_max_vocab_size, is_dec=True)

print()
print("질문 정수화:", question_tokens, " -> ", q_sent2idx)
print("학습용 답변 정수화:", answer_tokens, " -> ", ta_sent2idx)
print("테스트용 답변 정수화:", answer_tokens, " -> ", da_sent2idx)






