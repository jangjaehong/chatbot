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
        self.extra_tokens = [self.PAD, self._GO, self.EOS]

        self.start_token = self.extra_tokens.index(self._GO)  # start_token = 0
        self.end_token = self.extra_tokens.index(self.EOS)  # end_token = 1
        self.pad_token = self.extra_tokens.index(self.PAD)

        # Batch_size: 2
        self.input_batches, self.target_batches = self.load_data()

        all_input_sentences = []
        for input_batch in self.input_batches:
            all_input_sentences.extend(input_batch)

        all_target_sentences = []
        for target_batch in self.target_batches:
            all_target_sentences.extend(target_batch)

        self.enc_vocab, self.enc_reverse_vocab, self.enc_vocab_size, self.enc_sentence_length = self.build_vocab(all_input_sentences)
        self.dec_vocab, self.dec_reverse_vocab, self.dec_vocab_size,  self.dec_sentence_length = self.build_vocab(all_target_sentences, is_target=True)
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


    def build_vocab(self, sentences, is_target=False, max_vocab_size=None, max_sequence_len=None):
        word_counter = Counter()
        vocab = dict()
        reverse_vocab = dict()

        for sentence in sentences:
            tokens = self.tokenizer(sentence)
            #print("토큰화 대상: ", sentences, ' | 결과:', tokens)
            word_counter.update(tokens)

        if max_vocab_size is None:
            max_vocab_size = len(word_counter)

        if max_sequence_len is None:
            max_sequence_len = self.max_sequence_len(sentences)

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

        return vocab, reverse_vocab, max_vocab_size, max_sequence_len

    def token2idx(self, word, vocab):
        return vocab[word]

    def sent2idx(self, sent, vocab, max_sentence_length, is_dec=False, is_target=False):
        tokens = self.tokenizer(sent)
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
