import re
from collections import Counter
import algorithm4.db as db


class DataUtil:

    def __init__(self):
        _GO = '_GO'
        EOS = '_EOS'  # also function as PAD
        UNK = '_UNK'

        extra_tokens = [_GO, EOS, UNK]

        self.start_token = extra_tokens.index(_GO)  # start_token = 0
        self.end_token = extra_tokens.index(EOS)  # end_token = 1
        self.unk_token = extra_tokens.index(UNK)

        # Batch_size: 2
        input_batches, target_batches = self.load_data()

        all_input_sentences = []
        for input_batch in input_batches:
            all_input_sentences.extend(input_batch)

        all_target_sentences = []
        for target_batch in target_batches:
            all_target_sentences.extend(target_batch)

        self.enc_vocab, self.enc_reverse_vocab, self.enc_vocab_size, self.enc_sentence_length = self.build_vocab(all_input_sentences)
        self.dec_vocab, self.dec_reverse_vocab, self.dec_vocab_size,  self.dec_sentence_length = self.build_vocab(all_target_sentences, is_target=True)

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

    def max_sequence_len(self, input):
        max_seq_len = 0
        for seq in input:
            if len(seq) > max_seq_len:
                max_seq_len = len(seq)
        return max_seq_len

    def tokenizer(self, sentence):
        tokens = re.findall(r"[\w]+|[^\s\w]", sentence)
        return tokens

    def build_vocab(self, sentences, is_target=False, max_vocab_size=None, max_sequence_len=None):
        word_counter = Counter()
        vocab = dict()
        reverse_vocab = dict()

        for sentence in sentences:
            tokens = self.tokenizer(sentence)
            word_counter.update(tokens)

        if max_vocab_size is None:
            max_vocab_size = len(word_counter)

        if max_sequence_len is None:
            max_sequence_len = self.max_sequence_len(sentences)

        if is_target:
            vocab['_PAD'] = 0
            vocab['_GO'] = 1
            vocab['_EOS'] = 2
            vocab_idx = 3
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

        return vocab, reverse_vocab, max_vocab_size, max_sequence_len

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

    def idx2token(self, idx, reverse_vocab):
        return reverse_vocab[idx]

    def idx2sent(self, indices, reverse_vocab):
        return " ".join([self.idx2token(idx, reverse_vocab) for idx in indices])

