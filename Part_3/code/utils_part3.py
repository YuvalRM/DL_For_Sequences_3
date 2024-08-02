import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def read_data(file_name, pad_token):
    sentences, tags, prefixes, suffixes = [], [], [], []
    max_len = 0

    with open(file_name, "r", encoding="utf-8") as file:
        data = file.readlines()
        sentence, sentence_tags, sentence_prefixes, sentence_suffixes = [], [], [], []

        # For each line in the file.
        for line in data:

            # As long as the sentence isn't over.
            if line != '\n':

                word, tag = line.strip().split()
                word = word.lower()
                sentence.append(word)
                sentence_tags.append(tag)
                # For each word save its prefix and suffix.
                sentence_prefixes.append(word[:3])
                sentence_suffixes.append(word[-3:])

            else:  # EOS
                max_len = max(max_len, len(sentence))
                sentences.append(sentence)
                prefixes.append(sentence_prefixes)
                suffixes.append(sentence_suffixes)
                tags.append(sentence_tags)
                sentence, sentence_tags, sentence_prefixes, sentence_suffixes = [], [], [], []

    return sentences, prefixes, suffixes, tags, max_len


def rare_to_unknown(data, threshold=1, unknown_token='UUUNKKK'):
    words_count = {}
    count = 0
    how_many = 0
    for sentence in data:
        for word in sentence:
            count += 1
            if word not in words_count.keys():
                words_count[word] = 1
            else:
                words_count[word] += 1

    for sentence in data:
        for i in range(len(sentence)):
            if words_count[sentence[i]] <= threshold:
                sentence[i] = unknown_token
                how_many += 1

    return data


def words_to_indexes(data, unknown_token='UUUNKKK', pad_token='PPPADDD', tags=False):
    vocab = set()
    for sentence in data:
        for word in sentence:
            vocab.add(word)
    vocab.discard(unknown_token)
    vocab.discard(pad_token)
    vocab = sorted(vocab)
    if not tags:
        vocab = [pad_token, unknown_token] + vocab
    word_index = {word: i for i, word in enumerate(vocab)}
    index_word = {i: word for i, word in enumerate(vocab)}
    return word_index, index_word


def chars_to_indexes(data, unknown_token='UUUNKKK', pad_token='PPPADDD'):
    vocab = set()
    for sentence in data:
        for word in sentence:
            for char in word:
                vocab.add(char)

    vocab = sorted(vocab)
    vocab = [pad_token, unknown_token] + vocab
    ch_index = {ch: i for i, ch in enumerate(vocab)}
    index_ch = {i: ch for i, ch in enumerate(vocab)}
    return ch_index, index_ch


def convert_data_to_indexes(data, word_2_i, unknown_token='UUUNKKK'):
    n_data = []

    for sentence in data:
        n_sentence = []
        for word in sentence:
            if word in word_2_i.keys():
                n_sentence.append(word_2_i[word])
            else:
                n_sentence.append(word_2_i[unknown_token])

        n_data.append(n_sentence)
    return n_data


def convert_chars_to_indexes(data, char_2_i, unknown_token='UUUNKKK'):
    n_data = []

    for sentence in data:
        n_sentence = []
        for word in sentence:
            n_word = []
            for ch in word:
                if ch in char_2_i.keys():
                    n_word.append(char_2_i[ch])
                else:
                    n_word.append(char_2_i[unknown_token])
            n_sentence.append(n_word)
        n_data.append(n_sentence)
    return n_data


def pad_data(sentences, max_len, tags=False):
    padded = np.zeros((len(sentences), max_len), dtype=int)
    if tags:
        padded -= 1
    lengths = []
    for i, sentence in enumerate(sentences):
        padded[i, :len(sentence)] = sentence
        lengths.append(len(sentence))
    return padded, lengths


def pad_chars(data, max_len, max_word):
    padded = np.zeros((len(data), max_len, max_word), dtype=int)
    for i, sentence in enumerate(data):
        for j, word in enumerate(sentence):
            padded[i, j, :len(word)] = word
    return padded
