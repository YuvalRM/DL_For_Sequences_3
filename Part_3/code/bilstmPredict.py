import copy
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
import sys
from Part_3.code.utils_part3 import *
import os
from Part_3.code.bilstmTrain import LSTM_Model
"""HYPER PARAMETERS"""

EPOCHS = 5
BATCH_SIZE = 64
EMBED_DIM = 50
RARETY_THRESHOLD = 1
UNKNOWN_TOKEN = 'UUUNKKK'
PAD_TOKEN = 'PPPADDD'
HIDDEN_LAYER = 50
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LR = 5e-3
MAX_WORD_LEN = 70
CHAR_EMBBED_DIM = 10
SPECIAL_DIM = 50
"""END OF HYPER PARAMETERS"""




def main(repr,model_file,input_file):
    """TRAIN DATA SET"""

    sentences, prefixes, suffixes, tags, max_len_sentence = read_data(train_file, PAD_TOKEN)

    chars_i, i_char = chars_to_indexes(sentences, unknown_token=UNKNOWN_TOKEN, pad_token=PAD_TOKEN)

    chars = convert_chars_to_indexes(copy.deepcopy(sentences), chars_i, unknown_token=UNKNOWN_TOKEN)

    sentences = rare_to_unknown(sentences, unknown_token=UNKNOWN_TOKEN, threshold=RARETY_THRESHOLD)
    prefixes = rare_to_unknown(prefixes, unknown_token=UNKNOWN_TOKEN, threshold=RARETY_THRESHOLD)
    suffixes = rare_to_unknown(suffixes, unknown_token=UNKNOWN_TOKEN, threshold=RARETY_THRESHOLD)

    words_i, i_words = words_to_indexes(sentences)
    prefixes_i, i_prefixes = words_to_indexes(prefixes, unknown_token=UNKNOWN_TOKEN, pad_token=PAD_TOKEN)
    suffixes_i, i_suffixes = words_to_indexes(suffixes, unknown_token=UNKNOWN_TOKEN, pad_token=PAD_TOKEN)
    tags_i, i_tags = words_to_indexes(tags, unknown_token=UNKNOWN_TOKEN, pad_token=PAD_TOKEN, tags=True)

    sentences = convert_data_to_indexes(sentences, words_i, unknown_token=UNKNOWN_TOKEN)
    prefixes = convert_data_to_indexes(prefixes, prefixes_i, unknown_token=UNKNOWN_TOKEN)
    suffixes = convert_data_to_indexes(suffixes, suffixes_i, unknown_token=UNKNOWN_TOKEN)
    tags = convert_data_to_indexes(tags, tags_i, unknown_token=UNKNOWN_TOKEN)

    sentences, lengths = pad_data(sentences, max_len_sentence)
    prefixes, _ = pad_data(prefixes, max_len_sentence)
    suffixes, _ = pad_data(suffixes, max_len_sentence)
    tags, _ = pad_data(tags, max_len_sentence, tags=True)
    chars = pad_chars(chars, max_len_sentence, MAX_WORD_LEN)

    """DEV DATA SET"""
    dev_sentences, dev_prefixes, dev_suffixes, dev_tags, dev_max_len_sentence = read_data(dev_file, PAD_TOKEN)

    dev_chars = convert_chars_to_indexes(copy.deepcopy(dev_sentences), chars_i, unknown_token=UNKNOWN_TOKEN)

    dev_sentences = convert_data_to_indexes(dev_sentences, words_i, unknown_token=UNKNOWN_TOKEN)
    dev_prefixes = convert_data_to_indexes(dev_prefixes, prefixes_i, unknown_token=UNKNOWN_TOKEN)
    dev_suffixes = convert_data_to_indexes(dev_suffixes, suffixes_i, unknown_token=UNKNOWN_TOKEN)
    dev_tags = convert_data_to_indexes(dev_tags, tags_i, unknown_token=UNKNOWN_TOKEN)

    dev_sentences, dev_lengths = pad_data(dev_sentences, max_len_sentence)
    dev_prefixes, _ = pad_data(dev_prefixes, max_len_sentence)
    dev_suffixes, _ = pad_data(dev_suffixes, max_len_sentence)
    dev_tags, _ = pad_data(dev_tags, max_len_sentence, tags=True)
    dev_chars = pad_chars(dev_chars, max_len_sentence, MAX_WORD_LEN)

    model = LSTM_Model(len(words_i.keys()), len(prefixes_i), len(suffixes_i), HIDDEN_LAYER, len(tags_i),
                       embed_dim=EMBED_DIM, repr=repr, num_chars=len(chars_i),
                       char_embed_dim=CHAR_EMBBED_DIM, special_dim=SPECIAL_DIM)

    if os.path.isfile(model_file):
        model.load_state_dict(torch.load(model_file))



    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_data_set = TensorDataset(torch.LongTensor(sentences), torch.LongTensor(prefixes), torch.LongTensor(suffixes),
                                   torch.LongTensor(chars), torch.LongTensor(tags), torch.LongTensor(lengths))
    train_data = DataLoader(train_data_set, batch_size=BATCH_SIZE, shuffle=True)

    dev_data_set = TensorDataset(torch.LongTensor(dev_sentences), torch.LongTensor(dev_prefixes),
                                 torch.LongTensor(dev_suffixes),
                                 torch.LongTensor(dev_chars), torch.LongTensor(dev_tags), torch.LongTensor(dev_lengths))
    dev_data = DataLoader(dev_data_set, batch_size=BATCH_SIZE, shuffle=False)

    accs = train(model, train_data, dev_data, optimizer, task, tags_i,model_file)


    return  accs


if __name__ == '__main__':
    args = sys.argv
    repr = args[1]
    model_file = args[2]
    input_file = args[3]

    assert repr in ['a', 'b', 'c', 'd']

    main(repr, model_file, input_file)