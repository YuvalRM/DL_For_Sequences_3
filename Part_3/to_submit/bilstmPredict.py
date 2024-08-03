import copy

import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
import sys
from Part_3.to_submit.utils_part3 import *
import os
from Part_3.to_submit.bilstmTrain import LSTM_Model
import json

"""HYPER PARAMETERS"""

BATCH_SIZE = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_WORD_LEN = 70
"""END OF HYPER PARAMETERS"""


def pred(model, data, i_tags, output_file, i_words,pad_token,orig_sentences):
    model.to(device)
    model.eval()
    with open(output_file, 'w') as f:
        for indx,batch in enumerate(data):
            inputs, prefixes, suffixes, chars, labels, lengths = batch
            inputs = inputs.to(device)
            prefixes = prefixes.to(device)
            suffixes = suffixes.to(device)
            chars = chars.to(device)
            x = (inputs, prefixes, suffixes, chars)
            output = model(x)
            for i, out in enumerate(output):
                if inputs[0][i] == pad_token:
                    break
                f.write(orig_sentences[indx][i] + '\t' + i_tags[str(torch.argmax(out).item())] + '\n')
            f.write('\n')


def main(repr, model_file, input_file, output_file):
    """TRAIN DATA SET"""

    with open(model_file + ".json", 'r') as f:
        model_params = json.load(f)
    words_i = model_params['WORDS_2_i']
    prefixes_i = model_params['PREFIXES_2_i']
    suffixes_i = model_params['SUFFIXES_2_i']
    HIDDEN_LAYER = model_params['HIDDEN_LAYER']
    tags_i = model_params['TAGS_2_i']
    i_tags = model_params['i_2_TAGS']
    EMBED_DIM = model_params['EMBED_DIM']
    chars_i = model_params['CHARS_2_i']
    CHAR_EMBBED_DIM = model_params['CHAR_EMBBED_DIM']
    SPECIAL_DIM = model_params['SPECIAL_DIM']
    UNKNOWN_TOKEN = model_params['UNKNOWN_TOKEN']
    PAD_TOKEN = model_params['PAD_TOKEN']

    sentences, prefixes, suffixes, tags, max_len_sentence,orig_sentences = read_data_for_pred(input_file, PAD_TOKEN)
    chars = convert_chars_to_indexes(copy.deepcopy(sentences), chars_i, unknown_token=UNKNOWN_TOKEN)

    sentences = convert_data_to_indexes(sentences, words_i, unknown_token=UNKNOWN_TOKEN)
    prefixes = convert_data_to_indexes(prefixes, prefixes_i, unknown_token=UNKNOWN_TOKEN)
    suffixes = convert_data_to_indexes(suffixes, suffixes_i, unknown_token=UNKNOWN_TOKEN)

    sentences, lengths = pad_data(sentences, max_len_sentence)
    prefixes, _ = pad_data(prefixes, max_len_sentence)
    suffixes, _ = pad_data(suffixes, max_len_sentence)
    tags, _ = pad_data(tags, max_len_sentence, tags=True)
    chars = pad_chars(chars, max_len_sentence, MAX_WORD_LEN)

    model = LSTM_Model(len(words_i), len(prefixes_i), len(suffixes_i), HIDDEN_LAYER, len(tags_i),
                       embed_dim=EMBED_DIM, repr=repr, num_chars=len(chars_i),
                       char_embed_dim=CHAR_EMBBED_DIM, special_dim=SPECIAL_DIM)

    model.load_state_dict(torch.load(model_file))

    pred_data_set = TensorDataset(torch.LongTensor(sentences), torch.LongTensor(prefixes), torch.LongTensor(suffixes),
                                  torch.LongTensor(chars), torch.LongTensor(tags), torch.LongTensor(lengths))
    pred_data = DataLoader(pred_data_set, batch_size=BATCH_SIZE, shuffle=False)

    i_words = {i: word for word, i in words_i.items()}
    pred(model, pred_data, i_tags, output_file, i_words,words_i[PAD_TOKEN],orig_sentences)



if __name__ == '__main__':
    args = sys.argv
    repr = args[1]
    model_file = args[2]
    input_file = args[3]
    output_file = args[4]

    assert repr in ['a', 'b', 'c', 'd']

    main(repr, model_file, input_file, output_file)
