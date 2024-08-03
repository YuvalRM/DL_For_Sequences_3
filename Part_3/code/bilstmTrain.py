import copy
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
import sys
from Part_3.code.utils_part3 import *
import os

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


class Simple_Embedder(nn.Module):
    def __init__(self, input_size, embed_dim):
        super(Simple_Embedder, self).__init__()
        self.embed = nn.Embedding(input_size, embed_dim, padding_idx=0)

    def forward(self, x):
        input, suffixed, pref, chrs = x
        return self.embed(input)


class Char_Embedder(nn.Module):
    def __init__(self, input_size, embed_dim, hidden_layer_size):
        super(Char_Embedder, self).__init__()
        self.embed = nn.Embedding(input_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_layer_size, num_layers=1, batch_first=True, bidirectional=False)

    def forward(self, x):
        input, suffixed, pref, chrs = x
        chrs = self.embed(chrs)
        chrs = chrs.view(-1, 1, chrs.shape[-1])
        chrs, _ = self.lstm(chrs)
        chrs = chrs.view(input.shape[0], input.shape[1], -1)
        return chrs


class SuffixAndPrefixEmbedder(nn.Module):
    def __init__(self, input_size, prefix_size, suffix_size, embed_dim):
        super(SuffixAndPrefixEmbedder, self).__init__()
        self.input_embed = nn.Embedding(input_size, embed_dim, padding_idx=0)
        self.prefix_embed = nn.Embedding(prefix_size, embed_dim, padding_idx=0)
        self.suffix_embed = nn.Embedding(suffix_size, embed_dim, padding_idx=0)

    def forward(self, x):
        input, pref, suffixes, chrs = x
        batch_size, seq_len = input.size()
        input = self.input_embed(input).view(batch_size, -1)
        pref = self.prefix_embed(pref).view(batch_size, -1)
        suffixes = self.suffix_embed(suffixes).view(batch_size, -1)
        input = input + pref + suffixes

        return input.view(batch_size, seq_len, -1)


class SpecialEmbedder(nn.Module):
    def __init__(self, input_size, char_size, embed_dim, char_embed_dim, lstm_layer_dim):
        super(SpecialEmbedder, self).__init__()
        self.embed = Simple_Embedder(input_size, embed_dim)
        self.char_embed = Char_Embedder(char_size, embed_dim, char_embed_dim)
        self.LL = nn.Linear(embed_dim + char_embed_dim * MAX_WORD_LEN, lstm_layer_dim)

    def forward(self, x):
        input = self.embed(x)
        chars = self.char_embed(x)
        input = torch.concat((input, chars), dim=-1)

        return self.LL(input)


class LSTM_Model(nn.Module):
    def __init__(self, input_size, prefix_vocab_size, suffix_vocab_size, hidden_layer_size, output_size, embed_dim=50,
                 repr='a', num_chars=26,
                 char_embed_dim=50, special_dim=50):

        super(LSTM_Model, self).__init__()
        lstmdim = embed_dim
        if repr == 'a':
            self.embedding = Simple_Embedder(input_size, embed_dim)
        elif repr == 'b':
            self.embedding = Char_Embedder(num_chars, embed_dim, char_embed_dim)
            lstmdim = char_embed_dim * MAX_WORD_LEN
        elif repr == 'c':
            self.embedding = SuffixAndPrefixEmbedder(input_size, prefix_vocab_size, suffix_vocab_size, embed_dim)
        elif repr == 'd':
            self.embedding = SpecialEmbedder(input_size, num_chars, embed_dim, char_embed_dim, special_dim)
            lstmdim = special_dim
        else:
            raise NotImplementedError(f"Model is not implemented for task {repr}")
        #num layers=2 makes it run twice
        self.bilstm = nn.LSTM(lstmdim, hidden_layer_size // 2, num_layers=2, batch_first=True, bidirectional=True)
        self.LL = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.bilstm(x)
        x = x.contiguous().view(-1, x.shape[2])
        x = self.LL(x)
        #If I did softmax the results where worse...
        #x = torch.nn.functional.softmax(x, dim=1)
        return x


def check_accuracy(loader, model, task, tags):
    total = 0
    correct = 0
    running_loss = 0.0
    loss_f = nn.CrossEntropyLoss(ignore_index=-1).to(device)
    for batch_idx, batch in enumerate(loader):
        inputs, prefixes, suffixes, chars, labels, lengths = batch
        inputs = inputs.to(device)
        labels = labels.to(device)
        prefixes = prefixes.to(device)
        suffixes = suffixes.to(device)
        chars = chars.to(device)
        x = (inputs, prefixes, suffixes, chars)
        outputs = model(x)
        flat_labels = labels.view(-1)
        loss = loss_f(outputs, flat_labels)
        running_loss += loss.item()

        another_special = tags['O'] if task == 'NER' else -1

        mask = torch.logical_and(flat_labels != -1, flat_labels != another_special)
        _, outputs = torch.max(outputs, dim=-1)
        masked_pred = outputs[mask]
        masked_labels = flat_labels[mask]
        correct += (masked_pred == masked_labels).sum().item()
        total += mask.sum().item()
    #print(f"dev accuracy is: {correct / total}")
    #print(f"loss is: {running_loss / total}")
    return (correct / total)


def train(model, data_loader, dev_data_loader, optimizer, task, tags_i, model_file):
    model.to(device)
    loss_f = nn.CrossEntropyLoss(ignore_index=-1).to(device)
    model.train()
    sentences_done = 0
    accs = []
    for i in range(EPOCHS):
        print(f"start epoch {i}")
        running_loss = 0.0
        total = 0
        correct = 0
        torch.cuda.empty_cache()
        for batch_idx, batch in enumerate(data_loader):

            sentences_done += BATCH_SIZE
            if dev_data_loader:
                if sentences_done % 500 < BATCH_SIZE:
                    accs.append([check_accuracy(dev_data_loader, model, task, tags_i), sentences_done])
                if batch_idx == 0:
                    print(check_accuracy(dev_data_loader, model, task, tags_i))

            optimizer.zero_grad()
            inputs, prefixes, suffixes, chars, labels, lengths = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            prefixes = prefixes.to(device)
            suffixes = suffixes.to(device)
            chars = chars.to(device)
            x = (inputs, prefixes, suffixes, chars)
            outputs = model(x)
            flat_labels = labels.view(-1)
            loss = loss_f(outputs, flat_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            #mask = flat_labels != -1
            #_, outputs = torch.max(outputs, dim=-1)
            #masked_pred = outputs[mask]
            #masked_labels = flat_labels[mask]
            #correct += (masked_pred == masked_labels).sum().item()
            #total += mask.sum().item()
        torch.save(model.state_dict(), model_file)

        #print(f"train acc is: {correct / total}")
    return accs


def main(repr, train_file, model_file, dev_file, task):
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
    dev_data = None
    if dev_file != None:
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
        dev_data_set = TensorDataset(torch.LongTensor(dev_sentences), torch.LongTensor(dev_prefixes),
                                     torch.LongTensor(dev_suffixes),
                                     torch.LongTensor(dev_chars), torch.LongTensor(dev_tags),
                                     torch.LongTensor(dev_lengths))
        dev_data = DataLoader(dev_data_set, batch_size=BATCH_SIZE, shuffle=False)

    model = LSTM_Model(len(words_i.keys()), len(prefixes_i), len(suffixes_i), HIDDEN_LAYER, len(tags_i),
                       embed_dim=EMBED_DIM, repr=repr, num_chars=len(chars_i),
                       char_embed_dim=CHAR_EMBBED_DIM, special_dim=SPECIAL_DIM)

    if os.path.isfile(model_file):
        model.load_state_dict(torch.load(model_file))

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_data_set = TensorDataset(torch.LongTensor(sentences), torch.LongTensor(prefixes), torch.LongTensor(suffixes),
                                   torch.LongTensor(chars), torch.LongTensor(tags), torch.LongTensor(lengths))
    train_data = DataLoader(train_data_set, batch_size=BATCH_SIZE, shuffle=True)

    accs = train(model, train_data, dev_data, optimizer, task, tags_i, model_file)

    return accs


if __name__ == '__main__':
    args = sys.argv
    repr = args[1]
    train_file = args[2]
    model_file = args[3]
    if len(sys.argv) > 4:
        dev_file = args[4]
    else:
        dev_file = None
    if len(sys.argv) > 5:
        task = args[5]
    else:
        task = None

    assert repr in ['a', 'b', 'c', 'd']

    main(repr, train_file, model_file, dev_file, task)
