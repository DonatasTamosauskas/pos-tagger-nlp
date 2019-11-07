# python3.5 runtagger.py <test_file_absolute_path> <model_file_absolute_path> <output_file_absolute_path>

import os
import math
import datetime
import sys
from random import uniform
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


def tag_sentence(test_file, model_file, out_file):
    start = datetime.datetime.now()

    model, dataset = load_model(model_file)
    dataset = prepare_dataset_infer(dataset, test_file)
    predictions = generate_results(model, dataset, num_workers=32)
    export_preds(predictions, out_file)
    
    print('Finished in: ', start - datetime.datetime.now())


class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, to_lower=True, training=True, make_unknown=None, letter_emb_len=15, too_long_split=10):
        self.to_lower = to_lower
        self.training = training
        self.make_unknown = make_unknown
        self.letter_emb_len = letter_emb_len
        self.too_long_split = too_long_split

        self.sentences = []
        self.vocab = []
        self.tags = []

        self.generate_dataset(path)
        self.UNKONWN_WORD = len(self.vocab) + 1

        self.LETTER_MIN = 32
        self.LETTER_MAX = 126
        self.letter_len = self.LETTER_MAX - self.LETTER_MIN + 2
        self.UNKNOWN_LETTER = self.letter_len


    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, index):
        sentence_embs, tag_embs = self.transform_sentence(self.sentences[index])
        return sentence_embs, tag_embs
    
    def generate_dataset(self, path):
        with open(path, 'r') as input_file:
            self.sentences = input_file.read().split("\n")
            
            if len(self.vocab) == 0:
                self.create_vocabs(self.sentences)
                self.vocab_size = len(self.vocab) + 2 # Unkown words, padding
                self.tag_size = len(self.tags) + 1 # Padding
            
            if self.sentences[-1] == "":
                self.sentences.pop()
    
    def create_vocabs(self, sentences):
        vocab_set = set()
        tag_set = set()

        for sentence in sentences:
            for word in sentence.split(" "):
                try:
                    word, tag = self.split_words_tag(word)
                    vocab_set.add(word.lower() if self.to_lower else word)
                    tag_set.add(tag)
                except RuntimeError:
                    print("Not a valid word/tag pair: " + word)

        self.vocab = list(vocab_set)
        self.tags = list(tag_set)
            
    def transform_sentence(self, sentence):
        numeric_sent, letter_embs, tags = [], [], []

        for word_tag in sentence.split(" "):
            try:
                if self.training:
                    word, tag = self.split_words_tag(word_tag)
                    tag_id = self.tags.index(tag) + 1
                    word_id = self.get_word_id(word)                    
                else:
                    word = word_tag
                    word_id = self.vocab.index(word.lower() if self.to_lower else word) + 1

            except RuntimeError:
                print("Not a valid word/tag pair: " + word_tag)
            except ValueError:
                # The id of an unknown word
                word_id = self.UNKONWN_WORD

            numeric_sent.append(word_id)
            letter_embs.append(self.transform_word(word))
            if self.training: tags.append(tag_id)

        return (torch.tensor(numeric_sent), torch.cat(letter_embs).view(-1, self.letter_emb_len)), torch.tensor(tags) if self.training else []
    
    def get_word_id(self, word):
        # Replacing a random subset of words with the unkown word id
        if self.make_unknown is not None:
            if uniform(0, 1) < self.make_unknown:
                return self.UNKONWN_WORD
            else:
                return self.vocab.index(word.lower() if self.to_lower else word) + 1
        else:
            return self.vocab.index(word.lower() if self.to_lower else word) + 1
        
    def transform_word(self, word):
        # 32 -> 126 range
        word_len = len(word)
        letter_embs = torch.zeros(self.letter_emb_len, dtype=torch.int64)
        
        if word_len <= self.letter_emb_len:
            for i, letter in enumerate(word):
                emb = ord(letter)
                letter_embs[i] = emb - self.LETTER_MIN + 1 if(emb >= self.LETTER_MIN <= self.LETTER_MAX) else self.UNKNOWN_LETTER
        else:
            #Word is too long for embedding
            for i in range(self.letter_emb_len):
                if i <= self.too_long_split:
                    letter_embs[i] = ord(word[i]) - self.LETTER_MIN
                else:
                    letter_embs[i] = ord(word[-(self.letter_emb_len - i)]) - self.LETTER_MIN
                    
        return letter_embs

    @staticmethod
    def split_words_tag(word):
        words_tag = word.split("/")
        
        if len(words_tag) < 2: 
            raise RuntimeError("Not a valid word/tag pair:" + word)
            
        tag = words_tag.pop()
        word = "/".join(words_tag)
        
        return word, tag
    
    def print_sentence(self, sentence):
        print(" ".join([self.vocab[word.item()] for word in sentence.view(-1)]))
                
    def decode_sentence(self, sentence):
        print(sentence)
        answ = [self.vocab[word.item() - 1] for word in sentence.view(-1)]
        return answ
    
    def decode_tags(self, tag_ids):
        return [self.tags[tag.item() - 1] for tag in tag_ids.view(-1)]
    
    def __getstate__(self):
        d = dict(self.__dict__)
        del d['sentences']
        return d
    
    def __setstate(self, d):
        self.__dict__.update(d)
        self.__dict__.update({'sentences': []})


class LstmCnnTagger(nn.Module):
    def __init__(self, word_emb_dim, letter_emb_dim, hidden_dim, word_vocab_size, letter_vocab_size, letter_word_size, tagset_size):
        super(LstmCnnTagger, self).__init__()

        self.hidden_dim = hidden_dim
        self.letter_word_size = letter_word_size # 15
        self.letter_emb_dim = letter_emb_dim
        self.word_emb_dim = word_emb_dim

        self.word_embeddings = nn.Embedding(word_vocab_size, word_emb_dim)
        self.letter_embeddings = nn.Embedding(letter_vocab_size, letter_emb_dim)

        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.max_pool = nn.MaxPool2d(2,2)
        self.cnn2 = nn.Conv2d(in_channels=32, out_channels=word_emb_dim, kernel_size=3)
        self.max_pool_last = nn.MaxPool2d(2, 10)

        self.lstm = nn.LSTM(word_emb_dim * 2, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        words, letters = sentence
        batches = words.shape[0]

        word_embeds = self.word_embeddings(words)
        letter_embeds = self.letter_embeddings(letters.view(-1, self.letter_word_size))

        cnn_feat = self.cnn1(letter_embeds.view(-1, 1, self.letter_word_size, self.letter_emb_dim))
        cnn_feat = F.relu(self.max_pool(cnn_feat))
        cnn_feat = self.cnn2(cnn_feat)
        cnn_feat = F.relu(self.max_pool_last(cnn_feat))

        concat = torch.cat([
            cnn_feat.view(len(words[-1]), batches, self.word_emb_dim), 
            word_embeds.view(len(words[-1]), batches, -1)
        ], 2)

        lstm_out, _ = self.lstm(concat)
        tag_space = self.hidden2tag(lstm_out.view(len(words[-1]) * batches, -1))
        tag_scores = F.log_softmax(tag_space, dim=1)

        return tag_scores


def pad_seq(sequences):
    x_word, x_let, y = [], [], []
    
    for (word_embs, letter_embs), targets in sequences:
        x_word.append(word_embs)
        x_let.append(letter_embs)
        y.append(targets)
        
    return (pad_sequence(x_word, batch_first=True), pad_sequence(x_let, batch_first=True)) , pad_sequence(y, batch_first=True)


def load_model(model_filename):
    model = torch.load(model_filename + "_1.data")
    dictionaries = torch.load(model_filename + "_2.data")
    return model, dictionaries


def prepare_dataset_infer(dataset, input_filename):
    dataset.generate_dataset(input_filename)
    dataset.training = False
    return dataset


def generate_results(model, dataset, num_workers=0):
    cpu = torch.device("cpu")
    model.to(cpu)
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    preds = []
    for i, (x, _) in enumerate(dataloader):
        if i % 400 == 0: 
            print(i, " | ", len(dataloader.dataset))
            
        predictions = model(x)
        _, pos_tag_ids = predictions.max(1)

        words = dataset.sentences[i].split(" ")
        tags = dataset.decode_tags(pos_tag_ids)
        word_tags = ["/".join(word_tag) for word_tag in zip(words, tags)]

        preds.append(" ".join(word_tags))
    
    return preds
        
def export_preds(preds, filename):
    with open(filename, "w") as out_file:
        out_file.write("\n".join(preds))


if __name__ == "__main__":
    # make no changes here
    test_file = sys.argv[1]
    model_file = sys.argv[2]
    out_file = sys.argv[3]
    tag_sentence(test_file, model_file, out_file)
