import os
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from fastprogress import progress_bar, master_bar
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

KAGGLE = False

training_data = Path("../input/sents.train") if KAGGLE else Path("../data/sents.train")

import torch
import torch.utils.data
from random import uniform


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
#                 print("Word not in the vocab: " + word_tag)
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

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

def pad_seq(sequences):
    x_word, x_let, y = [], [], []
    
    for (word_embs, letter_embs), targets in sequences:
        x_word.append(word_embs)
        x_let.append(letter_embs)
        y.append(targets)
        
    return (pad_sequence(x_word, batch_first=True), pad_sequence(x_let, batch_first=True)) , pad_sequence(y, batch_first=True)


class PipelineTestModel(nn.Module):
    def __init__(self, vocab_size, emb_dims, output_dims):
        super(PipelineTestModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.emb_dims = emb_dims
        self.output_dims = output_dims
        
        self.emb = nn.Embedding(self.vocab_size, self.emb_dims)
        self.fc = nn.LSTM(self.emb_dims, self.output_dims)
        
    def forward(self, sentence):
        print(sentence.shape)
        emb = self.emb(sentence)
        print(emb.shape)
#         tags = F.softmax(self.fc(emb), dim=self.output_dims)
        tags = self.fc(emb.view(len(sentence), 1, -1))
        print(tags.shape)
        return tags    
    
    
class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
    
    def forward(self, sentence):
        batches = sentence.shape[0]
        embeds = self.word_embeddings(sentence)
#         print(embeds.shape)
#         print(embeds.view(len(sentence[-1]), batches, -1).shape)
        lstm_out, _ = self.lstm(embeds.view(len(sentence[-1]), batches, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence[-1]) * batches, -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores
    
class LstmCnnTagger(nn.Module):

    def __init__(self, word_emb_dim, letter_emb_dim, hidden_dim, word_vocab_size, letter_vocab_size, letter_word_size, tagset_size):
        super(LstmCnnTagger, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.letter_word_size = letter_word_size # 15
        self.letter_emb_dim = letter_emb_dim
        self.word_emb_dim = word_emb_dim

        self.word_embeddings = nn.Embedding(word_vocab_size, word_emb_dim)
        self.letter_embeddings = nn.Embedding(letter_vocab_size, letter_emb_dim)
        
        # Should I add more hidden layers?
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.max_pool = nn.MaxPool2d(2,2)
        self.cnn2 = nn.Conv2d(in_channels=32, out_channels=word_emb_dim, kernel_size=3)
        self.max_pool_last = nn.MaxPool2d(2, 10)

        self.lstm = nn.LSTM(word_emb_dim * 2, hidden_dim)
#         self.lstm = nn.LSTM(word_emb_dim, hidden_dim)
        
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
    
    def forward(self, sentence):
        words, letters = sentence
        batches = words.shape[0]
        
        word_embeds = self.word_embeddings(words)
        letter_embeds = self.letter_embeddings(letters.view(-1, self.letter_word_size))
#         print(letter_embeds.shape)
#         print(letter_embeds.view(batches, len(words[-1]), self.letter_word_size, self.letter_emb_dim).shape)
#         print(word_embeds.shape)
        
        cnn_feat = self.cnn1(letter_embeds.view(-1, 1, self.letter_word_size, self.letter_emb_dim))
        cnn_feat = F.relu(self.max_pool(cnn_feat))
        cnn_feat = self.cnn2(cnn_feat)
        cnn_feat = F.relu(self.max_pool_last(cnn_feat))
        
#         print(cnn_feat.shape)
        
        concat = torch.cat([
            cnn_feat.view(len(words[-1]), batches, self.word_emb_dim), 
            word_embeds.view(len(words[-1]), batches, -1)
        ], 2)
        
#         print(concat.shape)
#         print(word_embeds.view(len(words[-1]), batches, -1).shape)
        
#         print(concat.shape)
        
        # Add ReLU?
        lstm_out, _ = self.lstm(concat)
        tag_space = self.hidden2tag(lstm_out.view(len(words[-1]) * batches, -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        
        return tag_scores


def train_model(model, data_loader, epochs=1, lr=0.01, patience=10, lr_decrease=2):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Device: {}". format(device))

    loss_func = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    tagset_size = data_loader.dataset.tag_size
    losses = []
    i = 0
    
    last_min_loss = 10
    steps_after_loss = 0

    model.to(device)
    model.zero_grad()

    master = master_bar(range(epochs))
    for epoch in master:
        for (x1, x2), y in progress_bar(data_loader, parent=master):

            model.zero_grad()
            x, y = (x1.to(device), x2.to(device)), y.to(device)

            pred = model(x)
            loss = loss_func(pred.view(-1, tagset_size), y.view(-1))

            loss.backward()
            optimizer.step()
            
            if loss.item() < last_min_loss:
                last_min_loss = loss.item()
                steps_after_loss = 0
            elif steps_after_loss > patience:
                steps_after_loss = 0
                lr = lr / lr_decrease
                optimizer = optim.Adam(model.parameters(), lr=lr)
                print("No decrease in loss for " + str(patience) + " steps. Decrementing lr: " + str(lr))
            else:
                steps_after_loss += 1

            if i % 10 == 0: 
                losses.append(loss.item())
                print("Loss: " + str(loss.item()))
            i += 1
            
    plt.plot(losses)
    print("Min loss: " + str(min(losses)))


def export_model(model, dataset, train_time):
    model_name = str(type(model)).split(".")[-1][:-2]
    model_save_name = model_name + "_" + str(train_time)
    
    torch.save(model, model_save_name + "_1.data")
    torch.save(dataset, model_save_name + "_2.data")


from torch.utils.data import DataLoader

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
            print("{} | {}".format(i, len(dataloader.dataset)))
            
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








dataset = Dataset(
    training_data,
    make_unknown=0.01,
    letter_emb_len=15,
    too_long_split=10
)

num_workers = 48

pos_dataloader = DataLoader(
    dataset, 
    batch_size=1, 
    num_workers=num_workers
)

pos_dataloader_batched = DataLoader(
    dataset,
    batch_size=128,
    collate_fn=pad_seq,
    num_workers=num_workers
)

model_cnn = LstmCnnTagger(
    word_emb_dim=512, 
    letter_emb_dim=30, 
    hidden_dim=512, 
    word_vocab_size=dataset.vocab_size, 
    letter_vocab_size=dataset.letter_len, 
    letter_word_size=dataset.letter_emb_len, 
    tagset_size=dataset.tag_size
)

train_model(model_cnn, pos_dataloader_batched, epochs=1, lr=0.01, patience=70)
#train_model(model_cnn, pos_dataloader_batched, epochs=3, lr=0.001, patience=100)
#export_model(model_cnn, dataset, 5)

import copy
test_data = Path("../data/sents.test")

test_dataset = prepare_dataset_infer(copy.deepcopy(dataset), test_data)
predictions = generate_results(model_cnn, test_dataset, num_workers=48)
export_preds(predictions, "test_output.txt")
#!python3 ../data/eval.py test_output.txt ../data/sents.answer

#Accuracy= 0.8980328763672244 - 2 epochs, lr: 0.01, unk: 0.01->0.50
#Accuracy= 0.9002792181890706 - 2 epochs, lr: 0.005, unk: 0.05
#Accuracy= 0.9389288938341066 - 3 epochs, lr 0.01x2 -> 0.001, patience: 70 -> 100