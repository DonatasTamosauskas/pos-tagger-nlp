#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import math
import sys
import torch

def tag_sentence(test_file, model_file, out_file):
    # write your code here. You can add functions as well.
    # use torch library to load model_file
    print('Finished...')

if __name__ == "__main__":
    # make no changes here
#     test_file = sys.argv[1]
#     model_file = sys.argv[2]
#     out_file = sys.argv[3]
#     tag_sentence(test_file, model_file, out_file)
    pass


# # Loading model & dictionary

# In[2]:


def load_model(model_filename):
    model = torch.load(model_filename + "_1.data")
    dictionaries = torch.load(model_filename + "_2.data")
    return model, dictionaries


# In[3]:


import torch.nn as nn
import torch.nn.functional as F

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
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence[-1]), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence[-1]), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


# In[4]:


import torch
import torch.utils.data


class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, to_lower=True, training=True):
        self.to_lower = to_lower
        self.training = training
        
        self.sentences = []
        self.vocab = []
        self.tags = []
        
        self.generate_dataset(path)
    
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
                self.vocab_size = len(self.vocab)
                self.tag_size = len(self.tags)
            
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
        numeric_sent = []
        tags = []

        for word_tag in sentence.split(" "):
            try:
                if self.training:
                    word, tag = self.split_words_tag(word_tag)
                    tag_id = self.tags.index(tag)
                else:
                    word = word_tag
                    
                word_id = self.vocab.index(word.lower() if self.to_lower else word)

            except RuntimeError:
                print("Not a valid word/tag pair: " + word_tag)
            except ValueError:
                print("Word not in the vocab: " + word_tag)
                # The id of an unknown word
                word_id = len(self.vocab) - 1

            numeric_sent.append(word_id)
            if self.training: tags.append(tag_id)

        return torch.tensor(numeric_sent), torch.tensor(tags) if self.training else []

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
        return [self.vocab[word.item()] for word in sentence.view(-1)]
    
    def decode_tags(self, tag_ids):
        return [self.tags[tag.item()] for tag in tag_ids.view(-1)]
    
    def __getstate__(self):
        d = dict(self.__dict__)
        del d['sentences']
        return d
    
    def __setstate(self, d):
        self.__dict__.update(d)
        self.__dict__.update({'sentences': []})


# In[5]:


model, dataset = load_model("LSTMTagger_5")


# # Digesting the new input data

# In[6]:


from pathlib import Path

test_data = Path("../data/sents.test")
dataset.generate_dataset(test_data)
dataset.training = False


# In[7]:


get_ipython().run_cell_magic('time', '', 'from torch.utils.data import DataLoader\n\ngpu = torch.device("cuda")\ncpu = torch.device("cpu")\nmodel.to(cpu)\ndataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)\n\ni = 0\n\npreds = []\nfor x, _ in dataloader:\n#     if i > 100:\n#         break\n#     i += 1\n    predictions = model(x.to(cpu))\n    _, pos_tag_ids = predictions.max(1)\n    \n    words = dataset.decode_sentence(x)\n    tags = dataset.decode_tags(pos_tag_ids)\n    word_tags = ["/".join(word_tag) for word_tag in zip(words, tags)]\n    \n    preds.append(" ".join(word_tags))')


# In[8]:


def export_preds(preds, filename):
    with open(filename, "w") as out_file:
        out_file.write("\n".join(preds))


# In[9]:


export_preds(preds, "test_output.txt")

get_ipython().system('python3 ../data/eval.py test_output.txt ../data/sents.answer')


# In[ ]:




