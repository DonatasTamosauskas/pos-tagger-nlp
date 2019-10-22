# python3.5 buildtagger.py <train_file_absolute_path> <model_file_absolute_path>

import os
import math
import sys
import datetime
import numpy as np
import itertools
import _pickle as pickle

flatten = lambda x: list(itertools.chain.from_iterable(x))

class Dictionary():
    def __init__(self):
        self.word_dict = {}
        self.tag_dict = {}
        
        self.word_count = 0
        self.tag_count = 0
    
    def add_word(self, word):
        if word not in self.word_dict.values():
            self.word_dict[self.word_count] = word
            self.word_count += 1
            return self.word_count - 1
        return self.get_word_id(word)
    
    def add_tag(self, tag):
        if tag not in self.tag_dict.values():
            self.tag_dict[self.tag_count] = tag
            self.tag_count += 1
            return self.tag_count - 1
        return self.get_tag_id(tag)
            
    def get_word_id(self, word):
        for key, known_word in self.word_dict.items():
            if word == known_word:
                return key
    
    def get_tag_id(self, tag):
        for key, known_tag in self.tag_dict.items():
            if tag == known_tag:
                return key
    
    def get_word_tag_tuple(self, word_tag):
        words_tag = word_tag.split("/")
        assert len(words_tag) > 1, "Bad format for word/tag: {}".format(word_tag)
        
        word_ids = [self.get_word_id(word) for word in words_tag[:-1]]
        tag_id = self.get_tag_id(words_tag[-1])
        word_ids.append(tag_id)
        return tuple(word_ids)
    
    
class PosStatsText():
    def __init__(self):
        self.emissions = {}
        self.transitions = {}
        self.dictionary = Dictionary()
        
    def add_tag(self, tag):
        return self.dictionary.add_tag(tag)
        
    def add_emission(self, word_id, tag_id):
        if (word_id, tag_id) not in self.emissions.keys():
            self.emissions[(word_id, tag_id)] = 1
        else:
            self.emissions[(word_id, tag_id)] += 1
        
    def add_transition(self, previous_tag_id, tag_id):
        if (previous_tag_id, tag_id) not in self.transitions.keys():
            self.transitions[(previous_tag_id, tag_id)] = 1
        else:
            self.transitions[(previous_tag_id, tag_id)] += 1

            
def extract_frequencies_tuple_text(filename):    
    pos_stats = PosStatsText()

    with open(filename) as training_file:
        word_tag_pairs = flatten([sentence.split(" ") for sentence in training_file.read().split("\n")])        
        previous_tag_id = pos_stats.add_tag(word_tag_pairs[0].split("/")[-1])

        for word_tag in word_tag_pairs:
            if len(word_tag.split("/")) < 2:
                continue

            word, tag = word_tag.split("/")[-2:]
            tag_id = pos_stats.add_tag(tag)
            
            pos_stats.add_emission(word, tag_id)
            pos_stats.add_transition(previous_tag_id, tag_id)

            previous_tag_id = tag_id
    
    return pos_stats 

def write_output(model_stats, filename="model-file"):
    with open(filename, "wb") as model_file:
        model_file.write(pickle.dumps(model_stats))
    
def train_model(train_file, model_file):
    model_stats = extract_frequencies_tuple_text(train_file)
    write_output(model_stats, filename=model_file)
    print('Finished...')
    

if __name__ == "__main__":
    # make no changes here
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    start_time = datetime.datetime.now()
    train_model(train_file, model_file)
    end_time = datetime.datetime.now()
    print('Time:', end_time - start_time)
