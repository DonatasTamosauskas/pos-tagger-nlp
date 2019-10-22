# python3.5 runtagger.py <test_file_absolute_path> <model_file_absolute_path> <output_file_absolute_path>

import os
import math
import sys
import datetime
import numpy as np
import _pickle as pickle
from functools import partial


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

            
class PosStatsModel():
    def __init__(self, pos_stats=None):
        self.emissions = {}
        self.transitions = {}
        self.dictionary = Dictionary()
        self.MIN_PROB_VALUE = 10e-5
        
        if pos_stats is not None:
            self.emissions = pos_stats.emissions
            self.transitions = pos_stats.transitions
            self.dictionary = pos_stats.dictionary
        
    def add_tag(self, tag):
        return self.dictionary.add_tag(tag)
    
    def get_all_tag_ids(self):
        return self.dictionary.tag_dict.keys()
        
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
    
    def get_emission_prob(self, word, tag):
        try:
            return self.emissions[(word, tag)]
        except KeyError:
            return self.MIN_PROB_VALUE
        
    def get_transition_prob(self, prev_tag, tag):
        try:
            return self.transitions[(prev_tag, tag)]
        except KeyError:
            return self.MIN_PROB_VALUE
    
    
def get_transition_probs(previous_probs, pos_stats, tags):
    max_ids = []
    probs = []
    
    for tag_id in tags:
        transitions = [pos_stats.get_transition_prob(prev_tag_id, tag_id) for prev_tag_id in tags]
        transitions_with_previous = previous_probs * np.array(transitions)
        max_id = np.argmax(transitions_with_previous)
        max_ids.append(max_id)
        probs.append(transitions_with_previous[max_id])
        
    return probs, max_ids

def combine_words_tags(words, tag_list, pos_stats):
    for i in range(len(words)):
        words[i] = words[i] + "/" + pos_stats.dictionary.tag_dict[tag_list[i]]
    
    return " ".join(words)
    
def get_sentence_tags(sentence, pos_stats):
    pos_tags = pos_stats.get_all_tag_ids()
    words = sentence.split(" ")

    viterbi = np.zeros((len(pos_tags), len(words)), float)
    backpointers = np.zeros((len(pos_tags), len(words)), int)

    # fill in initial probs
    # for now skipping the beginning of the sentence probability

    viterbi[:, 0] = [pos_stats.get_emission_prob(words[0], tag_id) for tag_id in pos_tags]
    backpointers[:, 0] = -1

    # fill in the full matrix
    for i, word in enumerate(words[1:]):
        emission_probs = [pos_stats.get_emission_prob(word, tag_id) for tag_id in pos_tags]
        transition_probs, new_backpointers = get_transition_probs(viterbi[:, i], pos_stats, pos_tags)

        viterbi[:, i + 1] = np.array(emission_probs) * np.array(transition_probs)
        backpointers[:, i + 1] = new_backpointers

    # Also need to calculate the ending probabilities
    tag_list = np.zeros(len(words) + 1, int)
    
    final_tag = np.argmax(viterbi[:, -1]) # this should be in accordance to the end of the sentence
    tag_list[-1] = final_tag
    previous_back = final_tag

    for word_pos in reversed(range(1, len(words))):
        tag_list[word_pos] = backpointers[previous_back, word_pos]
        previous_back = tag_list[word_pos]
        
    tag_list = tag_list[1:]
        
    return combine_words_tags(words, tag_list, pos_stats)

def read_and_tag(input_filename, pos_tagger):
    tagged_sent = []

    with open(input_filename, "r") as input_file:
        sentences = input_file.read().split("\n")

        for sentence in sentences:
            tagged_sent.append(pos_tagger(sentence))
    
    return tagged_sent

def load_model_file(filename):
    with open(filename, "rb") as model_read:
        model_file = pickle.load(model_read)
        return model_file
    
def tag_sentence(test_file, model_file, out_file):
    model_stats = PosStatsModel(pos_stats=load_model_file(model_file))
    hmm_viterbi_tagger = partial(get_sentence_tags, pos_stats=model_stats)
    
    tagged_sentences = read_and_tag(test_file, pos_tagger=hmm_viterbi_tagger)
    with open(out_file, "w") as out_file:
        out_file.write("\n".join(tagged_sentences[:-1]))
        
    print('Finished...') 


if __name__ == "__main__":
    # make no changes here
    test_file = sys.argv[1]
    model_file = sys.argv[2]
    out_file = sys.argv[3]
    start_time = datetime.datetime.now()
    tag_sentence(test_file, model_file, out_file)
    end_time = datetime.datetime.now()
    print('Time:', end_time - start_time)
