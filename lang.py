
import json
import urllib.request
import urllib.error
import numpy as np

import unicodedata
import os
import re
import random

from nltk.tokenize import sent_tokenize, word_tokenize


try:
    import cPickle
except ImportError:
    import pickle as cPickle

from os import listdir, makedirs
from os.path import isfile, join, exists


class Lang(object):
    def __init__(self, name, base_dir, vocab_dir):
        self.name = name
        self.base_dir = base_dir
        self.vocab_dir = vocab_dir
        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.n_words = 0

        self.PAD_token = 0
        self.EOS_token = 1
        self.add_word('<PAD>')
        self.add_word('<EOS>')

    def save(self):
        output_dir = self.vocab_dir
        if not exists(output_dir):
            makedirs(output_dir)

        with open(output_dir + '/' + self.__class__.__name__ + '.pkl', 'wb') as f:
            saved_dict = self.__dict__.copy()
            for k in saved_dict:
                if saved_dict[k].__class__.__module__ != 'builtins':
                    saved_dict[k] = None
            cPickle.dump(saved_dict, f)

    def load(self):
        print('self.vocab_dir:', self.vocab_dir)
        with open(self.vocab_dir + '/' + self.__class__.__name__ + '.pkl', 'rb') as f:
            recovered = cPickle.load(f)
        for name in recovered:
            if recovered[name] is not None:
                setattr(self, name, recovered[name])

    def indexes_from_sentence(self, sentence):
        return [self.get_index(word) for word in self.tokenize(sentence)] + [self.EOS_token]

    def tokenize(self, sentence):
        #return sentence.split()
        words = word_tokenize(sentence)
        print(sentence.split())
        print(words)
        print("-------")
        return words

    def pad_seq(self, seq, max_length):
        # Pad a with the PAD symbol
        seq += [self.PAD_token] * (max_length - len(seq))
        return seq

    def get_index(self, word):
        if word in self.word2index:
            return self.word2index[word]
        # if its an unseen word, just pad instead
        return self.PAD_token

    def get_word(self, index):
        if index in self.index2word:
            return self.index2word[index]
        return "<PAD>"
    def get_vocab_size(self):
        return self.n_words

    def add_sentence(self, sentence, normalize=False):
        for word in self.tokenize(sentence):
            if normalize:
                word = self.normalize_string(word)
            if len(word.strip()) == 0:
                continue
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def unicode_to_ascii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

    def normalize_string(self, s):
        """
        Lowercase, trim, and remove non-letter characters
        """
        s = s.lower().strip()
        s = self.unicode_to_ascii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        return s


class NaturalLang(Lang):
    def __init__(self, name, base_dir, vocab_dir):
        super(NaturalLang, self).__init__(name, base_dir, vocab_dir)

        import spacy
        self.nlp = spacy.load('en')

    def tokenize(self, sentence):
        # document = self.nlp(sentence)
        # return [token.text for token in document]
        words = word_tokenize(sentence)
        #print(sentence.split())
        #print(words)
        #print("-------")
        return words

    def indexes_from_sentence(self, sentence, normalize=True):
        # print('sentence:', sentence)
        results = []
        for word in self.tokenize(sentence):
            if normalize:
                word = self.normalize_string(word)
            if len(word.strip()) == 0:
                continue
            results.append(self.get_index(word))
        results.append(self.EOS_token)
        return results

    def get_vocab_embeddings(self, output_dir):
        embeddings = {}
        with open(output_dir + 'embedding_oov.txt', 'w') as output_file:
            for word in self.word2index:
                url = "http://localhost:5000/api/vector/fasttext/" + word
                try:
                    response = urllib.request.urlopen(url)
                    data = json.loads(response.read())
                    embeddings[word] = data['vector']
                except Exception as err:
                    print('Cannot get embedding for', word, err)
                    output_file.write(word + '\t' + str(err) + '\n')
        return embeddings


class CUILang(Lang):
    def __init__(self, name, base_dir, vocab_dir):
        super(CUILang, self).__init__(name, base_dir, vocab_dir)

    def tokenize(self, sentence):
        if ',' in sentence:
            return [token.strip() for token in sentence.split(',') if len(token.strip()) > 0]
        else:
            return [token.strip() for token in sentence.split() if len(token.strip()) > 0]

    def get_vocab_embeddings(self, output_dir):
        embeddings = {}
        with open(output_dir + 'embedding_oov.txt', 'w') as output_file:
            for word in self.word2index:
                url = "http://localhost:5000/api/vector/cui/" + word
                try:
                    response = urllib.request.urlopen(url)
                    data = json.loads(response.read())
                    embeddings[word] = data['vector']
                except Exception as err:
                    print('Cannot get embedding for', word, err)
                    output_file.write(word + '\t' + str(err) + '\n')
        return embeddings

