import json

import networkx
import pandas as pd
from nltk import word_tokenize

from embedding import Embedding
import torch

from collections import OrderedDict

class GenerateGraph:

    def __init__(self, train_data, dev_data):
        self.train_data = train_data
        self.dev_data = dev_data
        self.graph = networkx.Graph()
        self.dic_token_nodes = {}
        self.dic_toxic_terms = {}
        self.node_sentence = {}
        self.node_toxicicity = {}
        self.node_toxic_terms = {}
        self.node_ids = 0
        self.mol_lex = self.toxic_terms('data/mol_toxic.csv')
        self.mol_df = self.toxic_terms_df('data/mol_toxic.csv')
        self.embeddings = Embedding('embeddings/glove_s300.txt')
        self.vocabulary = self.read_resource('data/vocabulary.txt', 'data/badword_list.json')

    @staticmethod
    def read_resource(vocabulary: str, word_list: str) -> list:
        vocab = []
        with open(vocabulary, 'r') as f:
            for line in f.readlines():
                vocab.extend(line.split(','))
        with open(word_list, 'r') as f:
            data = json.loads(f.read())
            vocab.extend(set(list(data)))
        return vocab
    
    @staticmethod
    def toxic_terms(lexicon):
        mol_df = pd.read_csv(lexicon, sep=',', encoding='utf8')
        mol_df = mol_df[mol_df['explicit-or-implicit'] == 'explicit']
        mol_lex = mol_df['pt-brazilian-portuguese'].to_list()
        return mol_lex
    
    @staticmethod
    def toxic_terms_df(lexicon):
        mol_df = pd.read_csv(lexicon, sep=',', encoding='utf8')
        mol_df = mol_df[mol_df['explicit-or-implicit'] == 'explicit']
        #mol_lex = mol_df['pt-brazilian-portuguese'].to_list()
        return mol_df
        
    
    #Generate graph with explicits terms node
    def create_graph_toxicity(self, split_data, split_name):
        for idx, sentence in enumerate(split_data):
            key = split_name + '_%d' % idx
            self.node_sentence[key] = self.node_ids
            self.node_ids += 1
            self.graph.add_node(self.node_sentence[key], type='sentence', value=key)
            
            tokens = word_tokenize(sentence, language='portuguese')       
            
            token_weight = 0        
            for tk in tokens:
                   
                for i,term in  enumerate(self.mol_lex):
                    if term == tk:
                        temp = self.mol_df.iloc[i]
                        score = temp['toxicity_score']
                        #token_weight = token_weight + 1 + score
                        token_weight = token_weight + score
                        break
                    #else:
                    #    token_weight += 1
                
                if tk not in self.dic_token_nodes:
                    self.dic_token_nodes[tk] = self.node_ids
                    self.node_ids += 1
                    self.graph.add_node(self.dic_token_nodes[tk])
                    self.graph.add_edge(self.node_sentence[key], self.dic_token_nodes[tk])
                    weight = float('%.4f' % (self.get_weight(tk, sentence)))

                    self.graph.add_edge(self.node_sentence[key], self.dic_token_nodes[tk], weight=weight)
                else:
                    weight = float('%.4f' % (self.get_weight(tk, sentence)))
                    self.graph.add_edge(self.node_sentence[key], self.dic_token_nodes[tk], weight=weight)
            
            if token_weight == 0:
                token_weight += 0.01
            
            self.node_toxic_terms[key] = self.node_ids
            self.node_ids += 1
            self.graph.add_node(self.node_toxic_terms[key])
            #w = (token_weight/len(tokens)) + 1
            w = token_weight
            self.graph.add_edge(self.node_sentence[key], self.node_toxic_terms[key], weight=w)

    def get_weight(self, token: str, sentence: str) -> float:
        if token in self.vocabulary:
            return self.embeddings.get_embeddings(token, sentence)
        else:
            return 0.0
    
    def generate_graph_toxicity(self):
        self.create_graph_toxicity(self.train_data['text'].values, 'train')
        self.create_graph_toxicity(self.dev_data['text'].values, 'dev')
        
        return self.graph, self.node_sentence
