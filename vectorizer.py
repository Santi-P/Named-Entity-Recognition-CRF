# Vectorizer class
# Santi(chai) Pornavalai
# 30.1.19
# tested on Python 3.7.2
# santichai94@gmail.com

# intended for Advanced Language Modelling course under TH

import numpy as np
import functools as fn
import codecs
from copy import copy

class Vectorizer:
    """ Vectorizer class responsible for:
     building word/tag indices maps
     sparse vector arithmetic
     feature extraction
        """

    def __init__(self):
        self.word_map = dict()
        self.tag_map = dict()
        self.map_sentinel = 0
        self.tag_sentinel = 0
        self.name_list = set()
        self.tag_list = list()
        self.vector_size = 0
        self.partitions = []
        self.tag2tag = dict()

        self.feature_list = []
        
#### API FUNCTIONS #####

    def transform(self, tokens,tags):
        """transforms list of token/tag sequences into vectorized form
            might be buggy and shouldn't be used yet"""
        res = []
        train_data= list(zip(tokens,tags))
        for token, tag in train_data:
            tmp = []
            for i in range(len(tag)):
                tmp.append(self.join_features(token,tag,i,i) )

            res.append(tmp)
        return res

    def add_feature(self, feature_name, fn_object,f_length, transition_feat = False):
        """ adds a feature to be used by the vectorizer object. 

            feature_name: specify the name of the feature. is irrelevant to performance.

            fn_object: function that takes either 
            token_sequence ,position OR token,tags ,pos_tok, pos_tag  as arguments
            this function object should return and a tuple in the form of INDEX,VALUE which 
            would be added to the main feature vector. 

            f_length: the size theoretically needed for this specific feature

            transition feat is used to specify the type of feature. In other words, word features 
            and transition features

              """

        self.feature_list.append((feature_name, fn_object,f_length, transition_feat ))



    def fit(self, tokens, tags):
        """ Wrapper for building and initializing vectorizer objects
        prints a summary of no. of params etc. 
            """
        self.build_word_map(tokens)
        self.build_tag_map(tags)
        self.build_partitions()

        summary = """Vectorizer Fitted:
            number of features: {}
            number of parameters: {}
            number of sentences: {}
            """.format(len(self.feature_list),self.vector_size, len(tokens))
        print(summary)





##### UTILITY FUNCTIONS #######


##### UTILITY FUNCTIONS FOR BUILDING CLASS #####

    def build_partitions(self):
        """ Builds the partition list. this tells the vectorizer
            where each individual feature starts. also gets the size 
            of the feature vector as a by-product"""
        
        f_size = sum([x[2] for x in self.feature_list])
        partitions = []
        acc = 1
        for f in self.feature_list:
            
            partitions.append(acc)
            acc+= f[2]

        self.vector_size = f_size
        self.partitions = partitions
        

    def build_name_list(self, fname ):
        """ Used for initializing list objects for gazeteer feature functions"""
        with open(fname,"r") as f:
            for line in f:
                line = line.split(",")
                self.name_list.add(line[0].strip())


    def build_word_map(self,tok_seq):
        """ build word index"""
        for seq in tok_seq:
            for tok in seq:
                if tok not in self.word_map.keys():
                    self.word_map[tok] = self.map_sentinel
                    self.map_sentinel += 1

    def build_tag_map(self,tag_seq):
        """ build tag index and also remembers transitions which
            is used to speed up the viterbi algorithm."""
        for seq in tag_seq:
            #print(seq)
            for tag,next_tag in zip(seq,seq[1:]):
                
                # if in t2t
                if self.tag2tag.get(next_tag) != None:
                    self.tag2tag[next_tag].add(tag)
                else:
                    self.tag2tag[next_tag] = set([tag])

            for tag in seq:
                if tag not in self.tag_map.keys():
                    self.tag_map[tag] = self.tag_sentinel
                    self.tag_sentinel += 1
                    self.tag_list.append(tag)
        

    #####  FEATURE COMBINATIONS ######


    def feature(self, token_seq, tag ,prev_tag,pos):
        tag_seq = [prev_tag,tag]
        F_vec = self.join_features(token_seq, tag_seq, pos, 1 )
        return F_vec



######## Sparse Feature #########
    ### check none return 
    def join_features(self, token,tags ,pos_tok, pos_tag):
        """ combines the individual features together"""
     
       # print("sparsifying")

        lil = {}
        # partitions starts from first. 
        for off_set, feat in zip(self.partitions,self.feature_list):
            feat_func, is_trans = feat[1],feat[3]
            if is_trans:
                idx , val = feat_func(token,tags,pos_tok,pos_tag,off_set=off_set)
                lil[idx] = val
            else:
                idx, val = feat_func(token, pos_tok,off_set = off_set)
                lil[idx] = val
        return lil

    def sparse_feat_word_in(self,token_seq,pos,off_set = 0):
        token = token_seq[pos]
        if self.word_map.get(token) != None: 
            idx = self.word_map[token] + off_set
            return idx,1
        return 0,0

    def sparse_feat_prev_word(self,token_seq,pos,off_set = 0):
        if pos <1:
            return 0,0
        token = token_seq[pos - 1]
        if self.word_map.get(token) != None: 
            idx = self.word_map[token] + off_set
            return idx,1
        return 0,0

    def sparse_feat_is_all_cap(self,token_seq,pos,off_set = 0):
        token = token_seq[pos]      
        if token.isupper():
            return off_set + 1, 1
        return off_set + 1,0
    
    def sparse_feat_is_one_cap(self,token_seq,pos,off_set = 0):
        token = token_seq[pos]
        
        if not token.islower():
            return off_set + 1, 1
        return off_set + 1,0
        
    def sparse_feat_hyphenated(self,token_seq,pos,off_set = 0):
        token = token_seq[pos]
        # this is slow
        if "-" in token:
            return off_set + 1, 1
        return off_set + 1,0


    def sparse_feat_word_and_tag(self, token_seq, tag_seq, pos_tok, pos_tag , off_set = 0):


        tag = tag_seq[pos_tag]
        tok = token_seq[pos_tok]

        if (self.word_map.get(tok) != None) and (self.tag_map.get(tag) != None):
            tag_id, tok_id = self.tag_map[tag], self.word_map[tok]
            idx = tok_id * self.tag_sentinel + tag_id + off_set

            return idx,1
        #print("bing")
        return 0,0

    def sparse_feat_prev_tag(self, token_seq, tag_seq, pos_tok, pos_tag, off_set = 0 ):
    #print(self.tag_list)

        if pos_tag  < 1:
            return 0,0
        tag = tag_seq[pos_tag]
        prev_tag = tag_seq[pos_tag-1]

        if self.tag_map.get(tag) != None and self.tag_map.get(prev_tag) != None:

            tag_curr_id, tag_prev_id = self.tag_map[tag], self.tag_map[prev_tag]
            idx = tag_curr_id * self.tag_sentinel + tag_prev_id + off_set 

            return idx, 1
        #print("bing")
        return 0,0

    def sparse_feat_in_names(self,token_seq,pos,off_set = 0):
        tk = token_seq[pos]
        if tk in self.name_list:
            return off_set + 1, 1
        return off_set + 1,0

    
    
    def get_trans_idx(self,token_seq, tag, prev, pos, weights):
        """ used in the maximization phase viterbi algorithm.
         fetches index of two tags and returns dot product"""
        tag_seq = [prev, tag]
        beg_trans = self.partitions[-1]
        idx, val = self.sparse_feat_prev_tag(token_seq, tag_seq, pos, 1, off_set= beg_trans)
        return weights[idx] * val



  ##### MATH FUNCTIONS FOR FEATURE VECTORS ######

    def feature_dot(self, token_seq, tag, pos, weights):
        """ Dot product for feature vectors"""
        # work arround to make it conform to join features
        tag_seq = [tag]
        F_vec = self.join_features(token_seq, tag_seq, pos, 0 )
        acc = sum([val*weights[k] for k,val in F_vec.items()])    
        return acc

    def feature_dot_prev(self, token_seq, tag, prev, pos, weights):
        """ Dot product with previous vectors"""
        # work arround to make it conform to join features
        tag_seq = [prev, tag]
        F_vec = self.join_features(token_seq, tag_seq, pos, 1 )
        acc = sum([val*weights[k] for k,val in F_vec.items()])
        return acc

    def sparse_dot(self,weights, fvec):
        """ another dot product for weights"""
        return sum([val*weights[k] for k,val in fvec.items()])

    def add_weights(self, weights,  featv, lr = 1):
        """ add sparse vector to weight """
        for k, val in featv.items():
            weights[k]  += lr*val
        

    def sum_features(self, v1,v2):
        """ add sparse features together"""

        #copy to ensure deep copy
        p1 = v1.copy()
        p2 = v2.copy()

        for key in p2.keys():
            if p1.get(key) != None:
                p1[key] += p2[key]
            else:
                p1[key] = p2[key]
        return p1

    def subtract_features(self, v1, v2):
        """ substract sparse features"""
        p1 = v1.copy()
        p2 = v2.copy()

        for key in p2.keys():
            if p1.get(key) != None:
                p1[key] -= p2[key]
            else:
                p1[key] = -(p2[key])
        return p1



def main():
    print("VECTORIZER TEST")

    tok = [["hello ","world", "hi"]]
    tags = [["H", "W", "h"]]
    vectorizer = Vectorizer()
    vectorizer.add_feature("word", vectorizer.sparse_feat_word_in, 3)
    vectorizer.add_feature("word tag", vectorizer.sparse_feat_word_and_tag, 9, True)
    vectorizer.add_feature("prev tag", vectorizer.sparse_feat_prev_tag, 9, True)
    #vectorizer.build_word_map(tok)
    #vectorizer.build_tag_map(tags)
    vectorizer.fit(tok,tags)
    w = [0 for i in range(21)]
    w[3 ]= 1
    v1 = vectorizer.join_features(tok[0],tags[0] , 1 ,1 )
    v2 = vectorizer.feature(tok[0],"W" , "H" ,1 )
    print(v1, v2)


if __name__ == "__main__":
    main()