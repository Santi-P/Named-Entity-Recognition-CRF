# Linear Conditional Random Fields CLASS
# Santi(chai) Pornavalai
# 30.3.19
# tested on Python 3.7.2
# santichai94@gmail.com
from tqdm import tqdm
# intended for Advanced Language Modelling course under TH
from copy import deepcopy
# mathy stuff
import numpy as np
import functools as fn
from random import shuffle, seed

# misc debugging and
from sys import getsizeof
import codecs

# vectorizer class from other file
from vectorizer import Vectorizer
# deqeue for front appending used in viterbi backtrace
from collections import deque
#object serialization
import pickle
from time import time

seed(42)

class CRF:
    """Class for training and predicting CRF based named entity recognition.
        main API functions are:
        CRF.fit(): training using avg. perceptron
        CRF.inference: prediction using viterbi algorithm
        save, load weights and vector transformers    
     """

    def __init__(self):

        # vectorizer class
        # based on composition instead of inheritence principles
        self.vectorizer = Vectorizer()

        # weights learned and used by model 
        self.weights = np.array([])
        self.tag_enums =  [] 

        self.tag_dict = {}

    
    def fit(self, file_name, iterations = 5):
        """ Wrapper function for initializing and training CRF model
            params: 
                file_name: training data file in GermEval format
                iterations: number of iterations. default to 5"""
        
        tags, tokens = self.read_file(file_name)
        # fix this shit later
        # build essential indices maps for the vectorizer
        self.vectorizer.build_tag_map(tags)
        self.vectorizer.build_word_map(tokens)
        self.vectorizer.build_name_list("data/first_names.txt")


        ### EDIT HERE TO ADD FUNCTIONS ###
        # add feature functions here #
        # descriptions in vectorizer file #
        self.vectorizer.add_feature("word", self.vectorizer.sparse_feat_word_in ,len(self.vectorizer.word_map))
        self.vectorizer.add_feature("prev word", self.vectorizer.sparse_feat_prev_word ,len(self.vectorizer.word_map))

        self.vectorizer.add_feature("word tag", self.vectorizer.sparse_feat_word_and_tag, len(self.vectorizer.word_map)*len(self.vectorizer.tag_map), True)
        self.vectorizer.add_feature("DE name gazetter", self.vectorizer.sparse_feat_in_names, 2)
       # self.vectorizer.add_feature("Caps", self.vectorizer.sparse_feat_is_all_cap, 2)
        #self.vectorizer.add_feature("hyphenated", self.vectorizer.sparse_feat_hyphenated, 2)
        
        # tag transitions must be added last so that the Viterbi can know where to look
        self.vectorizer.add_feature("prev tag", self.vectorizer.sparse_feat_prev_tag, len(self.vectorizer.tag_map)*len(self.vectorizer.tag_map), True)
        
        # fit vectorizer
        self.vectorizer.fit(tokens, tags)
        #initialize weight
        self.tag_enums = list(enumerate(self.vectorizer.tag_list))
        self.tag_dict = {word:idx for idx,word in self.tag_enums}
        self.initialize_weights(self.vectorizer.vector_size + 1)
        # perceptron train
        self.train(tokens, tags, iterations)
        print("Classifier Fitted")

    
    def predict(self,file_name):
        """ Wrapper for viterbi inference. Takes filename in GermEval format
            returns predicted tag sequence and actual tag seq in that order.
            returned in a flattened way"""
        tags, tokens = self.read_file(file_name)
        predicted_list = []

        actual_list = []
        for token_seq,tag_seq in zip(tokens,tags):
            predicted_tags = self.inference(token_seq)
            predicted_list.extend(predicted_tags)
            actual_list.extend(tag_seq)

        return predicted_list, actual_list

        


    def read_file(self, fname):
        """ GermEval file parser
            returns list of sequences of both tags and tokens
             """
        tag_seq = []
        tok_seq = []
        curr_tok = []
        curr_tag = []

        with open(fname,'r') as df:
            for line in df:

                line = line.strip().split("\t")
                if len(line) < 2:
                    tag_seq.append(curr_tag)
                    tok_seq.append(curr_tok)
                    curr_tok = []
                    curr_tag = []
                else:
                    if line[0] == '#':
                        #print("annot")
                        pass
                    else:
                        curr_tok.append(line[1])
                        curr_tag.append(line[2])
        return tag_seq, tok_seq


    def inference(self, token_seq,feats_list = False, int_tags = False):
        """ Viterbi Algorithm for decoding. Takes list of tokens and
            returns either list of predicted tags or additionally list of feature vectors
              """
        # check input for empty sequence     
        if len(token_seq) < 1:
            #print("invalid input encountered: empty tokens")
            return [],[]

        tag_len = len(self.vectorizer.tag_list)
        seq_len = len(token_seq)
        
        # initialize viterbi/ backpointer charts


        #### change this shit
        viterbi_chart = np.zeros((seq_len,tag_len))
        bp_chart = np.full((seq_len,tag_len),-1)
        feature_chart = [[{} for j in range(tag_len)] for i in range(seq_len)]  
        # initialize first trellis 
        for i,tag in self.tag_enums:
            viterbi_chart[0][i] = self.vectorizer.feature_dot(token_seq,tag,0,self.weights)
            feature_chart[0][i] = self.vectorizer.join_features(token_seq, [tag],0, 0 )

        # for each word
        for i in range(1,seq_len):
            #for each state
            for j,tag_1 in self.tag_enums:
                best_val = -1000000000000000000
                idx = -1

                # argmax
                # go through states with known transition
                for tag_2 in self.vectorizer.tag2tag[tag_1]:
                    vs = viterbi_chart[i-1][self.tag_dict[tag_2]] + \
                    self.vectorizer.get_trans_idx(token_seq,tag_1, tag_2,i,self.weights)

                    if vs > best_val:
                        best_val = vs
                        idx = self.tag_dict[tag_2]

                # update charts`
                bp_chart[i][j] = idx
                tag_2 = self.tag_enums[idx][1]

                # feature

                feature = self.vectorizer.feature(token_seq,tag_1,tag_2,i)         

                #scal_prod = self.vectorizer.sparse_dot(self.weights, feature)

                viterbi_chart[i][j] = viterbi_chart[i-1][idx] + sum([val*self.weights[k] for k,val in feature.items()])
                feature_chart[i][j] = feature

        # find max and initialize backtrace

        best = np.argmax (viterbi_chart[seq_len-1])
        # deque to append first
        res = deque()
        feat_vs = deque()
        res.append(best)
        feat_vs.append(feature_chart[len(token_seq)-1][best])
        # extract path
        for i in range(seq_len -1, 0, -1):
                # res.appendleft( self.vectorizer.tag_list[ int(bp_chart[i][int(best)]) ])

                res.appendleft(bp_chart[i][int(best)])
                feat_vs.appendleft(feature_chart[i][int(best)])
                best = bp_chart[i][int(best)]

        feat_vs.appendleft(feature_chart[0][int(best)])

        if int_tags == False:
            res = [self.vectorizer.tag_list[int(w)] for w in res]
        
        if feats_list:
            return list(res), feat_vs
        else: 
            return res

    def get_wrong_tags(self, y, tags):
        res = []
        #idx = 0
        for i in range(len(tags)):
            if y[i] != tags[i]:
                res.append(i)
        return res

    def train(self, tok_seq, tag_seq, iters = 5, learning_rate = 1):
        """ Train CRF model using avg. Perceptron algorithm. 
                takes list of token/tag sequences and attempts to learn
                something useful. learning rate can be set, but probably
                not that useful. iters set the number of iteration default to 5 """
        
        avg_weights = self.weights
        
        # zip training data to allow it to shuffle 
        # convert to list because shuffling won't work otherwise
        new_tags = [ [self.vectorizer.tag_map[tg] for tg in tg_list]  for tg_list in tag_seq]
        train_data= list(enumerate(zip(tok_seq,new_tags)))

        num_words = sum([len(seq) for seq in tok_seq])
        num_samples = len(tok_seq)
        
        # pre calculate gold vectors 
        gold_data = self.vectorizer.transform(tok_seq,tag_seq)
        # print(gold_data)
        # iterative loop
        for i in range(iters):
            # epoch timer
            #start = time()

            #shuffle(train_data)
            print("starting epoch:", i + 1)
            wrong = 0
            for idx, (tokens, tags) in tqdm(train_data):
                y, y_feats = self.inference(tokens,feats_list= True, int_tags= True)
                #wrong_tags = [ind for ind,pair in enumerate(zip(y,tags)) if pair[0] != pair[1]]
                wrong_tags = self.get_wrong_tags(y,tags)
                
                # if predicted wrong

                if len(wrong_tags) > 0:

                    # collect wrong for accuracy displayed after epoch
                    wrong += len(wrong_tags)

                    predicted = fn.reduce(self.vectorizer.sum_features,[y_feats[i] for i in wrong_tags])
                                        
                    gold_wrong = [ gold_data[idx][i] for i in wrong_tags]
                    gold =  fn.reduce(self.vectorizer.sum_features,gold_wrong)

                    diff = self.vectorizer.subtract_features(gold ,predicted)


                    self.vectorizer.add_weights(avg_weights,diff, lr= learning_rate)

            #end = time()
            #print("epoch time", end - start)
            print("accuracy:", (num_words - wrong)/num_words)

        # average 
        self.weights = avg_weights/(num_samples * iters)
        
            
    def initialize_weights(self,size,fill = 0):
        # dtype to double just in case
        # but normal float or half could also work 
        self.weights = np.full((size,),fill,dtype=np.float64)

    
    def save_weights(self, weights_fname, vectorizer_fname):
        """ save to binary. 
            arg1: name of weight filename
            arg2: name of vectorizer filename"""

        with open(weights_fname, "wb") as w_file:
            np.save(w_file, self.weights)
        with open(vectorizer_fname, "wb") as vec_file:
            pickle.dump(self.vectorizer,vec_file)
    

    def load_weights(self, weights_fname, vectorizer_fname):
        """ load from binary. 
            arg1: name of weight filename
            arg2: name of vectorizer filename"""

        with open(weights_fname, "rb") as w_file:
            self.weights = np.load(w_file)

        with open(vectorizer_fname, "rb") as vec_file:
            self.vectorizer = pickle.load(vec_file)


def main():
    path2data = "data/NER-de-train.tsv"
    path2smalldata = "data/minimini"
    path2testdata = "data/NER-de-dev.tsv"
    
    crf_model = CRF()
    crf_model.fit(path2smalldata, iterations=10)
    print(crf_model.vectorizer.tag_list)
    print(crf_model.vectorizer.tag_map)
    print(crf_model.tag_dict)

    


if __name__ == "__main__":
    # import profile
    # profile.run("main()")
    main()
