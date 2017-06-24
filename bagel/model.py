import sys
import os
from itertools import permutations      # for unordered Sm
from random import shuffle              # for random spliting of train/test

sys.path.insert(0, '../data_wrangling/')

import data_loader
import blue


# TODO: incorporate end-stacks

class Bagel:

    # ---- CONSTRUCTOR ----

    def __init__(self):
        # data
        self.mrs_orig = []
        self.mrs_delex= []
        self.mrs_orig_test = []
        self.mrs_delex_test = []
        self.stack_sequences=[]
        self.phrase_sequences=[]
        self.stack_sequences_train = []
        self.phrase_sequences_train = []
        self.phrase_sequences_test= []

        # mappings
        self.stack_tup2idx = {}
        self.stack_idx2tup = []
        self.head_tup2idx = {}
        self.head_idx2tup = []
        self.tail_tup2idx = {}
        self.tail_idx2tup = []
        self.phrase_word2idx = {}
        self.phrase_idx2word = []

        # probabilities
        self.probs_stacks = {}          # 0th backoff level
        self.probs_stacks1 = {}         # 1st backoff level
        self.probs_stacks2 = {}         # 2nd backoff level
        self.probs_phrases = {}         # 0th backoff level
        self.probs_phrases1 = {}        # 1st backoff level
        self.probs_phrases2 = {}        # 2nd backoff level
        self.probs_phrases3 = {}        # 3rd backoff level
        self.probs_phrases4 = {}        # 4th backoff level
        self.probs_phrases5 = {}        # 5th backoff level
        self.probs_phrases6 = {}        # 6th backoff level
        self.prob_prune = 0.0

        self.train_test_split = 1.0     # if = 0.8 -> 0.8 train and 0.2 test
        self.cur_fold = 0

        self.stack_recursion_ctr = 0
        self.phrase_recursion_ctr = 0


    # ---- PUBLIC METHODS ----

    def set_split(self, new_split):
        self.train_test_split = new_split

    def set_fold(self, new_fold):
        self.cur_fold = new_fold


    def load_data(self, path_to_training, to_shuffle=True):
        self.mrs_orig, self.mrs_delex, self.stack_sequences, self.phrase_sequences = data_loader.load_data(path_to_training)

        # map stacks/phrases onto integers
        self.stack_tup2idx, self.stack_idx2tup = self.__create_mapping_stacks(self.stack_sequences)
        self.head_tup2idx, self.head_idx2tup, self.tail_tup2idx, self.tail_idx2tup = self.__create_mapping_heads_tails(self.stack_sequences)
        self.phrase_word2idx, self.phrase_idx2word = self.__create_mapping_phrases(self.phrase_sequences)

        # we cannot shuffle each list individually because element correspondance is lost
        # instead do it in unisen
        if to_shuffle:
            merged = list(zip(self.mrs_orig, self.mrs_delex, self.stack_sequences, self.phrase_sequences))
            shuffle(merged)
            self.mrs_orig, self.mrs_delex, self.stack_sequences, self.phrase_sequences = zip(*merged)

        # get training stack/phrase sequences. Note: corresponding test structures are not needed
        self.stack_sequences_train = self.stack_sequences[0 : int(self.train_test_split * len(self.stack_sequences))]
        self.phrase_sequences_train = self.phrase_sequences[0 : int(self.train_test_split * len(self.phrase_sequences))]

        # get test MRs and utterances
        self.mrs_orig_test = self.mrs_orig[int(self.train_test_split * len(self.mrs_orig)) : ]
        self.mrs_delex_test = self.mrs_delex[int(self.train_test_split * len(self.mrs_delex)) : ]
        self.phrase_sequences_test = self.phrase_sequences[int(self.train_test_split * len(self.phrase_sequences)) : ]
        #self.mrs_orig_test = self.mrs_orig[: int(self.train_test_split * len(self.mrs_orig))]
        #self.mrs_delex_test = self.mrs_delex[: int(self.train_test_split * len(self.mrs_delex))]
        #self.phrase_sequences_test = self.phrase_sequences[: int(self.train_test_split * len(self.phrase_sequences))]


        print('\t Size of the training set: ' + str(len(self.stack_sequences_train)))
        print('\t Size of the test set: ' + str(len(self.mrs_delex_test)))


    # calculate the probability of the given stack configuration
    def get_stack_config_probs(self, seq, stack_idx, backoff_depth):
        if backoff_depth > 2:
            raise IndexError('Stack backoff depth greater than 2!')

        subseq_n = seq[max(0, stack_idx - 2 + backoff_depth) : stack_idx + 1]     # numerator
        subseq_d = seq[max(0, stack_idx - 2 + backoff_depth) : stack_idx]         # denominator
           
        key = tuple([ self.stack_tup2idx[tuple(stack)] for stack in subseq_n ])
        
        if backoff_depth == 0 and key not in self.probs_stacks:
            value = self.__get_freq_stacks(subseq_n) / self.__get_freq_stacks(subseq_d)
            self.probs_stacks[key] = value
        
        if backoff_depth == 1 and key not in self.probs_stacks1:
            value = self.__get_freq_stacks(subseq_n) / self.__get_freq_stacks(subseq_d)
            self.probs_stacks1[key] = value
        
        if backoff_depth == 2 and key not in self.probs_stacks2:
            value = self.__get_freq_stacks(subseq_n) / self.__get_freq_stacks(subseq_d)
            self.probs_stacks2[key] = value


    # calculate the probability of the given phrase/stack configuration
    def get_phrase_config_probs(self, seq_phrases, seq_stacks, idx, backoff_depth):
        if backoff_depth >= 0:
            # r_{t-1}, r_{t}
            phrase_subseq_n = self.__phrase_seq_to_triple(seq_phrases, idx)
            phrase_subseq_n[2] = None
            # l_{t-1}, l_{t}, l_{t+1}
            tail_subseq_n = self.__stack_seq_to_triple_tails(seq_stacks, idx)
            # h_{t-1}, h_{t}, h_{t+1}
            head_subseq_n = self.__stack_seq_to_triple_heads(seq_stacks, idx)
            
            phrase_subseq_d = self.__phrase_seq_to_triple(seq_phrases, idx)
            phrase_subseq_d[1] = None
            phrase_subseq_d[2] = None
            tail_subseq_d = tail_subseq_n
            head_subseq_d = head_subseq_n

            self.__calculate_phrase_config_prob(self.probs_phrases, phrase_subseq_n, tail_subseq_n, head_subseq_n, phrase_subseq_d, tail_subseq_d, head_subseq_d)
        
        if backoff_depth >= 1:
            ## r_{t-1}, r_{t}
            #phrase_subseq_n = self.__phrase_seq_to_triple(seq_phrases, idx)
            #phrase_subseq_n[2] = None
            ## l_{t-1}, l_{t}, l_{t+1}
            #tail_subseq_n = self.__stack_seq_to_triple_tails(seq_stacks, idx)
            ## h_{t-1}, h_{t}
            #head_subseq_n = self.__stack_seq_to_triple_heads(seq_stacks, idx)
            #head_subseq_n[2] = None
            
            #phrase_subseq_d = self.__phrase_seq_to_triple(seq_phrases, idx)
            #phrase_subseq_d[1] = None
            #phrase_subseq_d[2] = None
            #tail_subseq_d = tail_subseq_n
            #head_subseq_d = head_subseq_n

            #self.__calculate_phrase_config_prob(self.probs_phrases1, phrase_subseq_n, tail_subseq_n, head_subseq_n, phrase_subseq_d, tail_subseq_d, head_subseq_d)

            # r_{t-1}, r_{t}
            phrase_subseq_n = self.__phrase_seq_to_triple(seq_phrases, idx)
            phrase_subseq_n[2] = None
            # l_{t-1}, l_{t}, l_{t+1}
            tail_subseq_n = self.__stack_seq_to_triple_tails(seq_stacks, idx)
            # h_{t}
            head_subseq_n = [None, seq_stacks[idx][-1], None]

            phrase_subseq_d = self.__phrase_seq_to_triple(seq_phrases, idx)
            phrase_subseq_d[1] = None
            phrase_subseq_d[2] = None
            tail_subseq_d = tail_subseq_n
            head_subseq_d = head_subseq_n

            self.__calculate_phrase_config_prob(self.probs_phrases1, phrase_subseq_n, tail_subseq_n, head_subseq_n, phrase_subseq_d, tail_subseq_d, head_subseq_d)
            
            ## --- || ---
            ## h_{t}, h_{t+1}
            #head_subseq_n = self.__stack_seq_to_triple_heads(seq_stacks, idx)
            #head_subseq_n[0] = None
            
            #head_subseq_d = head_subseq_n

            #self.__calculate_phrase_config_prob(self.probs_phrases1, phrase_subseq_n, tail_subseq_n, head_subseq_n, phrase_subseq_d, tail_subseq_d, head_subseq_d)
        
        if backoff_depth >= 2:
            # r_{t-1}, r_{t}
            phrase_subseq_n = self.__phrase_seq_to_triple(seq_phrases, idx)
            phrase_subseq_n[2] = None
            # l_{t-1}, l_{t+1}
            tail_subseq_n = self.__stack_seq_to_triple_tails(seq_stacks, idx)
            tail_subseq_n[1] = None
            # h_{t}
            head_subseq_n = [None, seq_stacks[idx][-1], None]
            
            phrase_subseq_d = self.__phrase_seq_to_triple(seq_phrases, idx)
            phrase_subseq_d[1] = None
            phrase_subseq_d[2] = None
            tail_subseq_d = tail_subseq_n
            head_subseq_d = head_subseq_n

            self.__calculate_phrase_config_prob(self.probs_phrases2, phrase_subseq_n, tail_subseq_n, head_subseq_n, phrase_subseq_d, tail_subseq_d, head_subseq_d)
        
        if backoff_depth >= 3:
            # r_{t-1}, r_{t}
            phrase_subseq_n = self.__phrase_seq_to_triple(seq_phrases, idx)
            phrase_subseq_n[2] = None
            # l_{t-1}
            tail_subseq_n = self.__stack_seq_to_triple_tails(seq_stacks, idx)
            tail_subseq_n[1] = None
            tail_subseq_n[2] = None
            # h_{t}
            head_subseq_n = [None, seq_stacks[idx][-1], None]
            
            phrase_subseq_d = self.__phrase_seq_to_triple(seq_phrases, idx)
            phrase_subseq_d[1] = None
            phrase_subseq_d[2] = None
            tail_subseq_d = tail_subseq_n
            head_subseq_d = head_subseq_n

            self.__calculate_phrase_config_prob(self.probs_phrases3, phrase_subseq_n, tail_subseq_n, head_subseq_n, phrase_subseq_d, tail_subseq_d, head_subseq_d)

            # --- || ---
            # l_{t+1}
            tail_subseq_n = self.__stack_seq_to_triple_tails(seq_stacks, idx)
            tail_subseq_n[0] = None
            tail_subseq_n[1] = None

            tail_subseq_d = tail_subseq_n

            self.__calculate_phrase_config_prob(self.probs_phrases3, phrase_subseq_n, tail_subseq_n, head_subseq_n, phrase_subseq_d, tail_subseq_d, head_subseq_d)
            
            # --- || ---
            # r_{t}
            phrase_subseq_n = [None, seq_phrases[idx], None]
            # l_{t-1}, l_{t+1}
            tail_subseq_n = self.__stack_seq_to_triple_tails(seq_stacks, idx)
            tail_subseq_n[1] = None
            
            phrase_subseq_d = [None, None, None]
            tail_subseq_d = tail_subseq_n

            self.__calculate_phrase_config_prob(self.probs_phrases3, phrase_subseq_n, tail_subseq_n, head_subseq_n, phrase_subseq_d, tail_subseq_d, head_subseq_d)
            
        if backoff_depth >= 4:
            # r_{t-1}, r_{t}
            phrase_subseq_n = self.__phrase_seq_to_triple(seq_phrases, idx)
            phrase_subseq_n[2] = None
            # no tail
            tail_subseq_n = [None, None, None]
            # h_{t}
            head_subseq_n = [None, seq_stacks[idx][-1], None]
            
            phrase_subseq_d = self.__phrase_seq_to_triple(seq_phrases, idx)
            phrase_subseq_d[1] = None
            phrase_subseq_d[2] = None
            tail_subseq_d = tail_subseq_n
            head_subseq_d = head_subseq_n

            self.__calculate_phrase_config_prob(self.probs_phrases4, phrase_subseq_n, tail_subseq_n, head_subseq_n, phrase_subseq_d, tail_subseq_d, head_subseq_d)

            # --- || ---
            # r_{t}
            phrase_subseq_n = [None, seq_phrases[idx], None]
            # l_{t-1}
            tail_subseq_n = self.__stack_seq_to_triple_tails(seq_stacks, idx)
            tail_subseq_n[1] = None
            tail_subseq_n[2] = None
            
            phrase_subseq_d = [None, None, None]
            tail_subseq_d = tail_subseq_n

            self.__calculate_phrase_config_prob(self.probs_phrases4, phrase_subseq_n, tail_subseq_n, head_subseq_n, phrase_subseq_d, tail_subseq_d, head_subseq_d)
            
            # --- || ---
            # l_{t+1}
            tail_subseq_n = self.__stack_seq_to_triple_tails(seq_stacks, idx)
            tail_subseq_n[0] = None
            tail_subseq_n[1] = None
            
            tail_subseq_d = tail_subseq_n

            self.__calculate_phrase_config_prob(self.probs_phrases4, phrase_subseq_n, tail_subseq_n, head_subseq_n, phrase_subseq_d, tail_subseq_d, head_subseq_d)
        
        if backoff_depth >= 5:
            # r_{t}
            phrase_subseq_n = [None, seq_phrases[idx], None]
            # no tail
            tail_subseq_n = [None, None, None]
            # h_{t}
            head_subseq_n = [None, seq_stacks[idx][-1], None]
            
            phrase_subseq_d = [None, None, None]
            tail_subseq_d = tail_subseq_n
            head_subseq_d = head_subseq_n

            self.__calculate_phrase_config_prob(self.probs_phrases5, phrase_subseq_n, tail_subseq_n, head_subseq_n, phrase_subseq_d, tail_subseq_d, head_subseq_d)
        
        if backoff_depth >= 6:
            # r_{t}
            phrase_subseq_n = [None, seq_phrases[idx], None]
            # no tail
            tail_subseq_n = [None, None, None]
            # no head
            head_subseq_n = [None, None, None]
            
            phrase_subseq_d = [None, None, None]
            tail_subseq_d = tail_subseq_n
            head_subseq_d = head_subseq_n

            self.__calculate_phrase_config_prob(self.probs_phrases6, phrase_subseq_n, tail_subseq_n, head_subseq_n, phrase_subseq_d, tail_subseq_d, head_subseq_d)


    # train the dynamic Bayesian network (DBN) from the training set
    def train(self, max_stack_backoff, max_phrase_backoff):
        for seq in self.stack_sequences_train:
            for stack_idx in range(0, len(seq)):
                for b in range(max_stack_backoff + 1):
                    self.get_stack_config_probs(seq, stack_idx, b)
                    
        # DEBUG PRINT
        #print('---- Stack 3-gram probabilities ({0}) ----\n'.format(len(self.probs_stacks)))
        #for k, v in self.probs_stacks.items():
        #    print(k, ':', v)
        #print('\n---- Stack 2-gram probabilities ({0}) ----\n'.format(len(self.probs_stacks1)))
        #for k, v in self.probs_stacks1.items():
        #    print(k, ':', v)
        #print('\n---- Stack 1-gram probabilities ({0}) ----\n'.format(len(self.probs_stacks2)))
        #for k, v in self.probs_stacks2.items():
        #    print(k, ':', v)
    
        for seq_phrases, seq_stacks in zip(self.phrase_sequences_train, self.stack_sequences_train):
            for idx in range(0, len(seq_phrases)):
                self.get_phrase_config_probs(seq_phrases, seq_stacks, idx, max_phrase_backoff)
    
        # DEBUG PRINT
        print('Stack dict 0 size:', len(self.probs_stacks))
        print('Stack dict 1 size:', len(self.probs_stacks1))
        print('Stack dict 2 size:', len(self.probs_stacks2))
        print('Phrase dict 0 size:', len(self.probs_phrases))
        print('Phrase dict 1 size:', len(self.probs_phrases1))
        print('Phrase dict 2 size:', len(self.probs_phrases2))
        print('Phrase dict 3 size:', len(self.probs_phrases3))
        print('Phrase dict 4 size:', len(self.probs_phrases4))
        print('Phrase dict 5 size:', len(self.probs_phrases5))
        print('Phrase dict 6 size:', len(self.probs_phrases6))
        #print('\n\n---- Phrase n-gram probabilities ({0}) ----\n'.format(len(probs_phrases)))
        #for k, v in probs_phrases.items():
            #print(k, ':', v)


    def predict(self, max_stack_backoff, max_phrase_backoff):
        folder_name = 'results/Backoff_%d_%d/' %(max_stack_backoff, max_phrase_backoff)
        predictions_file = folder_name + 'results_%.2f_%d.txt' %(1-self.train_test_split, self.cur_fold) #have identifying name for later use
        utterance_file = 'data/utterances_%.2f_%d.txt' %(1-self.train_test_split, self.cur_fold)

        with open(predictions_file, mode='wt') as f_predictions, open(utterance_file, mode='wt') as f_references:
            for mr_delex, true_utterance in zip(self.mrs_delex_test, self.phrase_sequences_test):
                mr_test_stack = [self.stack_tup2idx[tuple(stack)] for stack in mr_delex]
                #print("Number of stacks= "+str(len(mr_test_stack)))
                #perm = permutations(mr_test_stack)
                #print(' '.join(true_utterance))
                f_references.write(' '.join(true_utterance))
                f_references.write('\n')

                self.prob_prune = 0.0
                prob_stacks, path_stacks = self.__infer_stack_seq(mr_test_stack, 1.0, [], max_depth=max_stack_backoff)
                # the phrase inference would not work with a single-stack sequence
                if len(path_stacks) == 1:
                    path_stacks = []

                #all_stacks_paths=[ self.__infer_stack_seq(x, 0, 1.0, []) for x in perm]
                #print all_stacks_paths
                       
                stack_seq = [self.stack_idx2tup[stack_idx] for stack_idx in path_stacks]
                print('S* =', stack_seq)
                print('-> Prob =', prob_stacks)
                print('-> Recursive calls =', self.stack_recursion_ctr)
                self.stack_recursion_ctr = 0

                # print the results to a file
                
                #predict.write('S* = ' + str(stack_seq))
                #predict.write('\n')
                #predict.write('-> Prob = ' + str(prob_stacks))
                #predict.write('\n')

                self.prob_prune = 0.0
                prob_phrases, path_phrases = self.__infer_phrase_seq(path_stacks, 0, 1.0, [], max_depth=max_phrase_backoff)

                phrase_seq = [self.phrase_idx2word[phrase_idx] for phrase_idx in path_phrases]
                print('R* =', phrase_seq)
                print('-> Prob =', prob_phrases)
                print('-> Recursive calls =', self.phrase_recursion_ctr)
                self.phrase_recursion_ctr = 0
                print()

                # print the results to a file
                #predict.write('R* = ' + str(phrase_seq))
                #predict.write('\n')
                #predict.write('-> Prob = ' + str(prob_phrases))
                #predict.write('\n\n')

                f_predictions.write(' '.join(phrase_seq))
                f_predictions.write('\n')


    def reset(self): #used in cross validation to reset data training/inference without reseting the data loading. 
        # reset probabilities
        self.probs_stacks = {}      
        self.probs_stacks1 = {}     
        self.probs_stacks2 = {}     
        self.probs_phrases = {}     
        self.probs_phrases1 = {}    
        self.probs_phrases2 = {}    
        self.probs_phrases3 = {}    
        self.probs_phrases4 = {}    
        self.probs_phrases5 = {}    
        self.probs_phrases6 = {}    



    # ---- PRIVATE METHODS ----

    def __create_mapping_stacks(self, stack_sequences): #mapping of stacks 
        mapping = dict()
        mapping_rev = []
        counter = 0
    
        for seq in stack_sequences:
            for stack in seq:
                key = tuple(stack)
                if key not in mapping:
                    mapping[key] = counter
                    mapping_rev.append(key)
                    counter += 1
    
        return (mapping, mapping_rev)


    def __create_mapping_heads_tails(self, stack_sequences): #mapping of heads 
        mapping_heads = dict()
        mapping_rev_heads = []
        mapping_tails = dict()
        mapping_rev_tails = []
        counter_head = 0
        counter_tail = 0
    
        for seq in stack_sequences:
            for stack in seq:
                key_head = tuple(stack[-1])
                if key_head not in mapping_heads:
                    mapping_heads[key_head] = counter_head
                    mapping_rev_heads.append(key_head)
                    counter_head += 1

                key_tail = tuple(stack[:-1])
                if key_tail not in mapping_tails:
                    mapping_tails[key_tail] = counter_tail
                    mapping_rev_tails.append(key_tail)
                    counter_tail += 1
    
        return (mapping_heads, mapping_rev_heads, mapping_tails, mapping_rev_tails)


    def __create_mapping_phrases(self, phrase_sequences): #mapping of stacks 
        mapping = dict()
        mapping_rev = []
        counter = 0
        
        for seq in phrase_sequences:
            for phrase in seq:
                key = phrase
                if key not in mapping:
                    mapping[key] = counter
                    mapping_rev.append(key)
                    counter += 1
                    
        return (mapping, mapping_rev)


    def __calculate_phrase_config_prob(self, prob_table, phrase_subseq_n, tail_subseq_n, head_subseq_n, phrase_subseq_d, tail_subseq_d, head_subseq_d):
        key = tuple([ self.phrase_word2idx.get(phrase) for phrase in phrase_subseq_n ] \
                + self.__tails_to_idxs(tail_subseq_n) \
                + self.__heads_to_idxs(head_subseq_n))
        
        if key not in prob_table:
            value = self.__get_freq_phrases(phrase_subseq_n, tail_subseq_n, head_subseq_n) \
                    / self.__get_freq_phrases(phrase_subseq_d, tail_subseq_d, head_subseq_d)
            prob_table[key] = value


    def __get_freq_stacks(self, subseq):
        subseq_len = len(subseq)
        if subseq_len > 0:
            offset = 1
        else:
            offset = 0      # to prevent an empty list as argument from causing an overflow in the inner loop
            
        freq = 0
        for seq in self.stack_sequences_train:
            for idx in range(len(seq) - subseq_len + offset):
                if seq[idx : idx + subseq_len] == subseq:
                    freq += 1

        return float(freq) #to divide later


    def __get_freq_phrases(self, subseq_phrases, subseq_tails, subseq_heads):
        # set offsets from the beginning and the end of the sequence
        offset_beg = 1
        offset_end = 1
        if (subseq_phrases[0] is None) and (subseq_tails[0] is None) and (subseq_heads[0] is None):
            offset_beg = 0
        if (subseq_phrases[-1] is None) and (subseq_tails[-1] is None) and (subseq_heads[-1] is None):
            offset_end = 0

        freq = 0
        for seq_phrases, seq_stacks in zip(self.phrase_sequences_train, self.stack_sequences_train):
            for i in range(offset_beg, len(seq_phrases) - offset_end):
                if self.__match_triple(subseq_phrases, seq_phrases, i) \
                        and self.__match_triple(subseq_tails, [stack[:-1] for stack in seq_stacks], i) \
                        and self.__match_triple(subseq_heads, [stack[-1] for stack in seq_stacks], i):
                    freq += 1

        return float(freq) #to divide later


    # determine if the given stack is mandatory
    def __is_mand(self, stack_idx):
        return (len(self.stack_idx2tup[stack_idx]) == 3)


    # extract a triple of phrases from a sequence of phrases based on the given middle index
    def __phrase_seq_to_triple(self, phrase_seq, idx_middle):
        triple = []
        for idx in range(idx_middle - 1, idx_middle + 2):
            if idx >= 0 and idx < len(phrase_seq):
                triple.append(phrase_seq[idx])
            else:
                triple.append(None)

        return triple


    # extract a triple of stacks from a sequence of stacks based on the given middle index
    def __stack_seq_to_triple(self, stack_seq, idx_middle):
        triple = []
        for idx in range(idx_middle - 1, idx_middle + 2):
            if idx >= 0 and idx < len(stack_seq):
                triple.append(stack_seq[idx])
            else:
                triple.append(None)

        return triple


    # extract a triple of tails from a sequence of stacks based on the given middle index
    def __stack_seq_to_triple_tails(self, stack_seq, idx_middle):
        triple = []
        for idx in range(idx_middle - 1, idx_middle + 2):
            if idx >= 0 and idx < len(stack_seq):
                triple.append(stack_seq[idx][:-1])
            else:
                triple.append(None)

        return triple


    # extract a triple of heads from a sequence of stacks based on the given middle index
    def __stack_seq_to_triple_heads(self, stack_seq, idx_middle):
        triple = []
        for idx in range(idx_middle - 1, idx_middle + 2):
            if idx >= 0 and idx < len(stack_seq):
                triple.append(stack_seq[idx][-1])
            else:
                triple.append(None)

        return triple


    # convert a sequence of tails to a sequence of corresponding indices
    def __tails_to_idxs(self, tail_seq):
        idxs = []
        for tail in tail_seq:
            if tail is not None:
                idxs.append(self.tail_tup2idx.get(tuple(tail)))
            else:
                idxs.append(None)

        return idxs


    # convert a sequence of heads to a sequence of corresponding indices
    def __heads_to_idxs(self, head_seq):
        idxs = []
        for head in head_seq:
            if head is not None:
                idxs.append(self.head_tup2idx.get(tuple(head)))
            else:
                idxs.append(None)

        return idxs


    # determine if a triple matches a sequence around the given index
    def __match_triple(self, triple, seq, idx_seq):
        for i in range(len(triple)):
            if triple[i] is not None and seq[idx_seq - 1 + i] != triple[i]:
                return False

        return True


    # convert a stack index to the corresponding tail index
    def __stack_idx_to_tail_idx(self, stack_idx):
        stack_tup = self.stack_idx2tup[stack_idx]
        tail_idx = self.tail_tup2idx[tuple(stack_tup[:-1])]

        return tail_idx


    # convert a stack index to the corresponding head index
    def __stack_idx_to_head_idx(self, stack_idx):
        stack_tup = self.stack_idx2tup[stack_idx]
        head_idx = self.head_tup2idx[tuple(stack_tup[-1])]

        return head_idx


    # get the most likely sequence of stacks (S*), given sequence of mandatory stacks (Sm)
    def __infer_stack_seq(self, Sm, prob_from_root, path_from_root, max_depth, backoff_depth=0):
        if backoff_depth > 2:
            raise IndexError('Stack backoff depth greater than 2!')

        # stop the recursion when all the mandatory stacks have been included
        if len(Sm) <= 0:
            return (prob_from_root, path_from_root)

        # prune branches with a lower prob. than the best known prob. to leaf
        if prob_from_root < self.prob_prune:
            return (prob_from_root, path_from_root)

        # recursion threshold
        self.stack_recursion_ctr += 1
        if self.stack_recursion_ctr > 1000000:
            return (prob_from_root, path_from_root)


        best_prob = 0.0
        best_path = []

        # select the configuration probability dictionary according to the backoff depth
        level_dict = {}
        if backoff_depth == 0:
            level_dict = self.probs_stacks
        elif backoff_depth == 1:
            level_dict = self.probs_stacks1
        elif backoff_depth == 2:
            level_dict = self.probs_stacks2

        # try all known configurations of stacks (those encountered in training)
        for stack_config, config_prob in level_dict.items():
            prob_to_leaf = 0.0
            path_to_leaf = []

            # add a filler stack
            if not self.__is_mand(stack_config[-1]):
                # at the beginning
                if len(path_from_root) == 0:
                    if len(stack_config) == 1:
                        prob_to_leaf, path_to_leaf = self.__infer_stack_seq(Sm, prob_from_root * config_prob, path_from_root + [stack_config[-1]], max_depth)
                # directly after a mandatory stack
                elif self.__is_mand(path_from_root[-1]):
                    if backoff_depth < 2:
                        # s_{t-1}
                        if (len(stack_config) > 1 and path_from_root[-1] == stack_config[-2]):
                            if (len(path_from_root) > 1) and backoff_depth < 1:
                                # s_{t-2}
                                if (len(stack_config) > 2 and path_from_root[-2] == stack_config[-3]):
                                    prob_to_leaf, path_to_leaf = self.__infer_stack_seq(Sm, prob_from_root * config_prob, path_from_root + [stack_config[-1]], max_depth)
                            elif len(stack_config) == 2:
                                prob_to_leaf, path_to_leaf = self.__infer_stack_seq(Sm, prob_from_root * config_prob, path_from_root + [stack_config[-1]], max_depth)
                    elif len(stack_config) == 1:
                        prob_to_leaf, path_to_leaf = self.__infer_stack_seq(Sm, prob_from_root * config_prob, path_from_root + [stack_config[-1]], max_depth) 
            # add a mandatory stack (allow anytime)
            # s_{t}
            #elif (stack_config[-1] == Sm[0]):       # fixed ordering
            elif (stack_config[-1] in Sm):          # arbitrary ordering
                if (len(path_from_root) > 0) and backoff_depth < 2:
                    # s_{t-1}
                    if (len(stack_config) > 1 and path_from_root[-1] == stack_config[-2]):
                        if (len(path_from_root) > 1) and backoff_depth < 1:
                            # s_{t-2}
                            if (len(stack_config) > 2 and path_from_root[-2] == stack_config[-3]):
                                Sm_rem = Sm[:]
                                Sm_rem.remove(stack_config[-1])
                                prob_to_leaf, path_to_leaf = self.__infer_stack_seq(Sm_rem, prob_from_root * config_prob, path_from_root + [stack_config[-1]], max_depth)
                        elif len(stack_config) == 2:
                            Sm_rem = Sm[:]
                            Sm_rem.remove(stack_config[-1])
                            prob_to_leaf, path_to_leaf = self.__infer_stack_seq(Sm_rem, prob_from_root * config_prob, path_from_root + [stack_config[-1]], max_depth)
                elif len(stack_config) == 1:
                    Sm_rem = Sm[:]
                    Sm_rem.remove(stack_config[-1])
                    prob_to_leaf, path_to_leaf = self.__infer_stack_seq(Sm_rem, prob_from_root * config_prob, path_from_root + [stack_config[-1]], max_depth)

            if len(path_to_leaf) > 0 and prob_to_leaf > best_prob:
                best_prob = prob_to_leaf
                best_path = path_to_leaf

                if best_prob > self.prob_prune:
                    self.prob_prune = best_prob

        # DEBUG PRINT
        #print(best_path)
        #print(best_prob)
        #print()

        if backoff_depth < max_depth and best_prob == 0.0:
            #print('[Backing off from level {0}...]'.format(backoff_depth))
            best_prob, best_path = self.__infer_stack_seq(Sm, prob_from_root, path_from_root, max_depth, backoff_depth + 1)

        return best_prob, best_path


    # get the most likely sequence of phrases (R*), given sequence of stacks (S*)
    def __infer_phrase_seq(self, S, t, prob_from_root, path_from_root, max_depth, backoff_depth=0):
        if backoff_depth > 6:
            raise IndexError('Phrase backoff depth greater than 4!')

        # stop the recursion when all the stacks have been realized
        if t >= len(S):
            return (prob_from_root, path_from_root)

        # prune branches with a lower prob. than the best known prob. to leaf
        if prob_from_root < self.prob_prune:
            return (prob_from_root, path_from_root)

        # recursion threshold
        self.phrase_recursion_ctr += 1
        if self.phrase_recursion_ctr > 1000000:
            return (prob_from_root, path_from_root)


        best_prob = 0.0
        best_path = []

        if backoff_depth == 0:
            # try all known configurations of phrases and stacks (those encountered in training)
            for config, config_prob in self.probs_phrases.items():
                prob_to_leaf = 0.0
                path_to_leaf = []
            
                # match beginning (r_{t} | l_{t}, l_{t + 1}, h_{t}, h_{t + 1})
                if t == 0:
                    if config[0] is None and \
                            config[3] is None and config[4] == self.__stack_idx_to_tail_idx(S[t]) and config[5] == self.__stack_idx_to_tail_idx(S[t + 1]) and \
                            config[6] is None and config[7] == self.__stack_idx_to_head_idx(S[t]) and config[8] == self.__stack_idx_to_head_idx(S[t + 1]):
                        prob_to_leaf, path_to_leaf = self.__infer_phrase_seq(S, t + 1, prob_from_root * config_prob, path_from_root + [config[1]], max_depth)
                # match end (r_{t} | r_{t - 1}, l_{t - 1}, l_{t}, h_{t - 1}, h_{t})
                elif t == len(S) - 1:
                    if config[0] == path_from_root[-1] and \
                            config[3] == self.__stack_idx_to_tail_idx(S[t - 1]) and config[4] == self.__stack_idx_to_tail_idx(S[t]) and config[5] is None and \
                            config[6] == self.__stack_idx_to_head_idx(S[t - 1]) and config[7] == self.__stack_idx_to_head_idx(S[t]) and config[8] is None:
                        prob_to_leaf, path_to_leaf = self.__infer_phrase_seq(S, t + 1, prob_from_root * config_prob, path_from_root + [config[1]], max_depth)
                # match middle (r_{t} | r_{t - 1}, l_{t - 1}, l_{t}, l_{t + 1}, h_{t - 1}, h_{t}, h_{t + 1})
                else:
                    if config[0] == path_from_root[-1] and \
                            config[3] == self.__stack_idx_to_tail_idx(S[t - 1]) and config[4] == self.__stack_idx_to_tail_idx(S[t]) and config[5] == self.__stack_idx_to_tail_idx(S[t + 1]) and \
                            config[6] == self.__stack_idx_to_head_idx(S[t - 1]) and config[7] == self.__stack_idx_to_head_idx(S[t]) and config[8] == self.__stack_idx_to_head_idx(S[t + 1]):
                        prob_to_leaf, path_to_leaf = self.__infer_phrase_seq(S, t + 1, prob_from_root * config_prob, path_from_root + [config[1]], max_depth)

                if len(path_to_leaf) > 0 and prob_to_leaf > best_prob:
                    best_prob = prob_to_leaf
                    best_path = path_to_leaf

                    if best_prob > self.prob_prune:
                        self.prob_prune = best_prob
        
        elif backoff_depth == 1:
            # try all known configurations of phrases and stacks (those encountered in training)
            for config, config_prob in self.probs_phrases1.items():
                #prob_to_leaf = 0.0
                #path_to_leaf = []
            
                ## match beginning (r_{t} | l_{t}, l_{t + 1}, h_{t})
                #if t == 0:
                #    if config[0] is None and config[2] is None and \
                #            config[3] is None and config[4] == self.__stack_idx_to_tail_idx(S[t]) and config[5] == self.__stack_idx_to_tail_idx(S[t + 1]) and \
                #            config[6] is None and config[7] == self.__stack_idx_to_head_idx(S[t]) and config[8] is None:
                #        prob_to_leaf, path_to_leaf = self.__infer_phrase_seq(S, t + 1, prob_from_root * config_prob, path_from_root + [config[1]], desired_depth)
                ## match end (r_{t} | r_{t - 1}, l_{t - 1}, l_{t}, h_{t - 1}, h_{t})
                #elif t == len(S) - 1:
                #    if config[0] == path_from_root[-1] and config[2] is None and \
                #            config[3] == self.__stack_idx_to_tail_idx(S[t - 1]) and config[4] == self.__stack_idx_to_tail_idx(S[t]) and config[5] is None and \
                #            config[6] == self.__stack_idx_to_head_idx(S[t - 1]) and config[7] == self.__stack_idx_to_head_idx(S[t]) and config[8] is None:
                #        prob_to_leaf, path_to_leaf = self.__infer_phrase_seq(S, t + 1, prob_from_root * config_prob, path_from_root + [config[1]], desired_depth)
                ## match middle (r_{t} | r_{t - 1}, l_{t - 1}, l_{t}, l_{t + 1}, h_{t - 1}, h_{t})
                #else:
                #    if config[0] == path_from_root[-1] and config[2] is None and \
                #            config[3] == self.__stack_idx_to_tail_idx(S[t - 1]) and config[4] == self.__stack_idx_to_tail_idx(S[t]) and config[5] == self.__stack_idx_to_tail_idx(S[t + 1]) and \
                #            config[6] == self.__stack_idx_to_head_idx(S[t - 1]) and config[7] == self.__stack_idx_to_head_idx(S[t]) and config[8] is None:
                #        prob_to_leaf, path_to_leaf = self.__infer_phrase_seq(S, t + 1, prob_from_root * config_prob, path_from_root + [config[1]], desired_depth)

                #if len(path_to_leaf) > 0 and prob_to_leaf > best_prob:
                #    best_prob = prob_to_leaf
                #    best_path = path_to_leaf

                #    if best_prob > self.prob_prune:
                #        self.prob_prune = best_prob


                prob_to_leaf = 0.0
                path_to_leaf = []
            
                # match beginning (r_{t} | l_{t}, l_{t + 1}, h_{t})
                if t == 0:
                    if config[0] is None and \
                            config[3] is None and config[4] == self.__stack_idx_to_tail_idx(S[t]) and config[5] == self.__stack_idx_to_tail_idx(S[t + 1]) and \
                            config[7] == self.__stack_idx_to_head_idx(S[t]):
                        prob_to_leaf, path_to_leaf = self.__infer_phrase_seq(S, t + 1, prob_from_root * config_prob, path_from_root + [config[1]], max_depth)
                # match end (r_{t} | r_{t - 1}, l_{t - 1}, l_{t}, h_{t})
                elif t == len(S) - 1:
                    if config[0] == path_from_root[-1] and \
                            config[3] == self.__stack_idx_to_tail_idx(S[t - 1]) and config[4] == self.__stack_idx_to_tail_idx(S[t]) and config[5] is None and \
                            config[7] == self.__stack_idx_to_head_idx(S[t]):
                        prob_to_leaf, path_to_leaf = self.__infer_phrase_seq(S, t + 1, prob_from_root * config_prob, path_from_root + [config[1]], max_depth)
                # match middle (r_{t} | r_{t - 1}, l_{t - 1}, l_{t}, l_{t + 1}, h_{t})
                else:
                    if config[0] == path_from_root[-1] and \
                            config[3] == self.__stack_idx_to_tail_idx(S[t - 1]) and config[4] == self.__stack_idx_to_tail_idx(S[t]) and config[5] == self.__stack_idx_to_tail_idx(S[t + 1]) and \
                            config[7] == self.__stack_idx_to_head_idx(S[t]):
                        prob_to_leaf, path_to_leaf = self.__infer_phrase_seq(S, t + 1, prob_from_root * config_prob, path_from_root + [config[1]], max_depth)

                if len(path_to_leaf) > 0 and prob_to_leaf > best_prob:
                    best_prob = prob_to_leaf
                    best_path = path_to_leaf

                    if best_prob > self.prob_prune:
                        self.prob_prune = best_prob

                
                #prob_to_leaf = 0.0
                #path_to_leaf = []
            
                ## match beginning (r_{t} | l_{t}, l_{t + 1}, h_{t}, h_{t + 1})
                #if t == 0:
                #    if config[0] is None and config[2] is None and \
                #            config[3] is None and config[4] == self.__stack_idx_to_tail_idx(S[t]) and config[5] == self.__stack_idx_to_tail_idx(S[t + 1]) and \
                #            config[6] is None and config[7] == self.__stack_idx_to_head_idx(S[t]) and config[8] == self.__stack_idx_to_head_idx(S[t + 1]):
                #        prob_to_leaf, path_to_leaf = self.__infer_phrase_seq(S, t + 1, prob_from_root * config_prob, path_from_root + [config[1]], desired_depth)
                ## match end (r_{t} | r_{t - 1}, l_{t - 1}, l_{t}, h_{t})
                #elif t == len(S) - 1:
                #    if config[0] == path_from_root[-1] and config[2] is None and \
                #            config[3] == self.__stack_idx_to_tail_idx(S[t - 1]) and config[4] == self.__stack_idx_to_tail_idx(S[t]) and config[5] is None and \
                #            config[6] is None and config[7] == self.__stack_idx_to_head_idx(S[t]) and config[8] is None:
                #        prob_to_leaf, path_to_leaf = self.__infer_phrase_seq(S, t + 1, prob_from_root * config_prob, path_from_root + [config[1]], desired_depth)
                ## match middle (r_{t} | r_{t - 1}, l_{t - 1}, l_{t}, l_{t + 1}, h_{t}, h_{t + 1})
                #else:
                #    if config[0] == path_from_root[-1] and config[2] is None and \
                #            config[3] == self.__stack_idx_to_tail_idx(S[t - 1]) and config[4] == self.__stack_idx_to_tail_idx(S[t]) and config[5] == self.__stack_idx_to_tail_idx(S[t + 1]) and \
                #            config[6] is None and config[7] == self.__stack_idx_to_head_idx(S[t]) and config[8] == self.__stack_idx_to_head_idx(S[t + 1]):
                #        prob_to_leaf, path_to_leaf = self.__infer_phrase_seq(S, t + 1, prob_from_root * config_prob, path_from_root + [config[1]], desired_depth)

                #if len(path_to_leaf) > 0 and prob_to_leaf > best_prob:
                #    best_prob = prob_to_leaf
                #    best_path = path_to_leaf

                #    if best_prob > self.prob_prune:
                #        self.prob_prune = best_prob
        
        elif backoff_depth == 2:
            # try all known configurations of phrases and stacks (those encountered in training)
            for config, config_prob in self.probs_phrases2.items():
                prob_to_leaf = 0.0
                path_to_leaf = []
            
                # match beginning (r_{t} | l_{t + 1}, h_{t})
                if t == 0:
                    if config[0] is None and \
                            config[3] is None and config[5] == self.__stack_idx_to_tail_idx(S[t + 1]) and \
                            config[7] == self.__stack_idx_to_head_idx(S[t]):
                        prob_to_leaf, path_to_leaf = self.__infer_phrase_seq(S, t + 1, prob_from_root * config_prob, path_from_root + [config[1]], max_depth)
                # match end (r_{t} | r_{t - 1}, l_{t - 1}, h_{t})
                elif t == len(S) - 1:
                    if config[0] == path_from_root[-1] and \
                            config[3] == self.__stack_idx_to_tail_idx(S[t - 1]) and config[5] is None and \
                            config[7] == self.__stack_idx_to_head_idx(S[t]):
                        prob_to_leaf, path_to_leaf = self.__infer_phrase_seq(S, t + 1, prob_from_root * config_prob, path_from_root + [config[1]], max_depth)
                # match middle (r_{t} | r_{t - 1}, l_{t - 1}, l_{t + 1}, h_{t})
                else:
                    if config[0] == path_from_root[-1] and \
                            config[3] == self.__stack_idx_to_tail_idx(S[t - 1]) and config[5] == self.__stack_idx_to_tail_idx(S[t + 1]) and \
                            config[7] == self.__stack_idx_to_head_idx(S[t]):
                        prob_to_leaf, path_to_leaf = self.__infer_phrase_seq(S, t + 1, prob_from_root * config_prob, path_from_root + [config[1]], max_depth)

                if len(path_to_leaf) > 0 and prob_to_leaf > best_prob:
                    best_prob = prob_to_leaf
                    best_path = path_to_leaf

                    if best_prob > self.prob_prune:
                        self.prob_prune = best_prob

        elif backoff_depth == 3:
            # try all known configurations of phrases and stacks (those encountered in training)
            for config, config_prob in self.probs_phrases3.items():
                prob_to_leaf = 0.0
                path_to_leaf = []
            
                # match beginning (r_{t} | h_{t})
                if t == 0:
                    if config[0] is None and \
                            config[3] is None and config[5] is None and \
                            config[7] == self.__stack_idx_to_head_idx(S[t]):
                        prob_to_leaf, path_to_leaf = self.__infer_phrase_seq(S, t + 1, prob_from_root * config_prob, path_from_root + [config[1]], max_depth)
                # match end (r_{t} | r_{t - 1}, l_{t - 1}, h_{t})
                elif t == len(S) - 1:
                    if config[0] == path_from_root[-1] and \
                            config[3] == self.__stack_idx_to_tail_idx(S[t - 1]) and config[5] is None and \
                            config[7] == self.__stack_idx_to_head_idx(S[t]):
                        prob_to_leaf, path_to_leaf = self.__infer_phrase_seq(S, t + 1, prob_from_root * config_prob, path_from_root + [config[1]], max_depth)
                # match middle (r_{t} | r_{t - 1}, l_{t - 1}, h_{t})
                else:
                    if config[0] == path_from_root[-1] and \
                            config[3] == self.__stack_idx_to_tail_idx(S[t - 1]) and config[5] is None and \
                            config[7] == self.__stack_idx_to_head_idx(S[t]):
                        prob_to_leaf, path_to_leaf = self.__infer_phrase_seq(S, t + 1, prob_from_root * config_prob, path_from_root + [config[1]], max_depth)

                if len(path_to_leaf) > 0 and prob_to_leaf > best_prob:
                    best_prob = prob_to_leaf
                    best_path = path_to_leaf

                    if best_prob > self.prob_prune:
                        self.prob_prune = best_prob


                prob_to_leaf = 0.0
                path_to_leaf = []
            
                # match beginning (r_{t} | l_{t + 1}, h_{t})
                if t == 0:
                    if config[0] is None and \
                            config[3] is None and config[5] == self.__stack_idx_to_tail_idx(S[t + 1]) and \
                            config[7] == self.__stack_idx_to_head_idx(S[t]):
                        prob_to_leaf, path_to_leaf = self.__infer_phrase_seq(S, t + 1, prob_from_root * config_prob, path_from_root + [config[1]], max_depth)
                # match end (r_{t} | r_{t - 1}, h_{t})
                elif t == len(S) - 1:
                    if config[0] == path_from_root[-1] and \
                            config[3] is None and config[5] is None and \
                            config[7] == self.__stack_idx_to_head_idx(S[t]):
                        prob_to_leaf, path_to_leaf = self.__infer_phrase_seq(S, t + 1, prob_from_root * config_prob, path_from_root + [config[1]], max_depth)
                # match middle (r_{t} | r_{t - 1}, l_{t + 1}, h_{t})
                else:
                    if config[0] == path_from_root[-1] and \
                            config[3] is None and config[5] == self.__stack_idx_to_tail_idx(S[t + 1]) and \
                            config[7] == self.__stack_idx_to_head_idx(S[t]):
                        prob_to_leaf, path_to_leaf = self.__infer_phrase_seq(S, t + 1, prob_from_root * config_prob, path_from_root + [config[1]], max_depth)

                if len(path_to_leaf) > 0 and prob_to_leaf > best_prob:
                    best_prob = prob_to_leaf
                    best_path = path_to_leaf

                    if best_prob > self.prob_prune:
                        self.prob_prune = best_prob


                prob_to_leaf = 0.0
                path_to_leaf = []
            
                # match beginning (r_{t} | l_{t - 1}, h_{t})
                if t == 0:
                    if config[0] is None and \
                            config[3] is None and config[5] == self.__stack_idx_to_tail_idx(S[t + 1]) and \
                            config[7] == self.__stack_idx_to_head_idx(S[t]):
                        prob_to_leaf, path_to_leaf = self.__infer_phrase_seq(S, t + 1, prob_from_root * config_prob, path_from_root + [config[1]], max_depth)
                # match end (r_{t} | l_{t - 1}, h_{t})
                elif t == len(S) - 1:
                    if config[0] is None and \
                            config[3] == self.__stack_idx_to_tail_idx(S[t - 1]) and config[5] is None and \
                            config[7] == self.__stack_idx_to_head_idx(S[t]):
                        prob_to_leaf, path_to_leaf = self.__infer_phrase_seq(S, t + 1, prob_from_root * config_prob, path_from_root + [config[1]], max_depth)
                # match middle (r_{t} | l_{t - 1}, l_{t + 1}, h_{t})
                else:
                    if config[0] is None and \
                            config[3] == self.__stack_idx_to_tail_idx(S[t - 1]) and config[5] == self.__stack_idx_to_tail_idx(S[t + 1]) and \
                            config[7] == self.__stack_idx_to_head_idx(S[t]):
                        prob_to_leaf, path_to_leaf = self.__infer_phrase_seq(S, t + 1, prob_from_root * config_prob, path_from_root + [config[1]], max_depth)

                if len(path_to_leaf) > 0 and prob_to_leaf > best_prob:
                    best_prob = prob_to_leaf
                    best_path = path_to_leaf

                    if best_prob > self.prob_prune:
                        self.prob_prune = best_prob
        
        elif backoff_depth == 4:
            # try all known configurations of phrases and stacks (those encountered in training)
            for config, config_prob in self.probs_phrases2.items():
                prob_to_leaf = 0.0
                path_to_leaf = []
            
                # match beginning (r_{t} | h_{t})
                if t == 0:
                    if config[0] is None and \
                            config[3] is None and config[5] is None and \
                            config[7] == self.__stack_idx_to_head_idx(S[t]):
                        prob_to_leaf, path_to_leaf = self.__infer_phrase_seq(S, t + 1, prob_from_root * config_prob, path_from_root + [config[1]], max_depth)
                # match end (r_{t} | r_{t - 1}, h_{t})
                elif t == len(S) - 1:
                    if config[0] == path_from_root[-1] and \
                            config[3] is None and config[5] is None and \
                            config[7] == self.__stack_idx_to_head_idx(S[t]):
                        prob_to_leaf, path_to_leaf = self.__infer_phrase_seq(S, t + 1, prob_from_root * config_prob, path_from_root + [config[1]], max_depth)
                # match middle (r_{t} | r_{t - 1}, h_{t})
                else:
                    if config[0] == path_from_root[-1] and \
                            config[3] is None and config[5] is None and \
                            config[7] == self.__stack_idx_to_head_idx(S[t]):
                        prob_to_leaf, path_to_leaf = self.__infer_phrase_seq(S, t + 1, prob_from_root * config_prob, path_from_root + [config[1]], max_depth)

                if len(path_to_leaf) > 0 and prob_to_leaf > best_prob:
                    best_prob = prob_to_leaf
                    best_path = path_to_leaf

                    if best_prob > self.prob_prune:
                        self.prob_prune = best_prob
                        

                prob_to_leaf = 0.0
                path_to_leaf = []
            
                # match beginning (r_{t} | h_{t})
                if t == 0:
                    if config[0] is None and \
                            config[3] is None and config[5] == self.__stack_idx_to_tail_idx(S[t + 1]) and \
                            config[7] == self.__stack_idx_to_head_idx(S[t]):
                        prob_to_leaf, path_to_leaf = self.__infer_phrase_seq(S, t + 1, prob_from_root * config_prob, path_from_root + [config[1]], max_depth)
                # match end (r_{t} | l_{t - 1}, h_{t})
                elif t == len(S) - 1:
                    if config[0] is None and \
                            config[3] == self.__stack_idx_to_tail_idx(S[t - 1]) and config[5] is None and \
                            config[7] == self.__stack_idx_to_head_idx(S[t]):
                        prob_to_leaf, path_to_leaf = self.__infer_phrase_seq(S, t + 1, prob_from_root * config_prob, path_from_root + [config[1]], max_depth)
                # match middle (r_{t} | l_{t - 1}, h_{t})
                else:
                    if config[0] is None and \
                            config[3] == self.__stack_idx_to_tail_idx(S[t - 1]) and config[5] is None and \
                            config[7] == self.__stack_idx_to_head_idx(S[t]):
                        prob_to_leaf, path_to_leaf = self.__infer_phrase_seq(S, t + 1, prob_from_root * config_prob, path_from_root + [config[1]], max_depth)

                if len(path_to_leaf) > 0 and prob_to_leaf > best_prob:
                    best_prob = prob_to_leaf
                    best_path = path_to_leaf

                    if best_prob > self.prob_prune:
                        self.prob_prune = best_prob
                        

                prob_to_leaf = 0.0
                path_to_leaf = []
            
                # match beginning (r_{t} | l_{t + 1}, h_{t})
                if t == 0:
                    if config[0] is None and \
                            config[3] is None and config[5] == self.__stack_idx_to_tail_idx(S[t + 1]) and \
                            config[7] == self.__stack_idx_to_head_idx(S[t]):
                        prob_to_leaf, path_to_leaf = self.__infer_phrase_seq(S, t + 1, prob_from_root * config_prob, path_from_root + [config[1]], max_depth)
                # match end (r_{t} | h_{t})
                elif t == len(S) - 1:
                    if config[0] is None and \
                            config[3] is None and config[5] is None and \
                            config[7] == self.__stack_idx_to_head_idx(S[t]):
                        prob_to_leaf, path_to_leaf = self.__infer_phrase_seq(S, t + 1, prob_from_root * config_prob, path_from_root + [config[1]], max_depth)
                # match middle (r_{t} | l_{t + 1}, h_{t})
                else:
                    if config[0] is None and \
                            config[3] is None and config[5] == self.__stack_idx_to_tail_idx(S[t + 1]) and \
                            config[7] == self.__stack_idx_to_head_idx(S[t]):
                        prob_to_leaf, path_to_leaf = self.__infer_phrase_seq(S, t + 1, prob_from_root * config_prob, path_from_root + [config[1]], max_depth)

                if len(path_to_leaf) > 0 and prob_to_leaf > best_prob:
                    best_prob = prob_to_leaf
                    best_path = path_to_leaf

                    if best_prob > self.prob_prune:
                        self.prob_prune = best_prob
        

        elif backoff_depth == 5:
            # try all known configurations of phrases and stacks (those encountered in training)
            for config, config_prob in self.probs_phrases5.items():
                prob_to_leaf = 0.0
                path_to_leaf = []
            
                # match (r_{t} | h_{t})
                if config[7] == self.__stack_idx_to_head_idx(S[t]):
                    prob_to_leaf, path_to_leaf = self.__infer_phrase_seq(S, t + 1, prob_from_root * config_prob, path_from_root + [config[1]], max_depth)

                if len(path_to_leaf) > 0 and prob_to_leaf > best_prob:
                    best_prob = prob_to_leaf
                    best_path = path_to_leaf

                    if best_prob > self.prob_prune:
                        self.prob_prune = best_prob
        
        elif backoff_depth == 6:
            # try all known configurations of phrases and stacks (those encountered in training)
            for config, config_prob in self.probs_phrases6.items():
                prob_to_leaf = 0.0
                path_to_leaf = []
            
                # match (r_{t})
                prob_to_leaf, path_to_leaf = self.__infer_phrase_seq(S, t + 1, prob_from_root * config_prob, path_from_root + [config[1]], max_depth)

                if len(path_to_leaf) > 0 and prob_to_leaf > best_prob:
                    best_prob = prob_to_leaf
                    best_path = path_to_leaf

                    if best_prob > self.prob_prune:
                        self.prob_prune = best_prob

        # DEBUG PRINT
        #print(best_path)
        #print(best_prob)
        #print()

        if backoff_depth < max_depth and best_prob == 0.0:
            #print('[Backing off from level {0}...]'.format(backoff_depth))
            best_prob, best_path = self.__infer_phrase_seq(S, t, prob_from_root, path_from_root, max_depth, backoff_depth + 1)

        return best_prob, best_path
    


def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)

def record(s, switch=True): #prints a string s. if switch is true then also record to logfile
    print(s)
    if switch:
        with open ("results/log_file.txt", mode='a') as logfile:
            logfile.write(s)
            logfile.write('\n')


def cross_validate(stack_backoffs, phrase_backoffs, split_points, num_folds, path_to_data):

    # initialize the matrix of BLEU scores and the matrix of METEOR scores
    bleu_scores = [ [ [0] * len(split_points) for _ in range(max(phrase_backoffs) + 1) ] for _ in range(max(stack_backoffs) + 1) ]
    meteor_scores = [ [ [0] * len(split_points) for _ in range(max(phrase_backoffs) + 1) ] for _ in range(max(stack_backoffs) + 1) ]

    for split, split_index in zip(split_points, range(len(split_points))):
        record( '\n ###################### Split %.2f ###################### \n' %(split))

        model = Bagel() #get a fresh model 
        model.set_split(split)

        bleu_split_scores = [ [[]] * (max(phrase_backoffs) + 1) for _ in range(max(stack_backoffs) + 1) ]
        meteor_split_scores = [ [[]] * (max(phrase_backoffs) + 1) for _ in range(max(stack_backoffs) + 1) ]

        for fold in range(num_folds):
            record( '\n \t ###### Fold %d ###### \n' %(fold)) 
            model.set_fold(fold)

            record('\tLoading data...', False)
            model.load_data(path_to_data, to_shuffle=True)

            #reset model from all previous training/inference. Cant get a new instance because it would delete the data loaded
            model.reset()

            for stack_backoff in stack_backoffs:
                record('\t \t --> Stack backoff depth %d ' %(stack_backoff))

                for phrase_backoff in phrase_backoffs: #that way every depth works on the same data
                    record('\t \t --> Phrase backoff depth %d ' %(phrase_backoff))

                    #contruct folder paths
                    folder_name = 'results/Backoff_%d_%d/' %(stack_backoff, phrase_backoff)
                    os.makedirs(os.path.dirname(folder_name), exist_ok=True)
            
 
                    record('\t \t \t Training...', False)
                    model.train(stack_backoff, phrase_backoff)

                    record('\t \t \t Predicting...', False)
                    model.predict(stack_backoff, phrase_backoff)

                    #construct file names
                    predictions_file = folder_name + 'results_%.2f_%d.txt' %(1-split, fold)
                    utterance_file = 'data/utterances_%.2f_%d.txt' %(1-split, fold)

                    bleu_score = blue.get_blue(utterance_file, predictions_file) 
                    #meteor_score = blue.get_meteor(utterance_file, predictions_file)
                    meteor_score = 0

                    bleu_split_scores[stack_backoff][phrase_backoff].append(bleu_score)
                    meteor_split_scores[stack_backoff][phrase_backoff].append(meteor_score)

                    record('\t \t \t BLEU score :=' + str(bleu_score))
                    record('\t \t \t METEOR score :=' + str(meteor_score))

        record('BLEU scores for split (to be averaged):= ' + str(bleu_split_scores))
        record('METEOR scores for split (to be averaged):= ' + str(meteor_split_scores))
        for sb in stack_backoffs:
            for pb in phrase_backoffs:
                bleu_scores[sb][pb][split_index] = mean(bleu_split_scores[sb][pb]) #get mean of each split point, then store at appropriate depth
                meteor_scores[sb][pb][split_index] = mean(meteor_split_scores[sb][pb])
        record('\n \t BLEU so far ' + str(bleu_scores) + '\n', False)

    record('\n \n \n Final BLEU:=' + str(bleu_scores))
    with open('results/BLUE_SCORES.txt', mode='a') as f:
        f.write(str(bleu_scores))

    record('\n \n \n Final METEOR:=' + str(meteor_scores))
    with open('results/METEOR_SCORES.txt', mode='a') as f:
        f.write(str(meteor_scores))

    return bleu_scores


def main():
    path_to_training = 'data/ACL10-inform-training.txt'


    # ---- Single-Setting Testing ----
   
    #max_stack_backoff_depth = 2
    #max_phrase_backoff_depth = 6

    #model = Bagel()

    #print('Loading data...')
    #model.load_data(path_to_training, shuffle=False)

    #print('Training...')
    #model.train(max_stack_backoff_depth, max_phrase_backoff_depth)

    #print('Predicting...')
    #model.predict(max_stack_backoff_depth, max_phrase_backoff_depth)


    # ---- Batch Testing ----

    #depths_list = [0,1,2,3,4]
    #split_points = [0.4,0.5,0.6,0.7,0.8,0.9,0.95]
    #kfold = 10

    ####DUMMY VALUES FOR TESING#######
    stack_backoff_list = [0,1,2]
    phrase_backoff_list = [0,6]
    split_points = [0.4,0.5,0.6,0.7,0.8,0.9,0.95]
    kfold = 5
    ################################

    cross_validate(stack_backoff_list, phrase_backoff_list, split_points, kfold, path_to_training)


if __name__ == '__main__':
    sys.exit(int(main() or 0))
