import numpy as np
import random, os, sys, math, csv, re, collections, string
import torch
import csv
from torch import nn, Tensor
import torch.nn.functional as F
import transformers
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from collections import defaultdict, Counter
import heapq
import matplotlib
from transformers import GPT2TokenizerFast

#-----------------------------------1.1 Tokenize----------------------------------------------------------------------------------------------------------

def GPT2_Tokenizer():
    # Load the tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    # Add pad token using <|endoftext|>
    tokenizer.pad_token = tokenizer.eos_token

    # Option 1: Using `add_special_tokens`
    tokenizer.add_special_tokens({'bos_token': '<s>', 'eos_token': '</s>'})
    return tokenizer

tokenizer = GPT2_Tokenizer()

with open('songs.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)[1:-5]

"""CHECKPOINT 1.1"""

tokenized_lyrics = [tokenizer.convert_ids_to_tokens(tokenizer.encode(row[2])) for row in data]
# Get the first and last tokenized sequences
first_tokens = tokenized_lyrics[0]
last_tokens = tokenized_lyrics[-1]

# Checkpoint output
print("\nCheckpoint 1.1:\n")
print("First: \n", first_tokens, "\n")
print("Last: \n", last_tokens)

#---------------------------------------1.2 Smoothed Trigram Language Model------------------------------------------------------------------------------

from collections import defaultdict, Counter
from ordered_set import OrderedSet

class TrigramLM:
    def __init__(self):
        self.unigram_count = Counter()
        self.bigram_count = defaultdict(Counter)
        self.trigram_count = defaultdict(Counter)
        self.total_unigrams_count = 0
        self.oov_token = "<OOV>"
        self.vocab = tokenizer.get_vocab()

    def train(self, datasets):
        """Trains the trigram language model on a dataset."""
        for song in datasets:
            tokens = ["<s>"] + song + ["</s>"]
            self.total_unigrams_count += len(tokens)

            for i in range(len(tokens)):
                # Handle OOV tokens
                token = tokens[i] if tokens[i] in self.vocab else self.oov_token
                
                # Count unigrams
                self.unigram_count[token] += 1
                
                if i > 0:
                    prev_token = tokens[i - 1] if tokens[i - 1] in self.vocab else self.oov_token
                    self.bigram_count[prev_token][token] += 1
                
                if i > 1:
                    prev_prev_token = tokens[i - 2] if tokens[i - 2] in self.vocab else self.oov_token
                    prev_token = tokens[i - 1] if tokens[i - 1] in self.vocab else self.oov_token
                    self.trigram_count[(prev_prev_token, prev_token)][token] += 1

    def nextProb(self, history_toks, next_toks):
        """Returns the probability of next_toks given history_toks."""
        probabilities = {}
        vocab_size = len(self.vocab)

        # Replace unknown tokens with OOV token
        history_toks = [tok if tok in self.vocab else self.oov_token for tok in history_toks]
        next_toks = [tok if tok in self.vocab else self.oov_token for tok in next_toks]

        if len(history_toks) < 2:
            # Unigram case: P(w) = (count(w) + 1) / (total_unigrams + V)
            for tok in next_toks:
                probabilities[tok] = (self.unigram_count[tok] + 1) / (self.total_unigrams_count + vocab_size)
        else:
            # Trigram case: P(w | h1, h2) = (count(h1, h2, w) + 1) / (count(h1, h2) + V)
            h1, h2 = history_toks[-2], history_toks[-1]
            for tok in next_toks:
                # if (h1, h2) in self.trigram_count and tok in self.trigram_count[(h1, h2)]:
                    # Trigram probability
                    probabilities[tok] = (self.trigram_count[(h1, h2)][tok] + 1) / (self.bigram_count[h1][h2] + vocab_size)
                # else:
                #     # Back-off to unigram probability
                #     probabilities[tok] = (self.unigram_count[tok] + 1) / (self.total_unigrams_count + vocab_size)
        # print(self.trigram_count)
        return probabilities

# Load training data (Example Dataset)
datasets = tokenized_lyrics
# Create and train the model
lm = TrigramLM()
lm.train(datasets)

"""CHECKPOINT 1.2"""
history_toks_1 = ['<s>', 'Are','Ġwe']
next_toks_1 = ['Ġout', 'Ġin', 'Ġto', 'Ġpretending', 'Ġonly']

history_toks_2 = ['And', 'ĠI']
next_toks_2 = ['Ġwas', "'m", 'Ġstood', 'Ġknow', 'Ġscream', 'Ġpromise']

# Generate next token probabilities
probs_1 = lm.nextProb(history_toks_1, next_toks_1)
probs_2 = lm.nextProb(history_toks_2, next_toks_2)

# Print Checkpoint 1.2 Results
print("\nCheckpoint 1.2:")

# Print First Case:
print("\nHistory:")
print(f"word1: {history_toks_1[-3]}")
print(f"word2: {history_toks_1[-2]}")
print(f"word3: {history_toks_1[-1]}")

print("\nNext token probabilities:")
for tok, prob in probs_1.items():
    print(f"{tok}: {prob:.6f}")

# Print Second Case:
print("\nHistory:")
print(f"word1: {history_toks_2[-2]}")
print(f"word2: {history_toks_2[-1]}")

print("\nNext token probabilities:")
for tok, prob in probs_2.items():
    print(f"{tok}: {prob:.6f}")

print("\n")


#---------------------------------------1.3 Perplexity of TrigramLM---------------------------------------------------------------------------------------

import math

def get_perplexity(probs):
    """
    Calculate perplexity from a list of probabilities.
    
    Args:
        probs (list): List of predicted token probabilities
        
    Returns:
        float: Perplexity score
    """
    if not probs:
        return float('inf')  # Return infinity if no probabilities are provided
    
    # Compute the geometric mean of the negative log probabilities
    log_sum = sum(math.log(prob) for prob in probs if prob > 0)
    n = len(probs)
    perplexity = math.exp(-log_sum / n) 

    return perplexity



def compute_perplexity(model, sequence):
    """
    Compute perplexity for a given sequence using TrigramLM.
    
    Args:
        model (TrigramLM): Trained trigram model
        sequence (list): List of tokens representing the sequence
        
    Returns:
        float: Perplexity score
    """
    probs = []
    
    # Iterate over the sequence to compute token probabilities
    for i in range(len(sequence)):
        history_toks = sequence[:i]  # Take previous tokens as history
        next_tok = sequence[i]
        
        # Get predicted probabilities from the model
        predicted_probs = model.nextProb(history_toks, [next_tok])
        
        if next_tok in predicted_probs:
            probs.append(predicted_probs[next_tok])
        # else:
        #     # If the token has zero probability → back-off to small value
        #     probs.append(1e-10)  # To avoid log(0) error
    
    # print(probs)
    # Compute perplexity from the list of probabilities
    perplexity = get_perplexity(probs)
    return perplexity

"""CHECKPOINT 1.3"""

sequences = [
    ['And', 'Ġyou', 'Ġgotta', 'Ġlive', 'Ġwith', 'Ġthe', 'Ġbad', 'Ġblood', 'Ġnow'],
    ['Sit', 'Ġquiet', 'Ġby', 'Ġmy', 'Ġside', 'Ġin', 'Ġthe', 'Ġshade'],
    ['And', 'ĠI', "'m", 'Ġnot', 'Ġeven', 'Ġsorry', ',', 'Ġnights', 'Ġare', 'Ġso', 'Ġstar', 'ry'],
    ['You', 'Ġmake', 'Ġme', 'Ġcraz', 'ier', ',', 'Ġcraz', 'ier', ',', 'Ġcraz', 'ier', ',', 'Ġoh'],
    ['When', 'Ġtime', 'Ġstood', 'Ġstill', 'Ġand', 'ĠI', 'Ġhad', 'Ġyou']
]
print("Checkpoint 1.3:\n")

# Iterate through the sequences and compute perplexity
for i, sequence in enumerate(sequences):
    perplexity = compute_perplexity(lm, sequence)
    print(f"Perplexity for Sequence {i + 1}: {perplexity:.4f}")


#---------------------------------------------------------------------------------------------------------------------------------------------------------

print("\nWhat are your observations about these results? Are the values similar or different? What is one major reason for this? (2-4 lines).\n")

print("For an n-gram model to capture more context, we need a bigger dataset. With more data, the chances of word pairs repeating increases, making the model more familiar with those combinations. Sadly, in our case, limited data was a major issue. This explains the very low probability for the next tokens in Checkpoint 1.2, which in turn results in extremely large perplexity.")

