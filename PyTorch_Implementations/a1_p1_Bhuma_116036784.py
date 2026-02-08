import re
import numpy as np
from  collections import Counter
import string


#-------------------------------------------1.1. Word RegEx tokenizer------------------------------------------------------------
def wordTokenizer(sent):
    """
    Tokenizes a sentence into words, numbers and punctuations.
    Args:
        sent: A string.
    Returns:
        A list
    """
    re_pattern = r"[@#]\w+|\d+(?:\.\d+)+|\w+(?:\.\w+)+(?:\.)?|\w+['’`]?\w*|[^\w\s]"  # Regular expression pattern for tokenization, it includes all the patterns that were mentioned in the assignment
    return re.findall(re_pattern, sent) # Apply the above regex to the input

# This function is to read the file and format is accordingly to the WordTokenizer function
def tokenize_file(file_path): 
    """
    Tokenizes a file into sentences and words.
    Args:
        file_path: A string.
    Returns:
        A list of lists
    """ 
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines() # Reads the file line by line

    tokens = [wordTokenizer(line.strip()) for line in lines if line.strip()] # Apply tokenizer to each line separately
    
    return tokens  # Returns a list of lists, where each sublist is a tokenized sentence

file_path = "C:/Users/byash/Downloads/a1_p1_Bhuma_116036784.py daily547_tweets.txt"
tokens = tokenize_file(file_path)

"""CHECKPOINT 1.1"""

print("\nBelow are the answers for Checkpoint 1.1:\n")
print(f"Output of the first 5 documents:\n {tokens[:5]}")  # Output: List of first 5 tokenized sentences
print(f"\nOutput of the last document: \n {tokens[-1:]}") # Output: List of tokenized sentences of the last document






#-----------------------------------------------------1.2 Spaceless BytePair Tokenizer--------------------------------------------------
# Read and preprocess the file
file_path = "C:/Users/byash/Downloads/a1_p1_Bhuma_116036784.py daily547_tweets.txt"
with open(file_path, 'r', encoding='utf-8') as file:
    docs = file.readlines()

def clean_text(docs):
    """
    Converts all non-ASCII characters to '?' and tokenizes sentences.
    """
    cleaned_text = ["".join(char if char in string.printable else "?" for char in line) for line in docs] #Replace all the non-ASCII characters with "?"
    tokenized_sentences = [sentence.split() for sentence in cleaned_text] #Tokenize the words in a sentence, keeping the sentence structure intact
    return tokenized_sentences

def get_pair(curr_words):
    """
    Compute frequency of adjacent pairs in words.
    """
    pairs = Counter()  # Counter function helps us count the duplicates for the words, keeps a dictionary sort of structure
    for word in curr_words:
        for i in range(len(word) - 1):
            pairs[(word[i], word[i + 1])] += 1 # increment everytime a pair of same type is found
    return pairs    

def merge_vocab(pair, corpus):
    """
    Merges the most frequent pair in the corpus.
    """
    pair_joined = ''.join(pair)  # Merge the pair into a single token
    pattern = re.escape(' '.join(pair))  # Create regex pattern for the pair
    
    # Apply regex substitution to each tokenized word list
    new_corpus = [re.sub(pattern, pair_joined, ' '.join(word)).split() for word in corpus]
    
    return new_corpus

def log_iteration(iteration, pair_frequency):  #CHECK POINT 1.2 - Print the top five most frequent pairs at iterations 0, 1, 10, 100, and 500.
    """
    Print the top 5 most frequent pairs at certain iterations.
    """
    if iteration in {0, 1, 10, 100, 500}:
        print(f"\n🔹 Iteration {iteration}: Top 5 Frequent Pairs")
        print(sorted(pair_frequency.items(), key=lambda x: x[1], reverse=True)[:5])

def spacelessBPELearn(docs, max_vocabulary=1000):
    """
    Learn a spaceless Byte Pair Encoding (BPE) vocabulary from documents.

    Args:
        docs (list): A list of tokenized sentences (list of lists of words).
        max_vocabulary (int): Maximum number of vocabulary items to learn.

    Returns:
        set: The final BPE vocabulary.
    """
    iteration = 0

    tokenized_sentences = clean_text(docs)  # Use the correctly tokenized sentences
    final_vocabulary = set(char for doc in tokenized_sentences for word in doc for char in word)
    main_corpus = [word for doc in tokenized_sentences for word in doc]
    print("\nThis is the start of CHECKPOINT 1.2:\n ")   # Every output we get from here on is a a part of CheckPoint-1.2

    while len(final_vocabulary) < max_vocabulary:
        pair_frequency = get_pair(main_corpus)
        if not pair_frequency:
            break

        maximum_frequency_pair = max(pair_frequency, key=pair_frequency.get)
        log_iteration(iteration, pair_frequency)  # Log top 5 pairs

        # Only update the word list
        main_corpus = merge_vocab(maximum_frequency_pair, main_corpus)

        # Add merged characters to final vocabulary
        final_vocabulary.add(''.join(maximum_frequency_pair))
        
        iteration += 1

    return final_vocabulary

def spacelessBPETokenize(text, vocab):
    """
    Tokenize text using the learned BPE vocabulary.

    Args:
        text (str): A single string to be word-tokenized.
        vocab (set): A set of valid vocabulary words.

    Returns:
        list: A list of strings of all word tokens, in order, from the input string.
    """
    text = "".join(char if char in string.printable else "?" for char in text)
    text = text.replace(" ", "")  # Remove spaces since BPE is spaceless
    words = list(text)

    i = 0
    while i < len(words) - 1:
        adjacent_chars = ''.join([words[i], words[i + 1]])
        if adjacent_chars in vocab:
            words[i] = adjacent_chars
            del words[i + 1]
        else:
            i += 1

    return words

# Learn BPE vocabulary
bpe_vocabulary = spacelessBPELearn(docs, max_vocabulary=1000)

# printing the final vocabulary after BPE
print("\n Second question in the CheckPoint 1.2: Final BPE Vocabulary:")
print(sorted(bpe_vocabulary))

# Tokenize and print the first 5 documents and the last document
tokenized_docs = [spacelessBPETokenize(" ".join(doc), bpe_vocabulary) for doc in clean_text(docs)]

print("\nFirst half of third question in Checkpoint 1.2: Tokenization of First 5 Documents:\n")
for i, doc in enumerate(tokenized_docs[:5]):
    print(f"Document {i+1}:{doc}")

print("\nSecond half of third question in CheckPoint 1.2: Tokenization of the Last Document:\n")
print(tokenized_docs[-1])  # Print the last document's tokenization