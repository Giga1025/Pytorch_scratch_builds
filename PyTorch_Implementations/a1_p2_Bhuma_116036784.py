import re
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# We are defining wordTokenizer here again because we will be using it in the later sections
def wordTokenizer(sent): 
    """
    Tokenizes a sentence into words, numbers and punctuations.
    Args:
        sent: A string.
    Returns:
        A list
    """
    re_pattern = r"[@#]\w+|\d+(?:\.\d+)+|\w+(?:\.\w+)+(?:\.)?|\w+['’`]?\w*|[^\w\s]"  # Regular expression pattern for tokenization
    return re.findall(re_pattern, sent)

def tokenize_file(file_path):
    """
    Tokenizes a file into sentences and words.
    Args:
        file_path: A string.
    Returns:
        A list of lists
    """ 
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines() # Read the file line by line

    tokens = [wordTokenizer(line.strip()) for line in lines if line.strip()] # Apply tokenizer to each line separately
    
    return tokens  # Returns a list of lists, where each sublist is a tokenized sentence

file_path = "C:/Users/byash/Downloads/a1_p1_Bhuma_116036784.py daily547_tweets.txt"
tokens = tokenize_file(file_path)


# 2.0 Loading the data set 


def getConllTags(filename):     # The function that was provided
    """
    Loads a CoNLL-style POS-tagged file.
    
    Input: 
        filename - path to the file.
    
    Output: 
        A list of sentences, where each sentence is a list of (word, POS-tag) tuples.
    """
    wordTagsPerSent = [[]]
    sentNum = 0

    with open(filename, encoding='utf8') as f:
        for wordtag in f:
            wordtag = wordtag.strip()

            if wordtag:  # If it's a valid word-POS line
                word, tag = wordtag.split("\t")
                wordTagsPerSent[sentNum].append((word, tag))
            else:  # If it's a new sentence
                wordTagsPerSent.append([])
                sentNum += 1

    return [sent for sent in wordTagsPerSent if sent]  # Remove empty lists

# Load the data
file_path = "C:/Users/byash/Downloads/daily547_3pos.txt"
sentences = getConllTags(file_path)

# Extract unique words and POS tags
unique_tokens = set()
unique_pos_tags = set()
tokens = [word for sentence in sentences for word, _ in sentence]  # Extract only words (tokens) from the nested list
pos_tags = [pos for sentence in sentences for _, pos in sentence]  # Extract POS tags

# print("Tokens:", tokens)
# print("POS Tags:", pos_tags)

# All the unique tokens and POS tags. Please  feel free to uncomment to look into all the kind of tokens i used in this section

unique_tokens = list(dict.fromkeys(tokens))
unique_pos_tags = list(dict.fromkeys(pos_tags))
#print(unique_pos_tags)
token_to_index = {token: idx for idx, token in enumerate(unique_tokens)}
pos_to_index = {pos: idx for idx, pos in enumerate(unique_pos_tags)}
Sentence_tokens = [[word for word, _ in sentence] for sentence in sentences]  
Sentence_pos_tags = [[pos for _, pos in sentence] for sentence in sentences]  # Extract POS tags
# print("Unique tokens to idx mapping:",token_to_index)
# print("Unique POS to idx mapping:", pos_to_index)
# print("Words and their POS:", sentences)
# print("Tokens Sentense wise:", Sentence_tokens)
# print("Pos tags in sentences:", Sentence_pos_tags)

#-----------------------------------------------------2.1 Lexical feature set---------------------------------------------------

# Checking if the first letter of the word is capitalized

def is_capitalized(tokens, targetI):
    """
    Checks if the first letter of the target word is capitalized.

    Args:
        tokens (list): List of words in a sentence.
        targetI (int): Index of the current word in the sentence.

    Returns:
        int: 1 if the first letter is capitalized, 0 otherwise.
    """
    return 1 if tokens[targetI][0].isupper() else 0  # Check if the first letter is capitalized and retrun 1 if True, 0 otherwise

# Getting the ASCII value of the first letter of the word

def f_letter_ascii(tokens, targetI):
    """
    Returns the ASCII value of the first letter of the target word.

    Args:
        tokens (list): List of words in a sentence.
        targetI (int): Index of the current word in the sentence.
    
    Returns:
        list: One-hot encoded vector of the ASCII value of the first letter of the target word.
    """
    first_letter = tokens[targetI][0]  # Get the first letter of the word
    ascii_cond = ord(first_letter) if ord(first_letter) < 256 else 256 #take the asci values if it is less than 256 otherwise set it to 257
    ascii_vals = [0] * 257                                         # Create a list of 257 zeros
    ascii_vals[ascii_cond] = 1  # Set the corresponding index to 1
    return ascii_vals

# Normalized length of the target word

def get_normalized_length(tokens, targetI):
    """
    Returns the normalized length of the target word.
    
    Args:
        tokens (list): List of words in a sentence.
        targetI (int): Index of the current word in the sentence.
        
    Returns:
        float: Normalized length of the target word.
    """
    word = tokens[targetI] # Get the target word
    return min(len(word), 10) / 10  # Normalize the word. Any words longer than 10 characters will be set to 1 and below will be in the range of 0 to 1

# One-hot encoding the previous word (This is similar to getting lags in time series data)

def Ohe_prev_word(tokens,targetI, token_to_index):
    """
    One-hot encodes the previous word.

    Args:
        tokens (list): List of words in a sentence.
        targetI (int): Index of the current word in the sentence.
        token_to_index (dict): Mapping of words to indices.
    
    Returns:
        list: One-hot encoded vector of the previous word.
    """ 
    one_hot_prev = [0] * len(token_to_index)
    if  targetI > 0:
        prev_word = tokens[targetI - 1]
        if prev_word in token_to_index:
            one_hot_prev[token_to_index[prev_word]] = 1
    return one_hot_prev

# One-hot encoding for current word

def Ohe_current_word(tokens,targetI, token_to_index):
    """
    One-hot encodes the current word.
    
    Args:
        tokens (list): List of words in a sentence.
        targetI (int): Index of the current word in the sentence.
        token_to_index (dict): Mapping of words to indices.

    Returns:
        list: One-hot encoded vector of the current word.
    """
    one_hot_current = [0] * len(token_to_index)
    if tokens[targetI] in token_to_index:
        one_hot_current[token_to_index[tokens[targetI]]] = 1
    return one_hot_current

# One-hot encoding for next word

def Ohe_next_word(tokens, targetI, token_to_index):
    """
    One-hot encodes the next word.
    
    Args:
        tokens (list): List of words in a sentence.
        targetI (int): Index of the current word in the sentence.
        token_to_index (dict): Mapping of words to indices.

    Returns:
        list: One-hot encoded vector of the next word.
    """
    one_hot_next = [0] * len(token_to_index)

    # Fix: Ensure `targetI` is within bounds
    if targetI < len(tokens) - 1:  # Corrected index condition
        next_word = tokens[targetI + 1]
        if next_word in token_to_index:
            one_hot_next[token_to_index[next_word]] = 1

    return one_hot_next



def getFeaturesForTarget(tokens,targetI, wordToIndex):
    """
    Extracts a concatenated feature vector for a given target word.

    Parameters:
        tokens (list of lists): A list of sentences, each containing words (tokens).
        sentI (int): Index of the sentence in the tokens list.
        targetI (int): Index of the target word in the sentence.
        wordToIndex (dict): Dictionary mapping words to unique indices.

    Returns:
        np.array: Concatenated feature vector for the target word.
    """
    # Feature 1: Is first letter capitalized? (Binary)
    capitalized = np.array([is_capitalized(tokens,targetI)])

    # Feature 2: First letter ASCII encoding (One-hot of size 257)
    first_letter_encoding = np.array(f_letter_ascii(tokens, targetI))

    # Feature 3: Normalized length of the word (1 value)
    word_length = np.array([get_normalized_length(tokens,targetI)])

    # Feature 4: One-hot encoding of the previous word
    one_hot_prev = np.array(Ohe_prev_word(tokens,targetI, wordToIndex))

    # Feature 5: One-hot encoding of the current word
    one_hot_target = np.array(Ohe_current_word(tokens,targetI, wordToIndex))

    # Feature 6: One-hot encoding of the next word
    one_hot_next = np.array(Ohe_next_word(tokens,targetI, wordToIndex))

    # Concatenating all features into a single flat vector
    featureVector = np.concatenate([
        capitalized, 
        first_letter_encoding, 
        word_length, 
        one_hot_prev, 
        one_hot_target, 
        one_hot_next
    ])

    return featureVector


def createFeatureMatrix(Sentence_tokens, token_to_index):
    """
    Creates a feature matrix for all tokens in the dataset.

    Args:
        Sentence_tokens (list): A list of sentences, where each sentence is a list of tokens.
        token_to_index (dict): Dictionary mapping words to indices.

    Returns:
        np.array: Feature matrix with shape (Total Words, Feature Size).
    """
    feature_matrix = []

    for sent in Sentence_tokens:  # Iterate over sentences
        for targetI in range(len(sent)):  # Iterate over words in the sentence
            features = getFeaturesForTarget(sent, targetI, token_to_index)  # Extract features
            feature_matrix.append(features)  # Store each word's feature vector

    return np.array(feature_matrix)  # Convert to NumPy array for proper matrix structure

# Example usage:
feature_matrix = createFeatureMatrix(Sentence_tokens, token_to_index)
#print(feature_matrix.shape)  # Output: (Total Words, Feature Size)


def encodePOSTags(pos_tags, pos_to_index):
    """
    Encodes a list of POS tags into their corresponding indices.

    Args:
        pos_tags (list): List of POS tags.
        pos_to_index (dict): Dictionary mapping POS tags to indices.

    Returns:
        list: List of encoded POS tag indices.
    """
    return [pos_to_index[pos] for pos in pos_tags]

# Example usage:
y = encodePOSTags(pos_tags, pos_to_index)
#print(y)  # Output: List of encoded POS tag indices

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(feature_matrix, y, test_size=0.3,shuffle=False, random_state=123)

y_train = np.array(y_train)
y_test = np.array(y_test)

#Check point 2.1
print("\n CHECKPOINT 2.1: Sum of first and last 5 feature vectors\n")

# Assuming feature_matrix is your numpy array of features
sum_f5 = np.sum(feature_matrix[0:5, :], axis=0)    # Sum of the first 5 rows
sum_l5 = np.sum(feature_matrix[-5:, :], axis=0)    # Sum of the last 5 rows
print(f"Sum of first 5 feature vectors {sum_f5}")
print(f"Sum of last 5 feature vectors {sum_l5}")

#--------------------------------------------------2.2 Train Logistic Regression-----------------------------------------------

def trainLogReg(train_data, dev_data, learning_rate=0.01, l2_penalty=0.01, epochs=100):
    """
    Trains a multiclass logistic regression model using full-batch gradient descent.

    Args:
          Training data: Its a feature matrix with each row representing One hot encoded word
          Dev_data : The dev data consists integers showcasing the type of pos
          learning_rate: It control the flow of gradient descent
          l2_penalty: it penalizes the weights
          epochs: Number of times the run loops in gradient descent
    """
    # Unpack train and dev data
    X_train, y_train = train_data
    X_dev, y_dev = dev_data

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)  # Ensure integer labels

    X_dev_tensor = torch.tensor(X_dev, dtype=torch.float32)
    y_dev_tensor = torch.tensor(y_dev, dtype=torch.long)

    # Define Logistic Regression Model
    input_dim = X_train.shape[1]
    output_dim = len(set(y_train))  # Number of unique classes

    model = nn.Linear(input_dim, output_dim)  

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=l2_penalty)

    # Initialize tracking variables
    train_losses, dev_losses = [], []
    train_accuracies, dev_accuracies = [], []

    # Training loop
    for epoch in range(epochs):
        logits = model(X_train_tensor)
        loss = criterion(logits, y_train_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track training loss
        train_losses.append(loss.item())

        # Compute training accuracy
        with torch.no_grad():
            train_preds = torch.argmax(logits, dim=1).numpy()
            train_labels = y_train_tensor.numpy()
            train_accuracy = accuracy_score(train_labels, train_preds)
            train_accuracies.append(train_accuracy)

        # Evaluate on dev set
        with torch.no_grad():
            dev_logits = model(X_dev_tensor)
            dev_loss = criterion(dev_logits, y_dev_tensor)
            dev_losses.append(dev_loss.item())

            # Compute dev accuracy
            dev_preds = torch.argmax(dev_logits, dim=1).numpy()
            dev_labels = y_dev_tensor.numpy()
            dev_accuracy = accuracy_score(dev_labels, dev_preds)
            dev_accuracies.append(dev_accuracy)

        # Print progress every 10 epochs
        # if (epoch + 1) % 10 == 0:
        #     print(f"Epoch [{epoch+1}/{epochs}] - Loss: {loss.item():.4f}, Train Acc: {train_accuracy:.4f}, Dev Acc: {dev_accuracy:.4f}")

    return model, train_losses, train_accuracies, dev_losses, dev_accuracies

def predict(model, X_input):
    """
    Uses the trained model to predict classes for new input data.

    Args:
        model (torch.nn.Module): Trained logistic regression model.
        X_input (numpy.ndarray): Feature matrix for new data.

    Returns:
        numpy.ndarray: Predicted class labels.
    """
    model.eval()  # Set model to evaluation mode
 
    # Convert input data to a PyTorch tensor
    X_input_tensor = torch.tensor(X_input, dtype=torch.float32)

    # Perform forward pass to get logits
    with torch.no_grad():
        logits = model(X_input_tensor)

    # Get predicted class labels (argmax for multi-class classification)
    predictions = torch.argmax(logits, dim=1).numpy()
    
    return predictions


# Fit the model to get the accuracies and losses
model, train_losses, train_accuracies, dev_losses, dev_accuracies = trainLogReg(
    (X_train, y_train), (X_test, y_test)
)


## CHECKPOINT 2.2

print("\nCHECKPOINT 2.2: Plot the training and dev set loss and accuracy curves\n")

epochs = list(range(0, 100))  # Generates numbers from 0 to 99

fig, axs = plt.subplots(2, 2, figsize=(15, 10))  # Increase figure size to avoid overlap

# Plot Training Accuracy
axs[0, 0].plot(epochs, train_accuracies, label="Train Accuracy", color="blue")
axs[0, 0].set_xlabel("Epochs")
axs[0, 0].set_ylabel("Accuracy")
axs[0, 0].set_title("Training Accuracy")
axs[0, 0].legend()
axs[0, 0].grid()

# Plot Training Loss
axs[0, 1].plot(epochs, train_losses, label="Train Loss", color="red")
axs[0, 1].set_xlabel("Epochs")
axs[0, 1].set_ylabel("Loss")
axs[0, 1].set_title("Training Loss")
axs[0, 1].legend()
axs[0, 1].grid()

# Plot Test Accuracy
axs[1, 0].plot(epochs, dev_accuracies, label="Test Accuracy", color="blue")
axs[1, 0].set_xlabel("Epochs")
axs[1, 0].set_ylabel("Accuracy")
axs[1, 0].set_title("Test Accuracy")
axs[1, 0].legend()
axs[1, 0].grid()

# Plot Test Loss
axs[1, 1].plot(epochs, dev_losses, label="Test Loss", color="red")
axs[1, 1].set_xlabel("Epochs")
axs[1, 1].set_ylabel("Loss")
axs[1, 1].set_title("Test Loss")
axs[1, 1].legend()
axs[1, 1].grid()

plt.tight_layout()  #  Adjust layout to prevent overlap
plt.show()

#---------------------------------------------------2.3 Hyperparameter Tuning---------------------------------------------------

def gridSearch(train_set, dev_set, learning_rates, l2_penalties, epochs=100):
    """
    Performs grid search to find the best learning rate and L2 regularization.
    Prints a table with results and plots the best model performance.

    Returns:
        dict: Model accuracies for each (lr, l2) combination.
        float: Best learning rate.
        float: Best L2 penalty.
        nn.Module: Best trained model.
    """
    X_train, y_train = train_set
    X_dev, y_dev = dev_set

    best_accuracy = 0
    best_lr, best_l2_penalty = None, None
    best_model = None
    best_train_losses, best_dev_losses = None, None
    best_train_accuracies, best_dev_accuracies = None, None

    model_accuracies = {}

    # Initialize empty table for results
    results_table = np.zeros((len(learning_rates), len(l2_penalties)), dtype=object)

    for i, lr in enumerate(learning_rates):
        for j, l2 in enumerate(l2_penalties):
            # print(f"\nTraining with LR: {lr}, L2 Penalty: {l2} ...")

            # Train model
            model, train_losses, train_accuracies, dev_losses, dev_accuracies = trainLogReg(train_set, dev_set, lr, l2, epochs)

            # Get final accuracy from the last epoch
            final_accuracy = dev_accuracies[-1]
            model_accuracies[(lr, l2)] = final_accuracy

            # Store in results table
            results_table[i, j] = f"{final_accuracy:.4f}"

            # print(f"LR: {lr}, L2: {l2}, Final Dev Accuracy: {final_accuracy:.4f}")

            # Update best model
            if final_accuracy > best_accuracy:
                best_accuracy = final_accuracy
                best_lr, best_l2_penalty = lr, l2
                best_model = model
                best_train_losses, best_dev_losses = train_losses, dev_losses
                best_train_accuracies, best_dev_accuracies = train_accuracies, dev_accuracies
    print("\nCHECKPOINT 2.3: Question-1, as suggested we had printed a table for LR/L2\n")
    # Print results table manually
    print("\nGrid Search Results (Validation Accuracy):")
    print("LR/L2", end="  ")
    for l2 in l2_penalties:
        print(f"{l2:.0e}", end="  ")
    print()

    for i, lr in enumerate(learning_rates):
        print(f"{lr}  ", end="  ")
        for j in range(len(l2_penalties)):
            print(results_table[i, j], end="  ")
        print()

    print("\nBest Hyperparameters Found:")
    print(f"Best Learning Rate: {best_lr}")
    print(f"Best L2 Penalty: {best_l2_penalty}")
    print(f"Best Validation Accuracy: {best_accuracy:.4f}")

    # Plot the best model's accuracy and loss curves
    plot_metrics(best_train_losses, best_dev_losses, best_train_accuracies, best_dev_accuracies, best_lr, best_l2_penalty)

    return model_accuracies, best_lr, best_l2_penalty, best_model


def plot_metrics(train_losses, dev_losses, train_accuracies, dev_accuracies, best_lr, best_l2_penalty):
    """
    Plots training loss, validation loss, training accuracy, and validation accuracy.
    """
    epochs = range(1, len(train_losses) + 1)
    
    ######### CHECK POINT##########
    print("\nCHECKPOINT 2.3: Question 2, This plot below gives us the loss and accuracy for both training and test for the best model\n")
    
    
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss", color='blue')
    plt.plot(epochs, dev_losses, label="Validation Loss", color='red')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()

    # Plot Accuracy Curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label="Train Accuracy", color='green')
    plt.plot(epochs, dev_accuracies, label="Validation Accuracy", color='orange')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training & Validation Accuracy")
    plt.legend()

    plt.show()

# Define hyperparameters
learning_rates = [0.1, 1, 10]
l2_penalties = [0.00001, 0.001, 0.1]

# Run grid search
model_accuracies, best_lr, best_l2_penalty, best_model = gridSearch(
    (X_train, y_train), 
    (X_test, y_test), 
    learning_rates, 
    l2_penalties,
    epochs=100
)




learning_rates = sorted(set(lr for lr, _ in model_accuracies.keys()))
l2_penalties = sorted(set(l2 for _, l2 in model_accuracies.keys()))

# Convert L2 penalties to scientific notation strings for display
l2_labels = [f"{l2:.0e}" for l2 in l2_penalties]
lr_labels = [str(lr) for lr in learning_rates]

# Create an empty NumPy array to store accuracy values
results_table = np.full((len(learning_rates), len(l2_penalties)), "", dtype=object)

# Fill the array with accuracy values
for (lr, l2), accuracy in model_accuracies.items():
    row_idx = learning_rates.index(lr)
    col_idx = l2_penalties.index(l2)
    results_table[row_idx, col_idx] = f"{accuracy:.4f}"

# Print the results in a tabular format



#--------------------------------------------------------2.4 Best model inference-----------------------------------------------

model1, train_losses, train_accuracies, dev_losses, dev_accuracies = trainLogReg(
    (X_train, y_train), (X_test, y_test), 10, 0.00001) # Best model


sampleSentences = [['The horse raced past the barn fell.'],
                   ['For 3 years, we attended S.B.U. in the CS program.'],
                   ['Did you hear Sam tell me to "chill out" yesterday? #rude']]
# print(sampleSentences)

samp_tok = [wordTokenizer(sente[0]) for sente in sampleSentences] # As suggested, we are tokenizing the model by sentence using regex tokenizer

sample_feature = []

for sentI in range(len(samp_tok)):  # Iterate over sentences
    tokens = samp_tok[sentI]  # Get the current sentence's tokens

    for targetI in range(len(tokens)):  # Iterate over words in the sentence
        features = getFeaturesForTarget(tokens, targetI, token_to_index)  # Pass only `tokens`
        sample_feature.append(features)  # Store each word's feature vector

# Convert to a NumPy array for proper matrix structure
sample_feature= np.array(sample_feature) 


sample_model = predict(model1, sample_feature)
# Convert the predicted indices back to POS tags
id2tag = {v: k for k, v in pos_to_index.items()}
predicted_tags = [id2tag[idx] for idx in sample_model]



word_pos_dict = {}
for sentence in sentences:
    for word, pos in sentence:
        word_pos_dict[word] = pos  # Assign POS tag to word

# # Print the dictionary
# print(word_pos_dict)
# Flatten `samp_tok` if it's a list of lists
flat_samp_tok = [word for sublist in samp_tok for word in sublist]

# Match words with their POS tags

##### CHECKPOINT########

print("\n CHECK POINT:2.4, Question-1, Print the POS predicted for each token\n")

matched = [(word, word_pos_dict.get(word, "UNK")) for word in flat_samp_tok]

# Print results
print(f"\nMatching the new words with the pos we know {matched}\n") # The "UNK" means we dont know the POS of these words. 
print(f"\nThe prediction made by the model that we got from the best model {predicted_tags}\n")


print("\n CHECK POINT: 2.4, Question-2, My observation about the qualitative performance of the best model\n")


print("\nWell my model hit almost 80 percent accuracy, which is actually pretty good given us having a very small corpus, but the issue is the training test had an accuracy of 97 percent, which can easily be said that the model is overfitting, in simple terms, the model is just remembering the data and acts on it. This by the looks of it might be a big issue, but since the model is not grasping the logic, when there is data given out of the corpus, the model will struggle as it dont rememebr mugging that new data\n")

print("\nThat is what exactly happening to our model here, though it perfomed very well on the training set and test set, it struggled on the new data that was out of corpus, though we dont know the pos of new words(in the model), we as a human knows to identify Nouns and verbs and we can see how the model is struggling for unknown pos values.\n")

print("\n So, to make this model not overfit, we need to modulate the LR and L2 properly, try to increase the data(more data, more laerning), and since we are using neural netwrok logic its better to implement with few neurons and layers")









