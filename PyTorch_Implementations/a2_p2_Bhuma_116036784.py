import numpy as np
import random, os, sys, math, csv, re, collections, string
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import transformers
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from collections import defaultdict, Counter
import heapq
import matplotlib.pyplot as plt
from transformers import GPT2TokenizerFast

#------------------------------------------2.1. Preparing the dataset-------------------------------------------------
#Read the data from the csv and remove the last 5 rows which we will be using for testing the models
with open('songs.csv', newline='') as f:

    reader = csv.reader(f)
    data = list(reader)[1:-5]


def GPT2_Tokenizer():
    """
    Initialize GPT2TokenizerFast with special tokens.

    Returns:
        tokenizer (GPT2TokenizerFast): GPT-2 tokenizer with added special tokens
    """
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")  # Import the Gpt2 Tokenizer which uses BPE for tokenizing
    tokenizer.pad_token = tokenizer.eos_token               # Assign the end of sentence token as the pad token 
    tokenizer.add_special_tokens({'bos_token': '<s>', 'eos_token': '</s>'})  # Add special tokens for bos and eos
    # We have to remember that gpts has the option to add special tokens but it doesnt know how to use them, so we have work accordingly
    return tokenizer

def chunk_tokens(tokens, start_token_id, end_token_id, pad_token_id, chunk_len=128):
    """
    Split token sequences into fixed-length chunks.

    Args:
        tokens (list): List of token IDs
        start_token_id (int): BOS token ID
        end_token_id (int): EOS token ID
        pad_token_id (int): PAD token ID
        chunk_len (int): Length of output sequences

    Returns:
        torch.Tensor: Tensor of shape (#chunks, chunk_len)
    """
    chunks = []

    # Split tokens into chunks of size `chunk_len - 2` to account for BOS and EOS tokens
    for i in range(0, len(tokens), chunk_len - 2): # This start at 0, then initial value keeps changing to the start of the chunk, which is exactly a chunk_len-2 distance away
        chunk = tokens[i : i + (chunk_len - 2)]
        # print(chunk)
        # Add BOS and EOS tokens
        chunk = [start_token_id] + chunk + [end_token_id] # The subtracted -2 are added here for bos and eos, returning to the original chunk len
        
        # Pad if shorter than chunk_len
        if len(chunk) < chunk_len:
            chunk += [pad_token_id] * (chunk_len - len(chunk)) # If the chunk len is less than the required, we add pos_tokens differece times
        
        chunks.append(chunk) # This will return a list of list of chunks. The first list being the song, and the inner list is that song being chunked
    # Convert to tensor
    return torch.tensor(chunks)

def create_chunks(data, chunk_len=64):
    """
    Create a defaultdict mapping (album, song) → chunks.
    
    Args:
        data (list): Dataset containing [album, song, lyrics].
        chunk_len (int): Length of each chunk.
    
    Returns:
        defaultdict: Dictionary mapping (album, song) → chunks.
    """
    # Use defaultdict with a lambda function to return an empty tensor
    song_chunks = defaultdict(lambda: torch.empty(0))
    all_chunks = []

    for row in data:
        album = row[1].strip().lower() # Store the album name 
        song = row[0].strip().lower() # Store the song name
        lyrics = row[2] # Store the lyrics
        song_reg = re.sub(r'\n\[[\x20-\x7f]+\]', '', lyrics) #Clean with regex
        token_ids = tokenizer.encode(song_reg) # Convert the text to token_ids based on gp2 vocab
        chunks = chunk_tokens(token_ids, start_token_id, end_token_id, pad_token_id, chunk_len) #Call the previous defined function to chunk these tokens
        all_chunks.append(chunks)
        if len(chunks) > 0:     # Store in dictionary only if valid chunks exist
            song_chunks[(album, song)] = chunks
    
    return all_chunks, song_chunks

def get_song_chunk(song_chunks, album, song):
    """
    Get chunks for a specific album and song.
    
    Args:
        song_chunks (defaultdict): Dictionary mapping (album, song) → chunks.
        album (str): Album name.
        song (str): Song name.
    
    Returns:
        torch.Tensor: Tensor containing the song's chunks.
    """
    key = (album.strip().lower(), song.strip().lower()) # Find the respective song chunk for album and song
    
    if key in song_chunks:
        return song_chunks[key]
    else:
        raise ValueError(f"Song '{song}' from album '{album}' not found.")
    
tokenizer = GPT2_Tokenizer() #Assign the tokenizing function to a variable

#Initialize the special tokens to their respective variables
start_token_id = tokenizer.bos_token_id
end_token_id = tokenizer.eos_token_id
pad_token_id = tokenizer.pad_token_id

all_chunks, song_chunks = create_chunks(data)  # Get all the chunks of the songs and the dictionary mapping

all_chunks = torch.cat(all_chunks, dim=0)  # Concatenate all chunks into a single tensor

X = all_chunks[:, :-1]  # All but last token = input
y = all_chunks[:, 1:]   # All but first token = target

# Create TensorDataset and DataLoader
dataset = torch.utils.data.TensorDataset(X, y)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, drop_last=True)

"""CHECKPOINT 2.1"""
# Get chunks for a specific song
album = "Speak Now (Taylor's Version)"
song = "Enchanted (Taylor's Version)"
chunks = get_song_chunk(song_chunks, album, song)

# Output:
# print(song_chunks)
print("\n Checkpoint: 2.1 \n")
print(f"Chunks shape for '{song}' from '{album}': {chunks}")

#-------------------------------------2.2. Recurrent NN-based Language Model------------------------------------------------------------------------------

class RecurrentLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, rnn_hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim) # Embedding layer to convert token IDs into vectors of size embed_dim
        self.gru = nn.GRU(embed_dim, rnn_hidden_dim, batch_first=True) # GRU layer with hidden size = rnn_hidden_dim, batch_first=True
        self.layer_norm = nn.LayerNorm(rnn_hidden_dim) # Layer normalization for GRU's output
        self.fc = nn.Linear(rnn_hidden_dim, vocab_size) # Fully connected layer to map GRU output to vocab size

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len) - token IDs
        
        Returns:
            logits: Tensor of shape (batch_size, seq_len, vocab_size)
            hidden_state: Tensor of shape (1, batch_size, rnn_hidden_dim)
        """
        embedded = self.embedding(x)  # Embed the text into dence vectors
        gru_output, hidden_state = self.gru(embedded)  # Initialize the Gru and Hidden states
        norm_output = self.layer_norm(gru_output)  # Normalize the data
        logits = self.fc(norm_output)  # Applying linear transform

        return logits, hidden_state

    def stepwise_forward(self, x, prev_hidden_state):
        """
        Args:
            x: Tensor of shape (batch_size, 1) - single token ID
            prev_hidden_state: Tensor of shape (1, batch_size, rnn_hidden_dim)
        
        Returns:
            logits: Tensor of shape (batch_size, vocab_size)
            hidden_state: Tensor of shape (1, batch_size, rnn_hidden_dim)
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        embedded = self.embedding(x)  
        gru_output, hidden_state = self.gru(embedded, prev_hidden_state)  
        norm_output = self.layer_norm(gru_output)  
        logits = self.fc(norm_output)

        # Remove the sequence dimension since it's 1 (batch_size, vocab_size)
        logits = logits.squeeze(1)

        return logits, hidden_state
    

"""CheckPoint-2.2"""
print("\n Checkpoint 2.2\n")

seq_len = 64
batch_size = 32
embed_dim = 64
rnn_hidden_dim = 1024
vocab_size = len(tokenizer)

device = "cuda" if torch.cuda.is_available() else "cpu"
# Initializing the RNN model
RNNmodel = RecurrentLM(
vocab_size = vocab_size, 
embed_dim = embed_dim, 
rnn_hidden_dim = rnn_hidden_dim
).to(device)

dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)

logits, hidden_state = RNNmodel(dummy_input)

print(f"Shape of logits: {logits.shape}")
print(f"Shape of hidden_state: {hidden_state.shape}")

#--------------------------------------------2.3. Train RecurrentLM---------------------------------------------------------------------------------------------


def trainLM(model, data, pad_token_id, learning_rate, device):
    """
    Train RecurrentLM model using Cross-Entropy Loss.

    Args:
        model (nn.Module): Instance of RecurrentLM to be trained
        data (DataLoader): Contains X and y as defined in Part 2.1
        pad_token_id (int): Pad token ID for filtering
        learning_rate (float): Learning rate
        device (str): 'cuda' or 'cpu'

    Returns:
        list: Losses for each epoch
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)  # Set ignore_index = pad_token_id so that the model will ignore the paddings

    num_epochs = 15
    losses = []

    print("Starting Training...")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for x, y in data:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = model(x)  
            logits = logits.view(-1, logits.shape[-1]) # Reshaping the logits
            y = y.view(-1)    # Flattening the tensor
            # Compute Cross-Entropy Loss
            loss = criterion(logits, y)
            # Backward Pass and Update
            loss.backward()
            optimizer.step()
            total_loss+= loss.item()  

        # Average Loss for the Epoch
        avg_loss = total_loss / len(data)
        losses.append(avg_loss)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

    print("Training Completed.")
    
    # Plot Training Loss Curve
    plt.figure(figsize=(8, 6))
    plt.plot(losses, label="Training Loss", color='blue')
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.savefig("training_loss_curve.pdf")  
    plt.show()

    return losses

"""CHECKPOINT 2.3"""


print("\n Checkpoint 2.3\n")

print("Sub question-1: Plot the Loss curve\n")
learning_rate = 0.0007
  
torch.manual_seed(42)
losses = trainLM(RNNmodel, dataloader, tokenizer.pad_token_id, learning_rate, device)
print(losses)


print("Sub Question-2: Finding the perplexities for the samples\n")

def get_perplexity(probs):
  '''
    Given a list of probabilities, return the perplexity.
  '''

  log_sum = sum([math.log(p) for p in probs])
  perplexity = math.exp(-(log_sum / len(probs)))  # The formula for finding perplexity
  return perplexity

def compute_lyric_perplexity(model, sample_data, tokenizer, device):
  '''
  Computes the probability of a single lyric sample using the trained model.

  input: model(RecurrentLM) - Trained RNN language model
         sample_data - input lyric
         tokenizer - tokenizer used for the model
         device - whether to train model on CPU (="cpu") or GPU (="cuda")

  output: perplexity - perplexity of the lyric sample
  '''
  
  model.eval()
  with torch.inference_mode():
    token_ids = tokenizer.encode(sample_data)
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    tokens = [bos_id] + token_ids + [eos_id]
    
    input_token_ids = torch.tensor(tokens[:-1]).unsqueeze(0).to(device)  # Tokens what the model sees(After getting converted to 2d tensors)
    target_token_ids = torch.tensor(tokens[1:]).unsqueeze(0).to(device)  # Tokens what the model should predict(After getting converted to 2d tensors)
    logits, _ = model(input_token_ids) # predicting the next words in logits

    probs = F.softmax(logits, dim=-1) # Passing the logits through softmax to get the probabilities for the words
    
    seq_len = probs.size(1) # Get the number of tokens that we are processing
    probs_for_labels = probs[0, torch.arange(seq_len), target_token_ids[0]]  #Extracting the probabilities assigned to the correct tokens
    perplexity = get_perplexity(probs_for_labels.cpu().numpy()) #Call in the previously defined perplexity function
    return perplexity
  
print("\n2. Perplexity on samples:")
  # Sample sentences for perplexity evaluation.
sample_lyrics = [
    "And you gotta live with the bad blood now",
    "Sit quiet by my side in the shade",
    "And I'm not even sorry, nights are so starry",
    "You make me crazier, crazier, crazier, oh",
    "When time stood still and I had you"
]

for lyric in sample_lyrics:
    perplexity = compute_lyric_perplexity(RNNmodel, lyric, tokenizer, device)
    print(f"\nLyric: {lyric}")
    print(f"Perplexity: {perplexity:.5f}")

print("\n As we can see the from 1.3 and 2.3, RNN is doing much better in perplexities, it has lower scores than the ngram model, which is needed, as the model is less suprised of new words. In our ngram model, we are only looking at previous 2 words, which is not enough! we are losing a lot of context. In the case of rnn, due to the hidden state the model can look at previous words till long distances. Morever, the rnn model learns patterns,"
" this cause more generalization, whereas the ngram model only look at exact words matching, last but not least, the back bone of nlp, Word embeddings. Rnn uses word embeddings, which makes it MUCH easier to assess the context, but this is not possible with ngram.\n")

#--------------------------------------2.4. Autoregressive Lyric Generation------------------------------------------------------------------------------

def generate(model, tokenizer, start_phrase, max_len, device):
    """
    Generate text using a trained GRU model.

    Args:
        model (nn.Module): Trained instance of RecurrentLM
        tokenizer: Pre-trained GPT-2 tokenizer
        start_phrase (str): Start phrase
        max_len (int): Max number of tokens to generate
        device (str): 'cuda' or 'cpu'

    Returns:
        generated_tokens (list): List of generated token IDs
    """
    model.eval()
    generated_tokens = []

    with torch.no_grad():
        # Tokenize start_phrase
        start_tokens = tokenizer.encode(start_phrase)
        input_tensor = torch.tensor(start_tokens).unsqueeze(0).to(device)

        # Forward pass to get initial hidden state
        logits, hidden_state = model(input_tensor)
        last_token = input_tensor[:, -1]  # Start with last token from input

        generated_tokens = start_tokens[:]

        for _ in range(max_len - len(generated_tokens)):
            # Stepwise forward to get next token prediction
            logits, hidden_state = model.stepwise_forward(last_token, hidden_state)

            # Get the most likely next token
            next_token = torch.argmax(logits, dim=-1) #Get the highest probability token using argmax
            generated_tokens.append(next_token.item())

            # Stop if EOS or PAD token is generated
            if next_token.item() in [tokenizer.eos_token_id, tokenizer.pad_token_id]:
                break

            # Update last token for next step
            last_token = next_token

    # Convert token IDs back to text
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return generated_text


"""CHECKPOINT 2.4"""

print("\n Checkpoint 2.4\n")
start_phrases = [
    "<s>Are we",
    "<s>Like we're made of starlight, starlight",
    "<s>Once upon a time"
]

print("\n Generated Results:")
for phrase in start_phrases:
    generated_text = generate(RNNmodel, tokenizer, phrase, max_len=64, device=device)
    print(f"\n Start Phrase: {phrase}")
    print(f" Generated: {generated_text}")


#----------------------------------------------------------------Extra Credit-------------------------------------------------------------------------------------

def generate_sampled(model, tokenizer, start_phrase, max_len, device, temperature=1.0):
    """
    Generate text by sampling from the probability distribution over logits.
    """
    model.eval()
    generated_tokens = []

    with torch.no_grad():
        start_tokens = tokenizer.encode(start_phrase)
        input_tensor = torch.tensor(start_tokens).unsqueeze(0).to(device)

        logits, hidden_state = model(input_tensor)
        last_token = input_tensor[:, -1]
        generated_tokens = start_tokens[:]

        for _ in range(max_len - len(generated_tokens)): # Liop until desired length is reached
            logits, hidden_state = model.stepwise_forward(last_token, hidden_state)  # Applying the stepwise_forward
            logits = logits / temperature # Using tempoerature to control the randomness in the model

            # Convert logits to probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample from the distribution
            next_token = torch.multinomial(probs, num_samples=1) #Sampling tokens based on the probability distribution
            generated_tokens.append(next_token.item())

            if next_token.item() in [tokenizer.eos_token_id, tokenizer.pad_token_id]:
                break

            last_token = next_token

    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return generated_text


print("\n Extra Credit solution:\n")

print("\n 1)  Sampling-Based Generation Results:")
start_phrases = [
    "<s>Are we",
    "<s>Like we're made of starlight, starlight",
    "<s>Once upon a time"
]

for phrase in start_phrases:
    generated_text = generate_sampled(RNNmodel, tokenizer, phrase, max_len=64, device=device)
    print(f"\n Start Phrase: {phrase}")
    print(f" Generated: {generated_text}")


print("2)\n")
print("Both the functions here does almost same thing, but the way the model chooses the next word varies, In the first function our model will always choose the token with the highest probability for the next token(because of the argmax),"
" but in the second function the model we are introducing the concepts of Temperature and Probability distribution sampling, To put it simply these both parameters introduces the randomness during the prediction. So now we have wider range of predictions for the next token, instead of always picking the highest probabilty token this is the reason why we can see the 1st function has repetitive words, but the second function explores with other words ")
#---------------------------------------------------------------------------------------------------------------------------------------------------------



