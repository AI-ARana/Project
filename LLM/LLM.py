from importlib.metadata import version
import os
import urllib.request
import re
import importlib
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

# if not os.path.exists("the-verdict.txt"):
#     url = ("F:/Jupyter/Python/the-verdict.txt")
#     file_path = "the-verdict.txt"
#     urllib.request.urlretrieve(url, file_path)
    
file_path = "F:/Jupyter/Python/the-verdict.txt"
if os.path.exists(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
else:
    raise FileNotFoundError(f"File not found at {file_path}")

# with open("the-verdict.txt", "r", encoding="utf-8") as f:
#     raw_text = f.read()
   
# The following regular expression will split on whitespaces
text = "Hello, world. This, is a test."
result = re.split(r'(\s)', text)

# We don't only want to split on whitespaces but also commas and periods, 
# so let's modify the regular expression to do that as well

result = re.split(r'([,.]|\s)', text)

# This creates empty strings, let's remove them
# Strip whitespace from each item and then filter out any empty strings.

result = [item for item in result if item.strip()]

# This looks pretty good, but let's also handle other types of punctuation, such as periods, question marks, and so on

text = "Hello, world. Is this-- a test?"

result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
result = [item.strip() for item in result if item.strip()]

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]

# Let's calculate the total number of tokens
# Build a vocabulary that consists of all the unique tokens

all_words = sorted(set(preprocessed))
vocab_size = len(all_words)

vocab = {token:integer for integer,token in enumerate(all_words)}

# Below are the first 50 entries in this vocabulary

for i, item in enumerate(vocab.items()):
    # print(item)
    if i >= 50:
        break

class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
                                
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
        
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

tokenizer = SimpleTokenizerV1(vocab)

text = """"It's the last he painted, you know," 
           Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
# We can decode the integers back into text
tokenizer.decode(ids)

tokenizer.decode(tokenizer.encode(text))

# tokenize the following text

tokenizer = SimpleTokenizerV1(vocab)

# text = "Hello, do you like coffee. Is this-- a test?"

# tokenizer.encode(text)

# add another token called "<|endoftext|>" which is used in GPT-2 training to denote the end of a text 

all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])

vocab = {token:integer for integer,token in enumerate(all_tokens)}

len(vocab.items())

# for i, item in enumerate(list(vocab.items())[-5:]):
#     print(item)

# Adjust the tokenizer accordingly so that it knows when and how to use the new <unk> token

class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = { i:s for s,i in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [
            item if item in self.str_to_int 
            else "<|unk|>" for item in preprocessed
        ]

        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
        
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text

# Tokenize text with the modified tokenizer:

tokenizer = SimpleTokenizerV2(vocab)
text1 = "Hello, do you like coffee?"
text2 = "In the sunlight terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
tokenizer.encode(text)
tokenizer.decode(tokenizer.encode(text))
tokenizer = tiktoken.get_encoding("gpt2")
text = (
    "Hello, do you like coffee? <|endoftext|> In the sunlight terraces"
     "of someunknownPlace."
)

integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
strings = tokenizer.decode(integers)
with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

enc_text = tokenizer.encode(raw_text)

# For each text chunk, we want the inputs and targets
# Since we want the model to predict the next word, the targets are the inputs shifted by one position to the right

enc_sample = enc_text[50:]
context_size = 4
x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]

# One by one, the prediction would look like as
for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    
for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader_v1(txt, batch_size=4, max_length=256, 
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):

    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader

# Test the dataloader with a batch size of 1 for an LLM with a context size of 4

with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
dataloader = create_dataloader_v1(
    raw_text, batch_size=1, max_length=4, stride=1, shuffle=False
)
data_iter = iter(dataloader)
first_batch = next(data_iter)
second_batch = next(data_iter)
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Inputs:\n", inputs)
print("\nTargets:\n", targets)

# Suppose we have the following four input examples with input ids 2, 3, 5, and 1 (after tokenization)
input_ids = torch.tensor([2, 3, 5, 1])
# For the sake of simplicity, suppose we have a small vocabulary of only 6 words and we want to create embeddings of size 3
vocab_size = 6
output_dim = 3
torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
# This would result in a 6x3 weight matrix
# print(embedding_layer.weight)
# To convert a token with id 3 into a 3-dimensional vector
# print(embedding_layer(torch.tensor([3])))

# Note that the above is the 4th row in the embedding_layer weight matrix
# To embed all four input_ids values above
# print(embedding_layer(input_ids))
# The BytePair encoder has a vocabulary size of 50,257
# Suppose we want to encode the input tokens into a 256-dimensional vector representation

vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

# If we sample data from the dataloader, we embed the tokens in each batch into a 256-dimensional vector
# If we have a batch size of 8 with 4 tokens each, this results in a 8 x 4 x 256 tensor

max_length = 4
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=max_length,
    stride=max_length, shuffle=False
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)

# print("Token IDs:\n", inputs)
print("\nInputs shape: ", inputs.shape)

token_embeddings = token_embedding_layer(inputs)
print("Token Embeddings Shape: ",token_embeddings.shape)

# GPT-2 uses absolute position embeddings, so we just create another embedding layer

context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

pos_embeddings = pos_embedding_layer(torch.arange(max_length))
print("Position Embeddings Shape: ", pos_embeddings.shape)

# To create the input embeddings used in an LLM, we simply add the token and the positional embeddings

input_embeddings = token_embeddings + pos_embeddings
print("Tnput Embeddings Shape: ", input_embeddings.shape)
