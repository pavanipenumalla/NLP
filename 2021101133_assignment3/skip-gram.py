import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter

class SkipGramData():
    def __init__(self, data_dir, window_size):
        self.data_dir = data_dir
        self.threshold = 3
        self.sentences = None
        self.vocab = None
        self.word2idx = None
        self.idx2word = None
        self.word_embeddings = None
        self.subsampled_words = []
        self.window_size = window_size       
        self.embedding_dim = 300
        self.ns_exp = 0.75
        self.ns_array_len = 100000
        self.k = 5

    def load_data(self):
        data = pd.read_csv(self.data_dir)
        data = data['Description']
        data = data[:20000]
        self.sentences = data.tolist()

    def preprocess(self):
        self.sentences = [sentence.lower() for sentence in self.sentences]
        self.sentences = [re.sub(r'http\S+', 'URL', sentence) for sentence in self.sentences]
        self.sentences = [re.sub(r'www\S+', 'URL', sentence) for sentence in self.sentences]
        self.sentences = [sentence.translate(str.maketrans('', '', string.punctuation)) for sentence in self.sentences]
        self.sentences = [['<S>'] + word_tokenize(sentence) + ['</S>'] for sentence in self.sentences]

    def create_vocab(self):
        self.vocab = {}
        for sentence in self.sentences:
            for word in sentence:
                if word in self.vocab:
                    self.vocab[word] += 1
                else:
                    self.vocab[word] = 1

    def add_unk(self):
        unk_count = 0
        for word in list(self.vocab.keys()):
            if self.vocab[word] < self.threshold:
                unk_count += self.vocab[word]
                del self.vocab[word]
        self.vocab['<UNK>'] = unk_count

    def create_word_index_mappings(self):
        self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

    def subsample(self):
        total_word_count = sum(self.vocab.values())
        word_prob = {}
        for word in self.vocab:
            word_prob[word] = self.vocab[word] / total_word_count
        threshold = 1e-5
        for word in self.vocab:
            p = 1 - np.sqrt(threshold / word_prob[word])
            if np.random.rand() < p:
                self.subsampled_words.append(word)

    def get_data(self):
        self.load_data()
        self.preprocess()
        self.create_vocab()
        self.add_unk()
        self.create_word_index_mappings()
        self.subsample()

    def get_context_words(self, sentence, index):
        start = max(0, index - self.window_size)
        end = min(len(sentence), index + self.window_size + 1)
        context_words = []
        for i in range(start, end):
            if i != index:
                context_words.append(sentence[i])
        return context_words
    
    def get_positive_pairs(self):
        targets = []
        contexts = []
        for sentence in self.sentences:  
            for i, target in enumerate(sentence):
                if target in self.word2idx:
                    target_idx = self.word2idx[target]
                else:
                    target_idx = self.word2idx['<UNK>']
                if self.idx2word[target_idx] in self.subsampled_words:
                    context_words = self.get_context_words(sentence, i)
                    for context in context_words:
                        if context in self.word2idx:
                            context_idx = self.word2idx[context]
                        else:
                            context_idx = self.word2idx['<UNK>']
                        if self.idx2word[context_idx] in self.subsampled_words:
                            targets.append(target_idx)
                            contexts.append(context_idx)
        return targets, contexts
    
    def get_negative_samples(self,num_samples):
        freq = {}
        scaled_freq = {}
        negative_samples = []

        for word in self.vocab:
            freq[word] = self.vocab[word] ** self.ns_exp
        total_freq = sum(freq.values())

        for word in freq:
            scaled_freq[self.word2idx[word]] = max(1, int((freq[word] / total_freq) * self.ns_array_len))

        for word, count in scaled_freq.items():
            for _ in range(count):
                negative_samples.append(word)
        negative_samples = np.array(negative_samples)

        sample = np.random.choice(negative_samples, size=(num_samples, self.k))
        return sample

class SkipGram(nn.Module):
    def __init__(self,vocab_size,embedding_dim):
        super(SkipGram, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.in_embed = nn.Embedding(vocab_size, embedding_dim)
        self.out_embed = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, target, context):
        target_embed = self.in_embed(target)
        n1, n2 = target_embed.shape
        target_embed = target_embed.view(n1, 1, n2)
        context_embed = self.out_embed(context)
        scores = torch.bmm(target_embed, context_embed.permute(0, 2, 1))
        scores = scores.view(scores.shape[0], scores.shape[2])
        return scores

# Load data
data_dir = 'data/train.csv'
skipgram_data = SkipGramData(data_dir,1)
skipgram_data.get_data()
 

def train_skipgram(device,data, model, optimizer, criterion, num_epochs,batch_size):
    model.train()
    loss_ = []
    X, Y = skipgram_data.get_positive_pairs()
    print(f'Training on {len(X)} samples')
    for epoch in range(num_epochs):
        total_loss = []
        for i in range(0, len(X), batch_size):
            X_batch = torch.tensor(X[i:i+batch_size]).to(device)
            Y_batch = torch.tensor(Y[i:i+batch_size]).to(device)
            negative_samples = torch.tensor(data.get_negative_samples(len(X_batch)))
            context = torch.cat((Y_batch.view(-1, 1), negative_samples), dim=1).to(device)
            labels_pos = torch.ones(Y_batch.shape[0], 1).to(device)
            labels_neg = torch.zeros(Y_batch.shape[0], data.k).to(device)
            labels = torch.cat((labels_pos, labels_neg), dim=1)
            optimizer.zero_grad()
            output = model(X_batch, context)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            total_loss.append(loss.item())
        loss_.append(np.mean(total_loss))
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {np.mean(total_loss)}')
    return model, loss_

# Hyperparameters
vocab_size = len(skipgram_data.vocab)
embedding_dim = skipgram_data.embedding_dim #300
num_epochs = 5
batch_size = 32
lr = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize model
model = SkipGram(vocab_size, embedding_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.BCEWithLogitsLoss()

# Train model
model, loss = train_skipgram(device,skipgram_data, model, optimizer, criterion, num_epochs, batch_size)

# Save model
torch.save(model.state_dict(), 'skipgram_model.pt')

# Save word embeddings
word_embeddings = model.in_embed.weight.data.cpu().numpy()
torch.save(word_embeddings, 'skipgram-word-embeddings-1.pt')

# window_size = 2
# skipgram_data.window_size = 2
# skipgram_data.get_data()
# vocab_size = len(skipgram_data.vocab)
# model = SkipGram(vocab_size, embedding_dim).to(device)
# optimizer = optim.Adam(model.parameters(), lr=lr)
# criterion = nn.BCEWithLogitsLoss()
# model, loss = train_skipgram(device,skipgram_data, model, optimizer, criterion, num_epochs, batch_size)
# word_embeddings = model.in_embed.weight.data.cpu().numpy()
# torch.save(word_embeddings, 'skipgram-word-embeddings-2.pt')

# # window_size = 3
# skipgram_data.window_size = 3
# skipgram_data.get_data()
# vocab_size = len(skipgram_data.vocab)
# model = SkipGram(vocab_size, embedding_dim).to(device)
# optimizer = optim.Adam(model.parameters(), lr=lr)
# criterion = nn.BCEWithLogitsLoss()
# model, loss = train_skipgram(device,skipgram_data, model, optimizer, criterion, num_epochs, batch_size)
# word_embeddings = model.in_embed.weight.data.cpu().numpy()
# torch.save(word_embeddings, 'skipgram-word-embeddings-3.pt')

# Save word2idx and idx2word
torch.save(skipgram_data.word2idx, 'skipgram-word2idx.pt')
torch.save(skipgram_data.idx2word, 'skipgram-idx2word.pt')

# Save model
torch.save(model, 'skipgram-model.pt')









