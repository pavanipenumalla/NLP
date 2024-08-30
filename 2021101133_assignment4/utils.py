import numpy as np
import pandas as pd
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import re
import contractions
import tqdm

def preprocess(file_path):
    data = pd.read_csv(file_path)
    sentences = data['Description'].values
    labels = data['Class Index'].values
    # sentences = sentences[:1000]
    sentences = [sentence.lower() for sentence in sentences]
    sentences = [sentence.split() for sentence in sentences]
    final_sentences = []
    for sentence in sentences:
        curr_sentence = []
        for i, word in enumerate(sentence):
            if word == ' ' or word == '':
                continue
            elif word.find('\\') != -1:
                tmp = word.split('\\')
                if tmp[0] != '':
                    curr_sentence.append(tmp[0])
                if tmp[1] != '':
                    curr_sentence.append(tmp[1])
            elif word.find('-') != -1:
                tmp = word.split('-')
                if tmp[0] != '':
                    curr_sentence.append(tmp[0])
                if tmp[1] != '':
                    curr_sentence.append(tmp[1])
            else:
                curr_sentence.append(word)
        final_sentences.append(curr_sentence)

    sentences = final_sentences    
    sentences = [[contractions.fix(word) for word in sentence] for sentence in sentences]
    sentences = [[re.sub(r'[^\w\s]', '', word) for word in sentence] for sentence in sentences]
    sentences = [[re.sub(r'http\S+', '<url>', word) for word in sentence] for sentence in sentences]
    sentences = [[re.sub(r'www\S+', '<url>', word) for word in sentence] for sentence in sentences] 
    sentences = [['<s>'] + sentence + ['</s>'] for sentence in sentences]

    vocab = {}
    for sentence in sentences:
        for word in sentence:
            if word not in vocab:
                vocab[word] = 1
            else:
                vocab[word] += 1

    for sentence in sentences:
        for i, word in enumerate(sentence):
            if vocab[word] < 3:
                sentence[i] = '<unk>'
    return sentences, labels



def get_data(sentences, word2idx):
    num_classes = len(word2idx)
    seq_len = [len(sentence) for sentence in sentences]
    max_seq_len = np.percentile(seq_len, 95)
    max_seq_len = int(max_seq_len)
    # print(f'Max sequence length: {max_seq_len}')

    padded_sentences = []
    for sentence in sentences:
        if len(sentence) < max_seq_len:
            sentence += ['<pad>'] * (max_seq_len - len(sentence))
        elif len(sentence) > max_seq_len:
            sentence = sentence[:max_seq_len]
        padded_sentences.append(sentence)

    sentences_idx = [[word2idx[word] if word in word2idx else word2idx['<unk>'] for word in sentence] for sentence in padded_sentences]

   
    X_train_f = [sentence[:-1] for sentence in sentences_idx]
    y_train_f = [sentence[1:] for sentence in sentences_idx]

    X_train_f = torch.tensor(X_train_f)
    y_train_f = torch.tensor(y_train_f)

     
    sentences_rev = [sentence[::-1] for sentence in sentences_idx]
    X_train_b = [sentence[:-1] for sentence in sentences_rev]
    y_train_b = [sentence[1:] for sentence in sentences_rev]
 
    X_train_b = torch.tensor(X_train_b)
    y_train_b = torch.tensor(y_train_b)

    return X_train_f, y_train_f, X_train_b, y_train_b, num_classes


class ELMO(nn.Module):
    def __init__(self, num_classes, embedding_dim, hidden_dim, num_layers, batch_size,device):
        super(ELMO, self).__init__()
        self.embedding = nn.Embedding(num_classes, embedding_dim)
        self.lstm_f1 = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.lstm_f2 = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        # backward lstm
        self.lstm_b1 = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.lstm_b2 = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.batch_size = batch_size
        self.device = device
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, X_f, X_b):
        # forward lstm 
        embed_f = self.embedding(X_f)
        out_f1, _ = self.lstm_f1(embed_f) 
        out_f2, _ = self.lstm_f2(out_f1)
        # linear layer
        out_f = self.fc(out_f2)
        # backward lstm
        embed_b = self.embedding(X_b)
        out_b1, _ = self.lstm_b1(embed_b)
        out_b2, _= self.lstm_b2(out_b1)
        # linear layer
        out_b = self.fc(out_b2)

        return  out_f, out_b

def train(X_train_f, y_train_f, X_train_b, y_train_b, num_classes, batch_size, embedding_dim, hidden_dim, num_layers, learning_rate, num_epochs,train_loader, device):
    model = ELMO(num_classes, embedding_dim, hidden_dim, num_layers,  batch_size, device).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    losses = []

    for epoch in range(num_epochs):  
        running_loss = 0
        for i, (X_f, y_f, X_b, y_b) in enumerate(train_loader):
            X_f = X_f.to(device)
            y_f = torch.nn.functional.one_hot(y_f, num_classes).float().to(device)
            X_b = X_b.to(device)
            y_b = torch.nn.functional.one_hot(y_b, num_classes).float().to(device)

            y_f = y_f.permute(0, 2, 1)
            y_b = y_b.permute(0, 2, 1)
            out_f, out_b = model(X_f, X_b)
            out_f = out_f.permute(0, 2, 1)
            out_b = out_b.permute(0, 2, 1)
            loss_f = criterion(out_f, y_f)
            loss_b = criterion(out_b, y_b)
            loss = loss_f + loss_b

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
             
        losses.append(running_loss/len(train_loader))
        print(f'Epoch: {epoch+1}/{num_epochs}, Loss: {losses[-1]}')
    
    return losses , model

     

