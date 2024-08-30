import torch
import wandb
import torch.nn as nn
import torch.nn.functional as F
from io import open
import numpy as np
from torch.utils.data import Dataset, DataLoader
from conllu import parse_incr
from classes import Create_Dataset, FFNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
p = 1
s = 1

file = open("UD_English-Atis/en_atis-ud-train.conllu", "r", encoding="utf-8")
sentences = list(parse_incr(file))
file.close()

train_data = Create_Dataset(sentences,p,s,N=0)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

file = open("UD_English-Atis/en_atis-ud-dev.conllu", "r", encoding="utf-8")
sentences = list(parse_incr(file))
file.close()

val_data = Create_Dataset(sentences,p,s,vocab=train_data.vocab,upos_tags=train_data.upos_tags,N=0)
val_loader = DataLoader(val_data, batch_size=32, shuffle=True)

def train_FFNN(vocab_size, embedding_dim, hidden_dim, output_dim, p,s, vocab, upos_tags, num_layers, activation_fn, epochs,learning_rate):
    model = FFNN(vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, activation_fn, p,s,vocab,upos_tags).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    n_epochs = epochs

    val_loss = []
    train_loss = []
    for epoch in range(n_epochs):
        model.train()
        T_loss = 0
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            outputs = outputs.float()
            targets = targets.float()
            loss.backward()
            optimizer.step()
            T_loss+=loss.item()
            if i % 100 == 0:
                print(f"Epoch {epoch+1}, step {i}, loss: {T_loss/(i+1)}")
        train_loss.append(T_loss/len(train_loader))
        print(f"Epoch {epoch+1}, loss: {T_loss/len(train_loader)}")

        model.eval()
        with torch.no_grad():
            V_loss = 0
            for i , (inputs, targets) in enumerate(val_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                V_loss+=loss.item()
            print(f"Validation loss: {V_loss/len(val_loader)}")
            val_loss.append(V_loss/len(val_loader))
            print("-----------------------------------")

    return train_loss, val_loss,model

# Best hyperparameters
# num_layers: 1
# hidden_dim: 64
# embedding_dim: 128
# lr: 0.001
# activation_fn: tanh
# epochs: 5
 
train_loss,val_loss,model = train_FFNN(vocab_size=len(train_data.vocab), embedding_dim=64, hidden_dim=128, output_dim=len(train_data.upos_tags), p=p, s=s, vocab=train_data.vocab, upos_tags=train_data.upos_tags, num_layers=1, activation_fn=F.tanh, epochs=5,learning_rate=0.001)

torch.save(model, "model1.pth")
np.save("vocab1.npy",train_data.vocab)
np.save("upos_tags1.npy",train_data.upos_tags)


import matplotlib.pyplot as plt
plt.plot(train_loss, label="train loss")
plt.plot(val_loss, label="validation loss")
plt.legend()
plt.show()
