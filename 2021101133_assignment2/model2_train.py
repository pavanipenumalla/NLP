import torch
import torch.nn as nn
import torch.nn.functional as F
from conllu import parse_incr
from classes import Create_Dataset, LSTM
from torch.utils.data import DataLoader
import numpy as np
import wandb
from io import open

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

file = open("UD_English-Atis/en_atis-ud-train.conllu", "r", encoding="utf-8")
sentences = list(parse_incr(file))
file.close()

train_data = Create_Dataset(sentences, 2, 2, N=1)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

file = open("UD_English-Atis/en_atis-ud-dev.conllu", "r", encoding="utf-8")
sentences = list(parse_incr(file))
file.close()

val_data = Create_Dataset(sentences, p=2, s=2, vocab=train_data.vocab, upos_tags=train_data.upos_tags, N=1)
val_loader = DataLoader(val_data, batch_size=32, shuffle=True)
val_acc = []

def train_LSTM(input_dim, embedding_dim, hidden_dim, output_dim, num_layers, bidirectional, activation_fn, epochs,learning_rate):

    model = LSTM(input_dim, embedding_dim, hidden_dim, output_dim, num_layers = num_layers, bidirectional=bidirectional,activation_fn=activation_fn ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    n_epochs = epochs


    val_loss = []
    train_loss = []
    for epoch in range(n_epochs):
        model.train()
        T_loss = 0
        for i, (inputs, targets) in enumerate(train_loader):
            inputs,targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            outputs = outputs.view(-1, outputs.shape[-1])
            targets = targets.view(-1,targets.shape[-1])
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            T_loss += loss.item()
            if i % 100 == 0:
                print(f"Epoch {epoch+1}, step {i}, loss: {T_loss/(i+1)}")
        train_loss.append(T_loss/len(train_loader))
        print(f"Epoch {epoch+1}, loss: {T_loss/len(train_loader)}")

        model.eval()
        v_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(val_loader):
                inputs,targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                outputs = outputs.view(-1, outputs.shape[-1])
                targets = targets.view(-1,targets.shape[-1])
                loss = criterion(outputs, targets)
                v_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                _, targets = torch.max(targets, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        val_acc.append(correct/total)
        print(f"Validation loss: {v_loss/len(val_loader)}")
        val_loss.append(v_loss/len(val_loader))
        print("-----------------------------------")

    return train_loss, val_loss, model

# Best hyperparameters
# num_layers= 1
# hidden_dim= 128
# embedding_dim= 128
# lr= 0.001
# activation_fn= tanh
# epochs= 15
# bidirectional= True

train_loss, val_loss, model = train_LSTM(input_dim=len(train_data.vocab), embedding_dim=128, hidden_dim=128, output_dim=len(train_data.upos_tags), num_layers=1, bidirectional=True, activation_fn=F.tanh, epochs=15,learning_rate=0.001)

torch.save(model, "model2.pth")
np.save("vocab2.npy", train_data.vocab)
np.save("upos_tags2.npy", train_data.upos_tags)

import matplotlib.pyplot as plt
plt.plot(train_loss, label="Train Loss")
plt.plot(val_loss, label="Validation Loss")
plt.legend()
plt.show()

plt.plot(range(1,16), val_acc, label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Epochs vs Accuracy")
plt.legend()
plt.show()

plt.plot(range(1,16), val_loss, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Epochs vs Loss")
plt.legend()
plt.show()



