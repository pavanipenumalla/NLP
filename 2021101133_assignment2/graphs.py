import torch
import torch.nn as nn
import numpy as np
from conllu import parse_incr
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from classes import Create_Dataset, FFNN
from io import open

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32
p = [1,2,3,4]
loss_ = []
acc = []

vocab = np.load("vocab1.npy", allow_pickle=True).item()
upos_tags = np.load("upos_tags1.npy", allow_pickle=True).item()

for val in p:

    file = open("UD_English-Atis/en_atis-ud-train.conllu", "r", encoding="utf-8")
    sentences = list(parse_incr(file))
    file.close()
    train_data = Create_Dataset(sentences, val, val, N=0)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    file = open("UD_English-Atis/en_atis-ud-dev.conllu", "r", encoding="utf-8")
    sentences = list(parse_incr(file))
    file.close()
    val_data = Create_Dataset(sentences, val, val, vocab=train_data.vocab, upos_tags=train_data.upos_tags, N=0)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    num_layers= 1
    hidden_dim=  64
    embedding_dim= 128
    lr= 0.001
    activation_fn= nn.Tanh()
    epochs= 5

    model = FFNN(len(vocab), embedding_dim, hidden_dim, len(upos_tags), num_layers, activation_fn, val, val, vocab, upos_tags).to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    val_loss = []
    train_loss = []
    val_acc = []
    for epoch in range(epochs):
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
            correct = 0
            total = 0
            for i , (inputs, targets) in enumerate(val_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                V_loss+=loss.item()
                total += targets.size(0)
                correct += (outputs.argmax(1) == targets.argmax(1)).sum().item()
            print(f"Validation loss: {V_loss/len(val_loader)}")
            val_loss.append(V_loss/len(val_loader))
            val_acc.append(correct/total)
            print("-----------------------------------")

    loss_.append(val_loss[-1])
    acc.append(val_acc[-1])

plt.plot(p, loss_)
plt.xlabel("p")
plt.ylabel("Validation Loss")
plt.show()
plt.plot(p, acc)
plt.xlabel("p")
plt.ylabel("Validation Accuracy")
plt.show()

        

    
     
