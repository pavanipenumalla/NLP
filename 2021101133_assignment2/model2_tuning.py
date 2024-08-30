import torch
import torch.nn as nn
import torch.nn.functional as F
from conllu import parse_incr
from classes import Create_Dataset, LSTM
from model2_train import train_LSTM
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

sweep_config = {
    "name": "LSTM",
    "method": "grid",
    "parameters": {
        "num_layers": {
            "values": [1, 2, 3]
        },
        "hidden_dim": {
            "values": [64,128,256]
        },
        "embedding_dim": {
            "values": [64,128,256]
        },
        "lr": {
            "values": [0.001,0.01,0.1]
        },
        "activation_fn": {
            "values": ["relu","tanh"]
        },
        "epochs": {
            "values": [5,10,15]
        },
        "bidirectional": {
            "values": [True, False]
        }
    },
    "metric": {"name": "val_loss", "goal": "minimize"},
}

sweep_id = wandb.sweep(sweep_config, project="INLP-LSTM")

def train():
    config_defaults = {
        "num_layers": 1,
        "hidden_dim": 100,
        "embedding_dim": 100,
        "lr": 0.001,
        "activation_fn": "relu",
        "epochs": 5,
        "bidirectional": True
    }
    activation_functions = {
        'relu': F.relu,
        'tanh': F.tanh
    }
    wandb.init(config=config_defaults)
    config = wandb.config
    train_loss, val_loss = train_LSTM(len(train_data.vocab), config.embedding_dim, config.hidden_dim, len(train_data.upos_tags), config.num_layers, config.bidirectional,activation_functions[config.activation_fn] , config.epochs, config.lr)
    wandb.log({"train_loss": train_loss[-1], "val_loss": val_loss[-1],"num_layers": config.num_layers, "hidden_dim": config.hidden_dim, "embedding_dim": config.embedding_dim, "lr": config.lr, "activation_fn": activation_functions[config.activation_fn], "epochs": config.epochs, "bidirectional": config.bidirectional})

wandb.agent(sweep_id, train)
