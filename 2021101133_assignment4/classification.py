import pandas as pd
import torch
import sys
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from utils import preprocess
from classification_utils import get_classification_data, ClassifierLSTM_1, ClassifierLSTM_2, ClassifierLSTM_3,train_classification, get_test_results

def main(classifier_choice):
    sentences, labels = preprocess('data/train.csv')
    word2idx = torch.load('word2idx.pt')
    print("word2idx Loaded!")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elmo_model = torch.load('model.pt')
    print("ELMO Loaded!")

    X_train, y_train, X_val, y_val = get_classification_data(sentences, labels, word2idx)
    print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)

    # hyperparameters
    embedding_dim = 600
    hidden_dim = 300
    num_layers = 2
    num_classes = len(set(labels))
    batch_size = 64
    activation_fn = nn.ReLU()
    bidirectional = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if classifier_choice == 1:
        model = ClassifierLSTM_1(embedding_dim, hidden_dim, num_layers, num_classes, batch_size, bidirectional, activation_fn, device).to(device)
    elif classifier_choice == 2:
        model = ClassifierLSTM_2(embedding_dim, hidden_dim, num_layers, num_classes, batch_size, bidirectional, activation_fn, device).to(device)
    else:
        model = ClassifierLSTM_3(embedding_dim, hidden_dim, num_layers, num_classes, batch_size, bidirectional, activation_fn, device).to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Training Model...")
    trained_model = train_classification(X_train, y_train, X_val, y_val, model, elmo_model, criterion, optimizer, 5, batch_size, device,flag=0)
    torch.save(trained_model, f'classification_model_{classifier_choice}.pt')

    sentences, labels = preprocess('data/test.csv')
    X_test, y_test = get_classification_data(sentences, labels, word2idx, test_flag=True)

    classifier = torch.load(f'classification_model_{classifier_choice}.pt')
    get_test_results(classifier, X_test, y_test, elmo_model, device)

if __name__ == "__main__":
    classifier_choice = int(sys.argv[1])
    main(classifier_choice)
