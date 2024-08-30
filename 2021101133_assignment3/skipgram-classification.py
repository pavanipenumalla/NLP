import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from utils import train, predict, evaluate, LSTM, get_data
from torch.utils.data import DataLoader, TensorDataset

train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

word_embeddings = torch.load('word_embeddings_skip_gram_3.pt')
word2idx = torch.load('word2idx_skip_gram.pt')
idx2word = torch.load('idx2word_skip_gram.pt')

word_embeddings = torch.tensor(word_embeddings)
word_embeddings = torch.cat((word_embeddings, torch.zeros(1, 300)), dim=0)

word2idx['<PAD>'] = len(word2idx)
idx2word[len(idx2word)] = '<PAD>'

X_train, Y_train, X_val, Y_val, n_classes = get_data(train_data, word_embeddings, word2idx)
X_test, Y_test, n_classes = get_data(test_data, word_embeddings, word2idx, flag=True)

train_data = TensorDataset(X_train, Y_train)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

val_data = TensorDataset(X_val, Y_val)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

test_data = TensorDataset(X_test, Y_test)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

input_dim = 300
hidden_dim = 128
output_dim = n_classes
n_layers = 2
bidirectional = True
n_epochs = 10
lr = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = LSTM(input_dim, hidden_dim, output_dim, n_layers, bidirectional, device, activation_fn=torch.relu)

train_losses, val_losses, model_trained = train(model, device, train_loader, val_loader, n_epochs, lr)

model_trained.eval()

train_predictions, train_gt = predict(model_trained, device, train_loader)
val_predictions, val_gt = predict(model_trained, device, val_loader)
test_predictions, test_gt = predict(model_trained, device, test_loader)

train_accuracy, train_f1, train_precision, train_recall, train_cm = evaluate(train_predictions, train_gt)
val_accuracy, val_f1, val_precision, val_recall, val_cm = evaluate(val_predictions, val_gt)
test_accuracy, test_f1, test_precision, test_recall, test_cm = evaluate(test_predictions, test_gt)

print(f'Train Accuracy: {train_accuracy:.4f}, F1: {train_f1:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}')
print(f'Val Accuracy: {val_accuracy:.4f}, F1: {val_f1:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}, F1: {test_f1:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}')

print('Train Confusion Matrix:')
print(train_cm)
print('Val Confusion Matrix:')
print(val_cm)
print('Test Confusion Matrix:')
print(test_cm)




