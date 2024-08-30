from utils import LSTM, train, predict, evaluate, get_data
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')
word_embeddings = torch.load('word_embeddings_svd_3.pt')
word2idx = torch.load('word2idx_svd.pt')
idx2word = torch.load('idx2word_svd.pt')
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_dimensions = 300
hidden_dimensions = 128
output_dimensions = n_classes
bidirectional = True
n_layers = 2

model = LSTM(input_dimensions, hidden_dimensions, output_dimensions, n_layers, bidirectional, device, activation_fn=torch.relu)

num_epochs = 10
learning_rate = 0.001

model, train_losses, val_losses = train(model, device, train_loader, val_loader, num_epochs, learning_rate)

trained_model = model.eval()

train_preds, train_gt = predict(trained_model, device, train_loader)
val_preds, val_gt = predict(trained_model, device, val_loader)
test_preds, test_gt = predict(trained_model, device, test_loader)

train_accuracy, train_f1, train_precision, train_recall, train_cm = evaluate(train_preds, train_gt)
val_accuracy, val_f1, val_precision, val_recall, val_cm = evaluate(val_preds, val_gt)
test_accuracy, test_f1, test_precision, test_recall, test_cm = evaluate(test_preds, test_gt)

print(f'Train Accuracy: {train_accuracy:.4f}, F1: {train_f1:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}')
print(f'Val Accuracy: {val_accuracy:.4f}, F1: {val_f1:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}, F1: {test_f1:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}')

print('Train Confusion Matrix:')
print(train_cm)
print('Val Confusion Matrix:')
print(val_cm)
print('Test Confusion Matrix:')
print(test_cm)


 




