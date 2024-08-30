from utils import preprocess, get_data, train
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


sentences,labels = preprocess('data/train.csv')
word2idx = {word : idx for idx, word in enumerate(set([word for sentence in sentences for word in sentence]))}
word2idx['<pad>'] = len(word2idx)
 
torch.save(word2idx, 'word2idx.pt')

# word2idx = torch.load('word2idx.pt')
X_train_f, y_train_f, X_train_b, y_train_b, num_classes = get_data(sentences, word2idx)
print(X_train_f.size(), y_train_f.size(), X_train_b.size(), y_train_b.size(), num_classes)

# hyperparameters
batch_size = 64
embedding_dim = 300
hidden_dim = 300
num_layers = 1
learning_rate = 0.001
num_epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# train loader
train_data = TensorDataset(X_train_f, y_train_f, X_train_b, y_train_b)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

losses , model = train(X_train_f, y_train_f, X_train_b, y_train_b, num_classes, batch_size, embedding_dim, hidden_dim, num_layers, learning_rate, num_epochs, train_loader, device)

torch.save(model, 'model.pt')