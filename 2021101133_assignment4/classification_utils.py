from utils import preprocess, ELMO
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import  DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def get_classification_data(sentences,labels, word2idx, test_flag = False):
    num_classes = len(set(labels))
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

    X = torch.tensor(sentences_idx)
    y = [label-1 for label in labels]
    y = torch.tensor(y)
    y = nn.functional.one_hot(y, num_classes = num_classes).float()
    # print(labels[:5],y[:5])

    if test_flag:
        return X,y
    else:
        indices = torch.randperm(X.size(0))
        X = X[indices]
        y = y[indices]
        split = int(0.8*len(X))
        X_train = X[:split]
        y_train = y[:split]
        X_val = X[split:]
        y_val = y[split:]
        return X_train,y_train,X_val,y_val

# trainable lambdas
class ClassifierLSTM_1(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, num_classes, batch_size,bidirectional, activation_fn,device):
        super(ClassifierLSTM_1, self).__init__()     
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,bidirectional=bidirectional, batch_first = True)
        self.fc = nn.Linear(hidden_dim*2, num_classes) if bidirectional else nn.Linear(hidden_dim, num_classes)
        self.lambda1 = nn.Parameter(torch.rand(1), requires_grad=True)
        self.lambda2 = nn.Parameter(torch.rand(1), requires_grad=True)
        self.lambda3 = nn.Parameter(torch.rand(1), requires_grad=True)
        self.batch_size = batch_size
        self.device = device
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers    
        self.bidirectional = bidirectional
        self.activation_fn = activation_fn

    def forward(self, e0,h1,h2):
        X = self.lambda1*e0 + self.lambda2*h1 + self.lambda3*h2
        h0 = torch.zeros(self.num_layers*2 if self.bidirectional else 1, X.size(0), self.hidden_dim).to(self.device)
        c0 = torch.zeros(self.num_layers*2 if self.bidirectional else 1, X.size(0), self.hidden_dim).to(self.device)
        out, _ = self.lstm(X, (h0, c0))
        out = self.activation_fn(out)
        out = self.fc(out[:,-1,:])
        return out
    
# frozen lambdas
class ClassifierLSTM_2(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, num_classes, batch_size,bidirectional, activation_fn,device):
        super(ClassifierLSTM_2, self).__init__() 
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,bidirectional=bidirectional, batch_first = True)
        self.fc = nn.Linear(hidden_dim*2, num_classes) if bidirectional else nn.Linear(hidden_dim, num_classes)
        self.lambda1 = nn.Parameter(torch.rand(1), requires_grad=False)
        self.lambda2 = nn.Parameter(torch.rand(1), requires_grad=False)
        self.lambda3 = nn.Parameter(torch.rand(1), requires_grad=False)
        self.batch_size = batch_size
        self.device = device
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers    
        self.bidirectional = bidirectional
        self.activation_fn = activation_fn

    def forward(self, e0,h1,h2):
        X = self.lambda1*e0 + self.lambda2*h1 + self.lambda3*h2
        h0 = torch.zeros(self.num_layers*2 if self.bidirectional else 1, X.size(0), self.hidden_dim).to(self.device)
        c0 = torch.zeros(self.num_layers*2 if self.bidirectional else 1, X.size(0), self.hidden_dim).to(self.device)
        out, _ = self.lstm(X, (h0, c0))
        out = self.activation_fn(out)
        out = self.fc(out[:,-1,:])
        return out
    
class FFNN(nn.Module):
    def __init__(self, input_dim, output_dim, activation_fn):
        super(FFNN, self).__init__()
        self.fc1 = nn.Linear(input_dim,output_dim)
        self.activation_fn = activation_fn

    def forward(self, X):
        out = self.fc1(X)
        out = self.activation_fn(out)
        return out
    
# learnable function
class ClassifierLSTM_3(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, num_classes, batch_size,bidirectional, activation_fn,device):
        super(ClassifierLSTM_3, self).__init__() 
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,bidirectional=bidirectional, batch_first = True)
        self.fc = nn.Linear(hidden_dim*2, num_classes) if bidirectional else nn.Linear(hidden_dim, num_classes)
        self.ffnn = FFNN(embedding_dim*3, embedding_dim, activation_fn)
        self.batch_size = batch_size
        self.device = device
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.activation_fn = activation_fn
        
    def forward(self, e0,h1,h2):
        X = torch.cat((e0,h1,h2), dim = 2)
        X = self.ffnn(X)
        h0 = torch.zeros(self.num_layers*2 if self.bidirectional else 1, X.size(0), self.hidden_dim).to(self.device)
        c0 = torch.zeros(self.num_layers*2 if self.bidirectional else 1, X.size(0), self.hidden_dim).to(self.device)
        out, _ = self.lstm(X, (h0, c0))
        out = self.activation_fn(out)
        out = self.fc(out[:,-1,:])
        return out


def train_classification(X_train,y_train,X_val,y_val, model, elmo_model,criterion, optimizer, num_epochs, batch_size, device, flag = 1):
    train_data = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_data = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        accuracy = 0
        correct = []
        preds = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            X_flip = torch.flip(X_batch, [1])
            ef = elmo_model.embedding(X_batch)
            eb = elmo_model.embedding(X_flip)
            hf1, _ = elmo_model.lstm_f1(ef)
            hf2, _ = elmo_model.lstm_f2(hf1)
            hb1, _ = elmo_model.lstm_b1(eb)
            hb2, _ = elmo_model.lstm_b2(hb1)

            # X_batch = e0 + (hf1+hf2) + (hb1+hb2)
            # concatenating e0,e0
            e0 = torch.cat((ef, eb), dim = 2)
            # concatenating hf1,hf2
            h1 = torch.cat((hf1, hb1), dim = 2)
            # concatenating hb1,hb2
            h2 = torch.cat((hf2, hb2), dim = 2)

            
            optimizer.zero_grad()
            # output = model(X_batch)
            output = model(e0, h1, h2)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(output, 1)
            actual = torch.argmax(y_batch, 1)
            preds.extend(predicted.cpu().numpy())
            correct.extend(actual.cpu().numpy())

        accuracy = accuracy_score(correct, preds)
        train_loss /= len(train_loader)
       
        model.eval()
        val_loss = 0
        val_correct = []
        val_accuracy = 0
        val_preds = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                X_flip = torch.flip(X_batch, [1])
                ef = elmo_model.embedding(X_batch)
                eb = elmo_model.embedding(X_flip)
                hf1, _ = elmo_model.lstm_f1(ef)
                hf2, _ = elmo_model.lstm_f2(hf1)
                hb1, _ = elmo_model.lstm_b1(eb)
                hb2, _ = elmo_model.lstm_b2(hb1)
                e0 = torch.cat((ef, eb), dim = 2)
                # concatenating hf1,hf2
                h1 = torch.cat((hf1, hb1), dim = 2)
                # concatenating hb1,hb2
                h2 = torch.cat((hf2, hb2), dim = 2)

                # X_batch = e0 + (hf1+hf2) + (hb1+hb2)
                # output = model(X_batch)
                output = model(e0, h1, h2)
                loss = criterion(output, y_batch)
                val_loss += loss.item()
                _, predicted = torch.max(output, 1)
                actual = torch.argmax(y_batch, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_correct.extend(actual.cpu().numpy())
            
            val_accuracy = accuracy_score(val_correct, val_preds)
            val_loss /= len(val_loader)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

        # printing lambda values
        if epoch == num_epochs-1:
            if flag == 1:
                print(f'Lambda Values: {model.lambda1.item(), model.lambda2.item(), model.lambda3.item()}')
            # printing final metrics on train and validation data
            # metrics are accuracy, precision, recall, f1 score, confusion matrix
            print("----------------------------------------------")
            print(f'Final Metrics on Train Data:')
            print("----------------------------------------------")
            print(f'Accuracy: {accuracy_score(correct, preds)}')
            print(f'Precision: {precision_score(correct, preds, average="weighted")}')
            print(f'Recall: {recall_score(correct, preds, average="weighted")}')
            print(f'F1 Score: {f1_score(correct, preds, average="weighted")}')
            print(f'Confusion Matrix:')
            print(confusion_matrix(correct, preds))
            print("----------------------------------------------")
            print(f'Final Metrics on Validation Data:')
            print("----------------------------------------------")
            print(f'Accuracy: {accuracy_score(val_correct, val_preds)}')
            print(f'Precision: {precision_score(val_correct, val_preds, average="weighted")}')
            print(f'Recall: {recall_score(val_correct, val_preds, average="weighted")}')
            print(f'F1 Score: {f1_score(val_correct, val_preds, average="weighted")}')
            print(f'Confusion Matrix:')
            print(confusion_matrix(val_correct, val_preds))
    return model


def get_test_results(model,X_test,y_test,  elmo_model, device):
    model.eval()
    test_data = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
    test_preds = []
    test_correct = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            X_flip = torch.flip(X_batch, [1])
            ef = elmo_model.embedding(X_batch)
            eb = elmo_model.embedding(X_flip)
            hf1, _ = elmo_model.lstm_f1(ef)
            hf2, _ = elmo_model.lstm_f2(hf1)
            hb1, _ = elmo_model.lstm_b1(eb)
            hb2, _ = elmo_model.lstm_b2(hb1)
            e0 = torch.cat((ef, eb), dim = 2)
            h1 = torch.cat((hf1, hb1), dim = 2)
            h2 = torch.cat((hf2, hb2), dim = 2)
            output = model(e0, h1, h2)
            _, predicted = torch.max(output, 1)
            actual = torch.argmax(y_batch, 1)
            test_preds.extend(predicted.cpu().numpy())
            test_correct.extend(actual.cpu().numpy())
        
        print("----------------------------------------------")
        print(f'Final Metrics on Test Data:')
        print("----------------------------------------------")
        print(f'Accuracy: {accuracy_score(test_correct, test_preds)}')
        print(f'Precision: {precision_score(test_correct, test_preds, average="weighted")}')
        print(f'Recall: {recall_score(test_correct, test_preds, average="weighted")}')
        print(f'F1 Score: {f1_score(test_correct, test_preds, average="weighted")}')
        print(f'Confusion Matrix:')
        print(confusion_matrix(test_correct, test_preds))
        # return test_preds, test_correct
