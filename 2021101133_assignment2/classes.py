import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

START_TOKEN = "<s>"
END_TOKEN = "</s>"
UNKNOWN_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Create_Dataset(Dataset):
    def __init__(self,sent,p,s,N,vocab=None,upos_tags=None): # nn : 0 if FFNN and 1 if LSTM
        self.p = p
        self.s = s
        self.N = N
        self.sent = self.add_unks(sent, vocab)
        if vocab is not None:
            self.vocab = vocab
        else:
            self.vocab = self.create_vocab()   
        if upos_tags is not None:
            self.upos_tags = upos_tags
        else:
            self.upos_tags = self.create_upos_tags()
        self.X,self.Y = self.create_data()

    def add_unks(self,sent,vocab):
        if vocab is None:
            V = {}
            for s in sent:
                for t in s:
                    if t["form"] not in V:
                        V[t["form"]] = 1
                    else:
                        V[t["form"]] += 1
            for s in sent:
                for t in s:
                    if V[t["form"]] <= 3:
                        t["form"] = UNKNOWN_TOKEN

        else:
            for s in sent:
                for t in s:
                    if t["form"] not in vocab:
                        t["form"] = UNKNOWN_TOKEN
        return sent

    def create_vocab(self):
        vocab = set()
        for s in self.sent:
            for t in s:
                vocab.add(t["form"])
        vocab.add(START_TOKEN)
        vocab.add(END_TOKEN)
        vocab.add(UNKNOWN_TOKEN)
        vocab.add(PAD_TOKEN)
        vocab = list(vocab)
        vocab = {word: i for i, word in enumerate(vocab)}
        print(vocab[START_TOKEN])
        return vocab

    
    def create_upos_tags(self):
        upos_tags = set()
        for s in self.sent:
            for t in s:
                upos_tags.add(t["upos"])
        upos_tags.add("UNK")
        upos_tags = list(upos_tags)
        upos_tags = {upos: i for i, upos in enumerate(upos_tags)}
        return upos_tags
    
    def create_data(self):
        X = []
        Y = []
        if self.N == 0:
            for st in self.sent:
                for i in range(self.p):
                    st.insert(0,{"form":START_TOKEN,"upos": UNKNOWN_TOKEN})
                for i in range(self.s):
                    st.append({"form":END_TOKEN,"upos": UNKNOWN_TOKEN})
                for i in range(self.p,len(st)-self.s):
                    # input is of size p+s+1 and output is of size 1
                    X.append([st[j]["form"] for j in range(i-self.p,i+self.s+1)])
                    Y.append(st[i]["upos"])

            X = [[self.vocab[word] for word in x] for x in X]
            Y = [self.upos_tags[upos] if upos in self.upos_tags else self.upos_tags["UNK"] for upos in Y]
            Y = np.eye(len(self.upos_tags))[Y]
            print(len(X),len(Y))
            return X,Y
        
        elif self.N == 1:
            # for LSTM
            # add one start and end token to each sentence and pad the sentences to the length of the longest sentence
            for st in self.sent:
                st.insert(0,{"form":START_TOKEN,"upos": UNKNOWN_TOKEN})
                st.append({"form":END_TOKEN,"upos": UNKNOWN_TOKEN})
            max_len = max([len(s) for s in self.sent])
            for st in self.sent:
                while len(st) < max_len:
                    st.append({"form":PAD_TOKEN,"upos": UNKNOWN_TOKEN})

            X = [[self.vocab[word["form"]] for word in st] for st in self.sent]
            # Y = [self.upos_tags[upos] if upos in self.upos_tags else self.upos_tags["UNK"] for upos in Y]
            Y = [[self.upos_tags[word["upos"]] if word["upos"] in self.upos_tags else self.upos_tags["UNK"] for word in st] for st in self.sent]
            Y = np.eye(len(self.upos_tags))[Y]
            print(len(X),len(Y))
            return X,Y
             
             
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self,idx):
        return torch.tensor(self.X[idx]),torch.tensor(self.Y[idx])
    
class FFNN(nn.Module):
    def __init__(self,input_dim, embedding_dim, hidden_dim, output_dim,num_layers,activation_fn, p, s, vocab, upos_tags):
        super(FFNN, self).__init__()
        self.p = p
        self.s = s
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(input_dim,embedding_dim)
        self.hiddenum_layers = nn.ModuleList([nn.Linear(hidden_dim,hidden_dim) for i in range(num_layers-1)])
        self.fc1 = nn.Linear((p+s+1)*embedding_dim,hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,output_dim)
        self.activation_fn = activation_fn
        self.vocab = vocab
        self.upos_tags = upos_tags

    def forward(self, x):
        x = x.view(-1)
        x = self.embedding(x)
        x = x.view(-1,(self.p+self.s+1)*self.embedding_dim)
        # print(x.shape)
        x = self.fc1(x)
        x = self.activation_fn(x)
        for layer in self.hiddenum_layers:
            x = layer(x)
            x = self.activation_fn(x)
        x = self.fc2(x)
        return x
    
class LSTM(nn.Module):
    def __init__(self,input_dim,embedding_dim,hidden_dim,output_dim,num_layers,bidirectional,activation_fn):
        super(LSTM,self).__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.embedding = nn.Embedding(input_dim,embedding_dim)
        self.rnn = nn.LSTM(embedding_dim,hidden_dim,num_layers=num_layers,bidirectional=bidirectional)
        self.activation_fn = activation_fn
        self.fc = nn.Linear(hidden_dim*2,output_dim) if bidirectional else nn.Linear(hidden_dim,output_dim)


    def forward(self,x):
        embedded = self.embedding(x)
        hidden_layers = (torch.zeros(self.num_layers*2,embedded.size(1),self.hidden_dim),torch.zeros(self.num_layers*2,embedded.size(1),self.hidden_dim)) if self.bidirectional else (torch.zeros(self.num_layers,embedded.size(1),self.hidden_dim),torch.zeros(self.num_layers,embedded.size(1),self.hidden_dim)).to(device)
        output, (hidden,cell) = self.rnn(embedded,hidden_layers)
        out = self.activation_fn(output)
        out = self.fc(out)
        return out
    
 