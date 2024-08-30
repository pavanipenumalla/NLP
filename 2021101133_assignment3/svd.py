import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.tokenize import word_tokenize
import torch 

class SVD:
    def __init__(self, data_dir,context_window):
        self.data_dir = data_dir
        self.threshold =  3
        self.sentences = None
        self.vocab = None
        self.word2idx = None
        self.idx2word = None
        self.co_occurrence_matrix = None
        self.U = None
        self.embedding_dim = 300
        self.word_embeddings = None
        self.context_window = context_window

    def load_data(self):
        data = pd.read_csv(self.data_dir)
        data = data['Description']
        data = data[:15000]
        self.sentences = data.tolist()

    def preprocess(self):
        self.sentences = [sentence.lower() for sentence in self.sentences]
        self.sentences = [re.sub(r'http\S+', 'URL', sentence) for sentence in self.sentences]
        self.sentences = [re.sub(r'www\S+', 'URL', sentence) for sentence in self.sentences]
        self.sentences = [sentence.translate(str.maketrans('', '', string.punctuation)) for sentence in self.sentences]
        self.sentences = [['<S>'] + word_tokenize(sentence) + ['</S>'] for sentence in self.sentences]

    def create_vocab(self):
        self.vocab = {}
        for sentence in self.sentences:
            for word in sentence:
                if word in self.vocab:
                    self.vocab[word] += 1
                else:
                    self.vocab[word] = 1

    def add_unk(self):
        unk_count = 0
        for word in list(self.vocab.keys()):
            if self.vocab[word] < self.threshold:
                unk_count += self.vocab[word]
                del self.vocab[word]
        self.vocab['<UNK>'] = unk_count

    def create_word_index_mappings(self):
        self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

    def create_co_occurrence_matrix(self):
        self.co_occurrence_matrix = np.zeros((len(self.word2idx), len(self.word2idx)))
        for sentence in self.sentences:
            for i in range(1, len(sentence)):
                if sentence[i - 1] in self.word2idx:
                    idx1 = self.word2idx[sentence[i - 1]]
                else:
                    idx1 = self.word2idx['<UNK>']
                start_idx = max(0, i - self.context_window)
                end_idx = min(len(sentence), i + self.context_window)
                for j in range(start_idx, end_idx):
                    if i == j:
                        continue
                if sentence[i] in self.word2idx:
                    idx2 = self.word2idx[sentence[i]]
                else:
                    idx2 = self.word2idx['<UNK>']
                self.co_occurrence_matrix[idx1][idx2] += 1
                self.co_occurrence_matrix[idx2][idx1] += 1

    def svd_func(self):
        self.U, S, V = np.linalg.svd(self.co_occurrence_matrix)
        return self.U

    def get_embeddings(self):
        self.load_data()
        self.preprocess()
        self.create_vocab()
        self.add_unk()
        self.create_word_index_mappings()
        self.create_co_occurrence_matrix()
        self.U = self.svd_func()
        self.word_embeddings = self.U[:, :self.embedding_dim]

    def get_word_embedding(self, word):
        if word in self.word2idx:
            idx = self.word2idx[word]
        else:
            idx = self.word2idx['<UNK>']
        return self.word_embeddings[idx]

    def save_embeddings(self, path):
        torch.save(self.word_embeddings, path)

    def save_word2idx(self):
        torch.save(self.word2idx, "svd-word2idx.pt")
        torch.save(self.idx2word, "svd-idx2word.pt")

    def save_model(self):
        torch.save(self, "svd-model.pt")
        
svd = SVD('data/train.csv', 1)
svd.get_embeddings()
svd.save_embeddings('svd-word-vectors-1.pt')

# svd = SVD('data/train.csv',2)
# svd.get_embeddings
# svd.save_embeddings('svd-word-vectors-2.pt')
# svd = SVD('data/train.csv',3)
# svd.get_embeddings()
# svd.save_embeddings('svd-word-vectors-3.pt')
# svd.save_word2idx()
# svd.save_model()

