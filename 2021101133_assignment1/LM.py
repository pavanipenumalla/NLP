from tokenizer import tokenize
import random
import numpy as np
from scipy.stats import linregress
import pickle

class N_Gram_Model:
    def __init__(self,n,corpus):
        self.n = n
        self.corpus = corpus
        self.unigrams = {}
        self.bigrams = {}
        self.trigrams = {}
        self.unigrams_count = 0
        self.bigrams_count = 0
        self.trigrams_count = 0
        self.u_probabilities = {}
        self.b_probabilities = {}
        self.t_probabilities = {}
        self.probabilities_gt = {}
        self.u_probabilities_i = {}
        self.b_probabilities_i = {}
        self.t_probabilities_i = {}
        self.training_data = [[]]
        self.testing_data = [[]]
        self.a = None
        self.b = None
        self._Nr = {}
        self._Zr = {}
        self.smoothed_Nr = {}
        self.rstar = {}
        self.rlast = None


    def read_file(self):
        with open(self.corpus, 'r') as f:
            self.corpus = f.read()
        self.corpus = self.corpus.replace('\n',' ')
        self.corpus = self.corpus.lower()
        self.corpus = tokenize(self.corpus)

        for i in range(len(self.corpus)):
            self.corpus[i] = [token for token in self.corpus[i] if token.isalnum()]

        if self.n!=1:
            for i in range(len(self.corpus)):
                self.corpus[i] = ['<s>']*(self.n-1) + self.corpus[i] + ['</s>']
        return self.corpus
    
    def setup(self):
         
        random.shuffle(self.corpus)
        self.training_data = self.corpus[:int(0.8*len(self.corpus))]
        self.testing_data = self.corpus[int(0.8*len(self.corpus)):]

    def add_unk(self):
        vocab = {}
        for sent in self.training_data:
            for token in sent:
                if token not in vocab:
                    vocab[token] = 1
                else:
                    vocab[token] += 1

        for sent in self.training_data:
            for i in range(len(sent)):
                if vocab[sent[i]] == 1:
                    sent[i] = '<UNK>'

    
    def create_n_grams(self, data):
    # unigrams
        for sent in data:
            for token in sent:
                if token in self.unigrams:
                    self.unigrams[token] += 1
                else:
                    self.unigrams[token] = 1
                self.unigrams_count += 1
        # bigrams
            for i in range(len(sent)-1):
                bigram = tuple(sent[i:i+2])
                if bigram in self.bigrams:
                    self.bigrams[bigram] += 1
                else:
                    self.bigrams[bigram] = 1
                self.bigrams_count += 1
        # trigrams
            for i in range(len(sent)-2):
                trigram = tuple(sent[i:i+3])
                if trigram in self.trigrams:
                    self.trigrams[trigram] += 1
                else:
                    self.trigrams[trigram] = 1
                self.trigrams_count += 1


    def create_probabilities(self):
    # unigrams
        for unigram in self.unigrams:
            self.u_probabilities[unigram] = self.unigrams[unigram] / self.unigrams_count

        # bigrams
        for bigram in self.bigrams:
            denominator = self.unigrams.get(bigram[:-1], 0)
            if denominator > 0:
                self.b_probabilities[bigram] = self.bigrams[bigram] / denominator
            else:
                self.b_probabilities[bigram] = 0   
        # trigrams
        for trigram in self.trigrams:
            denominator = self.bigrams.get(trigram[:-1], 0)
            if denominator > 0:
                self.t_probabilities[trigram] = self.trigrams[trigram] / denominator
            else:
                self.t_probabilities[trigram] = 0   

    def Nr(self):
        N_r = {}
        for n_gram in self.trigrams:
            r = self.trigrams[n_gram]
            if r not in N_r:
                N_r[r] = 1
            else:
                N_r[r] += 1

        self._Nr = N_r

    def Zr(self):
        Z_r = {}
        sorted_N_r = sorted(self._Nr.keys())
        for r in sorted_N_r:
            if r==1:
                t = sorted_N_r[1]
                Z_r[r] = self._Nr[r]/(0.5 * t)
            elif r== sorted_N_r[-1]:
                q = sorted_N_r[-2]
                Z_r[r] = self._Nr[r]/ (r-q)
            else:
                q = sorted_N_r[sorted_N_r.index(r)-1]
                t = sorted_N_r[sorted_N_r.index(r)+1]
                Z_r[r] = self._Nr[r]/(0.5)*(t-q)
        self._Zr = Z_r

    def get_ab(self):
        x = np.log(list(self._Zr.keys()))
        y = np.log(list(self._Zr.values()))

        self.a, self.b, _, _, _ = linregress(x, y)
        
    def S_nr(self):
        Nr = {}
        self.r_last = max(self._Nr.keys())
        for r in range(0, self.r_last + 1):
            if r in self._Nr:
                continue
            else:
                Nr[r] = 0

        for r in range(0, self.r_last + 1):
            if r == 0:
                Nr[r] = 1
            elif r < 5:
                Nr[r] = self._Nr[r]
            else:
                Nr[r] = np.exp(self.a + self.b * np.log(r))
        
        Nr[self.r_last+1] = np.exp(self.a + self.b * np.log(self.r_last+1))

        self.smoothed_Nr = Nr

            


    def good_turing(self):
        self.Nr()
        self.Zr()
        self.get_ab()
        self.S_nr()

        r_star = {}
        
        for r in range(0, self.r_last + 1):
            r_star[r] = (r + 1) * self.smoothed_Nr[r + 1] / self.smoothed_Nr[r]


        self.rstar = r_star
        prob = {}

        for trigram in self.trigrams:
            num = r_star[self.trigrams[trigram]]
            den = 0
            for unigram in self.unigrams:
                x = tuple(trigram[:2]+ (unigram,))
                if x in self.trigrams:
                    den += r_star[self.trigrams[x]]
                else:
                    den += r_star[0]
            prob[trigram] = num/den
        
        self.probabilities_gt = prob
                

    def get_lambdas(self):
        lambdas = [0] * self.n
    
        for trigram in self.trigrams:
            bigram = tuple(trigram[:2]) 
            bi_ = tuple(trigram[-2:])
            unigram = trigram[-2]
            uni_ = trigram[-1]

            if self.bigrams[bigram] != 1:
                a = (self.trigrams[trigram] - 1) / (self.bigrams[bigram] - 1)
            else:
                a = 0

            if self.unigrams[unigram] != 1:
                b = (self.bigrams[bi_] - 1) / (self.unigrams[unigram] - 1)
            else:
                b = 0

            c = (self.unigrams[uni_] - 1) / (self.unigrams_count - 1)

            max_value = max(a, b, c)
            max_index = [a, b, c].index(max_value)

            lambdas[max_index] += self.trigrams[trigram]

         
        lambdas = [l / self.unigrams_count for l in lambdas]

        return lambdas
    
    def interpolated(self):
        lambdas = self.get_lambdas()

        for trigram in self.trigrams:
            bigram = tuple(trigram[:2])
            bi_ = tuple(trigram[-2:])
            unigram = trigram[-2]
            uni_ = trigram[-1]

            self.t_probabilities_i[trigram] = lambdas[0] * self.trigrams[trigram]/self.bigrams[bigram] + lambdas[1] * self.bigrams[bi_]/self.unigrams[unigram] + lambdas[2] * self.unigrams[uni_]/self.unigrams_count

        for bigram in self.bigrams:
            unigram = bigram[0]
            uni_ = bigram[1]

            self.b_probabilities_i[bigram] = lambdas[1] * self.bigrams[bigram]/self.unigrams[unigram] + lambdas[2] * self.unigrams[uni_]/self.unigrams_count

        for unigram in self.unigrams:
            self.u_probabilities_i[unigram] = lambdas[2] * self.unigrams[unigram]/self.unigrams_count

    def get_perplexity(self,sent,method):
        if method == 'gt':
            sum = 0
            for i in range(len(sent)-2):
                trigram = tuple(sent[i:i+3])
                if trigram in self.probabilities_gt:
                    sum += np.log2(self.probabilities_gt[trigram])
                else:
                    num = self.rstar[0]
                    den = 0
                    for unigram in self.unigrams:
                        x = tuple(trigram[:2]+ (unigram,))
                        if x in self.trigrams:
                            den += self.rstar[self.trigrams[x]]
                        else:
                            den += self.rstar[0]
                    sum += np.log2(num/den)
            return np.exp(-sum/(len(sent)-2))
        
        elif method == 'i':
            sum = 0
            for i in range(len(sent)-2):
                trigram = tuple(sent[i:i+3])
                bigram = tuple(trigram[1:])
                unigram = trigram[2]
                if trigram in self.t_probabilities_i:
                    sum += np.log2(self.t_probabilities_i[trigram])
                elif bigram in self.b_probabilities_i:
                    sum += np.log2(self.b_probabilities_i[bigram])
                elif unigram in self.u_probabilities_i:
                    sum += np.log2(self.u_probabilities_i[unigram])
            return np.exp(-sum/(len(sent)-2))
        
        else:
            sum = 0
            for i in range(len(sent)-2):
                trigram = tuple(sent[i:i+3])
                if trigram in self.t_probabilities:
                    sum += np.log2(self.t_probabilities[trigram])
                else:
                    sum += np.log(10**-15)
            return np.exp(-sum/(len(sent)-2))
                 

    def evaluation(self,method,sent):
        sent = sent.replace('\n',' ')
        sent = sent.lower()
        sent = tokenize(sent)

        for i in range(len(sent)):
            sent[i] = [token for token in sent[i] if token.isalnum()]

        if self.n!=1:
            for i in range(len(sent)):
                sent[i] = ['<s>']*(self.n-1) + sent[i] + ['</s>']
        
        for s in sent:
            for token in s:
                if token not in self.unigrams:
                    token = '<UNK>'

        perplexity = self.get_perplexity(sent[0],method)
        return perplexity
    
    def test_eval(self,method,path):
        self.testing_data = [
            ['<unk>' if word not in self.unigrams else word for word in sentence]
            for sentence in self.testing_data
        ]

        perplexities = [self.get_perplexity(sentence, method) for sentence in self.testing_data]

        avg_perplexity = np.mean(perplexities)
        # print(avg_perplexity)

        with open(path, 'w') as f:
            f.write(f'avg perplexity: {avg_perplexity}\n')
            for sentence, perplexity in zip(self.testing_data, perplexities):
                f.write(f"{' '.join(sentence)}\t{perplexity}\n")

        return perplexities
    
    def train_eval(self,method,path):
        # Calculate perplexities for each sentence in the training corpus
        perplexities = [self.get_perplexity(sentence, method) for sentence in self.training_data]

        # Print and write results to a file
        avg_perplexity = np.mean(perplexities)
        with open(path, 'w') as f:
            f.write(f'avg perplexity: {avg_perplexity}\n')
            for sentence, perplexity in zip(self.training_data, perplexities):
                f.write(f"{' '.join(sentence)}\t{perplexity}\n")

        return perplexities
    
    def train(self):
        self.read_file()
        self.setup()
        self.add_unk()
        self.create_n_grams(self.training_data)
        self.create_probabilities()
        self.good_turing()
        self.interpolated()
    
    # def test(self,method):
    #     self.test_eval(method,'test_eval.txt')

    def n_grams(self,n):
        if n!=1 and n!=2 and n!=3:
            count = 0
            ngrams= {}
            for sent in self.training_data:
                for i in range(len(sent)-n+1):
                    ngram = tuple(sent[i:i+n])
                    if ngram in ngrams:
                        ngrams[ngram] += 1
                    else:
                        ngrams[ngram] = 1
                    count += 1
            return ngrams,count
       
    
    def generate_next_token(self,prev_tokens,method,n,k):
        if n==1:
            n_grams,count = self.unigrams,self.unigrams_count
        elif n==2:
            n_grams,count = self.bigrams,self.bigrams_count
        elif n==3:
            n_grams,count = self.trigrams,self.trigrams_count
        else:
            n_grams,count = self.n_grams(n)

        prev_tokens = prev_tokens.replace('\n',' ')
        prev_tokens = prev_tokens.lower()
        prev_tokens = tokenize(prev_tokens)

        for i in range(len(prev_tokens)):
            prev_tokens[i] = [token for token in prev_tokens[i] if token.isalnum()]

        if self.n!=1:
            for i in range(len(prev_tokens)):
                if i != len(prev_tokens)-1:
                    prev_tokens[i] = ['<s>']*(self.n-1) + prev_tokens[i] + ['</s>']
                else:
                    prev_tokens[i] = ['<s>']*(self.n-1) + prev_tokens[i]

        for s in prev_tokens:
            for token in s:
                if token not in self.unigrams:
                    token = '<UNK>'

        if n==1:
            sent = ['<s>']

        sent = prev_tokens[-1][-2:] if method == 'i' else prev_tokens[-1][-n+1:]

        probs = {}
        for unigram in self.unigrams:
            if unigram not in ('<s>', '</s>'):
                x = tuple(list(sent) + [unigram])
                if method == 'i':
                    y = tuple(list(sent[-1:]) + [unigram])
                    if x in self.trigrams:
                        probs[unigram] = self.t_probabilities_i[x]
                    elif y in self.bigrams:
                        probs[unigram] = self.b_probabilities_i[y]
                    else:
                        probs[unigram] = self.u_probabilities_i[unigram]
                else:
                    if x in n_grams:
                        probs[unigram] = n_grams[x] / count

        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        return sorted_probs[:k]
    
    def save(self, path):
        with open(path,'wb') as f:
            pickle.dump(self,f)

    def load(self,path):
        with open(path,'rb') as f:
            model = pickle.load(f)
        self.n = model.n
        self.corpus = model.corpus
        self.unigrams = model.unigrams
        self.bigrams = model.bigrams
        self.trigrams = model.trigrams
        self.unigrams_count = model.unigrams_count
        self.bigrams_count = model.bigrams_count
        self.trigrams_count = model.trigrams_count
        self.u_probabilities = model.u_probabilities
        self.b_probabilities = model.b_probabilities
        self.t_probabilities = model.t_probabilities
        self.probabilities_gt = model.probabilities_gt
        self.u_probabilities_i = model.u_probabilities_i
        self.b_probabilities_i = model.b_probabilities_i
        self.t_probabilities_i = model.t_probabilities_i
        self.training_data = model.training_data
        self.testing_data = model.testing_data
        self.a = model.a
        self.b = model.b
        self._Nr = model._Nr
        self._Zr = model._Zr
        self.smoothed_Nr = model.smoothed_Nr
        self.rstar = model.rstar
        self.rlast = model.rlast
    

if __name__ == '__main__':
    ngram = N_Gram_Model(3,'Data/Pride and Prejudice - Jane Austen.txt')
    ngram.train()
    ngram.save('pride.pkl')

    n_gram1 = N_Gram_Model(3,'Data/Ulysses  James Joyce.txt')
    n_gram1.train()
    n_gram1.save('ulysses.pkl')
     
        


    






            




         

    