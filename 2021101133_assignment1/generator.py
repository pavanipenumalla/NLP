from LM import N_Gram_Model
import sys

lm_type = sys.argv[1]
corpus_path = sys.argv[2]
k = int(sys.argv[3])
ngram = N_Gram_Model(3, corpus_path)
if corpus_path == 'INLP assignment 1 corpus/Pride and Prejudice - Jane Austen.txt':
    ngram.load('pride.pkl')
else:
    ngram.load('ulysses.pkl')

input_text = input("Enter the sentence: ")
n  = int(input("Enter n:"))

print(ngram.generate_next_token(input_text,lm_type,n,k))
