from LM import N_Gram_Model
import sys

lm_type = sys.argv[1]
corpus_path = sys.argv[2]
ngram = N_Gram_Model(3, corpus_path)
if corpus_path == 'Data/Pride and Prejudice - Jane Austen.txt':
    ngram.load('pride.pkl')
else:
    ngram.load('ulysses.pkl')


input_text = input("Enter the sentence: ")
print(ngram.get_perplexity(input_text, lm_type))