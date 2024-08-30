from LM import N_Gram_Model

ngram = N_Gram_Model(3,'Data/Pride and Prejudice - Jane Austen.txt')
ngram.load('pride.pkl')
ngram.test_eval('gt', '2021101133_LM1_test-perplexity.txt')
ngram.test_eval('i', '2021101133_LM2_test-perplexity.txt')
ngram.train_eval('gt', '2021101133_LM1_train-perplexity.txt')
ngram.train_eval('i', '2021101133_LM2_train-perplexity.txt')

n_gram1 = N_Gram_Model(3,'Data/Ulysses  James Joyce.txt')
n_gram1.load('ulysses.pkl')
n_gram1.test_eval('gt', '2021101133_LM3_test-perplexity.txt')
n_gram1.test_eval('i', '2021101133_LM4_test-perplexity.txt')
n_gram1.train_eval('gt', '2021101133_LM3_train-perplexity.txt')
n_gram1.train_eval('i', '2021101133_LM4_train-perplexity.txt')