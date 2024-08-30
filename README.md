
# INLP Assignments

## Introduction
This repository contains solutions for various assignments in Natural Language Processing (NLP) focusing on tokenization, language models, neural networks, word embeddings, and ELMo-based classification.

## Assignment 1: Tokenization and Language Models

- **`tokenizer.py`**: Tokenizes input text.  
  ```bash
  python3 tokenizer.py
  ```

- **`language_model.py`**: Calculates perplexity for a given sentence.  
  ```bash
  python3 language_model.py <lm_type> <corpus_path>
  ```

- **`generator.py`**: Generates next word predictions.  
  ```bash
  python3 generator.py <lm_type> <corpus_path> <k>
  ```

- **Models and Perplexity Scores**: Pretrained models and perplexity evaluation files.

## Assignment 2: Feed Forward Neural Networks (FFNN) and LSTM

- **`Classes.py`**: Defines dataset creation and model classes (FFNN and LSTM).

- **`model_1_train.py`**: Trains and saves the FFNN model.  
  ```bash
  python3 model_1_train.py
  ```

- **`model_1_tuning.py`**: Tunes hyperparameters for the FFNN model.  
  ```bash
  python3 model_1_tuning.py
  ```

- **`model_1_test.py`**: Tests the FFNN model and evaluates metrics.  
  ```bash
  python3 model_1_test.py
  ```

- **`model_2_train.py`**: Trains the LSTM model, saves it, and plots performance.  
  ```bash
  python3 model_2_train.py
  ```

- **`model_2_tuning.py`**: Tunes hyperparameters for the LSTM model.  
  ```bash
  python3 model_2_tuning.py
  ```

- **`model_2_test.py`**: Tests the LSTM model and evaluates metrics.  
  ```bash
  python3 model_2_test.py
  ```

- **`graphs.py`**: Plots FFNN model performance graphs.  
  ```bash
  python3 graphs.py
  ```

- **`pos_tagger.py`**: Predicts POS tags using FFNN or LSTM.  
  ```bash
  python3 pos_tagger.py -method <method>
  ```

## Assignment 3: Word Embeddings and Classification

- **SVD Embeddings**: Generates embeddings using SVD.  
  ```bash
  python3 svd.py
  ```

- **Word2Vec (SGNS) Embeddings**: Generates embeddings using Skip-gram.  
  ```bash
  python3 skip-gram.py
  ```

- **Classification Models**: Classifies using pretrained SVD or Word2Vec embeddings.  
  ```bash
  python3 svd-classification.py
  python3 skip-gram-classification.py
  ```

## Assignment 4: ELMo Model and Classification

- **Training ELMo**: Trains an ELMo model.  
  ```bash
  python3 ELMO.py
  ```

- **Classification**: Trains classification models with various hyperparameter settings.  
  ```bash
  python3 classification.py 1  # Trainable Lambdas
  python3 classification.py 2  # Frozen Lambdas
  python3 classification.py 3  # Learnable Function
  ```

- **Models**: Download pretrained ELMo and classification models.

 
