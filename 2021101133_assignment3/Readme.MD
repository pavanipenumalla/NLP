# INLP Assignment 3

## Word Embeddings Generation

Data used: [link](https://iiitaphyd-my.sharepoint.com/personal/advaith_malladi_research_iiit_ac_in/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fadvaith%5Fmalladi%5Fresearch%5Fiiit%5Fac%5Fin%2FDocuments%2FiNLP%5FA2%2FiNLP%2DA2%2Ezip&parent=%2Fpersonal%2Fadvaith%5Fmalladi%5Fresearch%5Fiiit%5Fac%5Fin%2FDocuments%2FiNLP%5FA2&ga=1).

Generate word embeddings using the following methods:
1. SVD
2. Skip-gram (SGNS)

### SVD Embeddings

1. Modify the hyperparameters in the `svd.py` file.
2. Run the following command in your terminal:
    ```
    python3 svd.py 
    ```
3. Word embeddings will be saved to your folder.

### Word2Vec (SGNS) Embeddings
1. Modify the hyperparameters in the `skip-gram.py` file.
2. Run the following command in your terminal:
    ```
    python3 skip-gram.py 
    ```
3. Word embeddings will be saved to your folder.


## Classification Model

Pretrained models and embeddings can be downloaded from this [link.](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/aditya_pavani_students_iiit_ac_in/EqvazgPLJYpMjyrEl2f3USQBY2GNIP9GI7etddITH2g-7Q?e=s8HHbz)

### Using SVD Embeddings
1. Download the pretrained embeddings `svd-word-vectors.pt` and `svd-word2idx.pt` from the provided link.
2. Run the following command in your terminal:
    ```
    python3 svd-classification.py
    ```

### Using Word2Vec Embeddings
1. Download the pretrained model `skip-gram-word-vectors.pt` and `skipgram-word2idx.pt` from the provided link.
2. Run the following command in your terminal:
    ```
    python3 skip-gram-classification.py
    ```

## Loading the Models

1. Download the best performing models for the downstream task from the provided link. They are saved as `skip-gram-classification-model.pt` and `svd-classification-model.pt`.

 

