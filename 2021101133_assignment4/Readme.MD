## INLP ASSIGNMENT - 4


- Data used : [link](https://iiitaphyd-my.sharepoint.com/personal/advaith_malladi_research_iiit_ac_in/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fadvaith%5Fmalladi%5Fresearch%5Fiiit%5Fac%5Fin%2FDocuments%2FiNLP%5FA2%2FiNLP%2DA2%2Ezip&parent=%2Fpersonal%2Fadvaith%5Fmalladi%5Fresearch%5Fiiit%5Fac%5Fin%2FDocuments%2FiNLP%5FA2&ga=1)

### Training ELMO Model

- utils.py : Contains the utility functions for training the elmo model.
- ELMO.py : Contains the hyperparameters for training the elmo model. Hyperparameters can be adjusted as required.

Run the follwoing command to train ELMO model:
```
python3 ELMO.py
```
Model will be saved in the folder.

### Classification Task

- classification_utils.py :
    - Contains the utility functions for classification task.
    - Model architecture, training function and evaluation functions are present in this file.
- classification.py :
    - Contains the hyperparameters for training the classification model.
    - Different methods of hypereparameter tuning can be given as arguments while running the file.
    - Run the following command to train the classification model using trainable lambdas:
    ```
    python3 classification.py 1
    ```
    - Run the following command to train the classification model using frozen lambdas:
    ```
    python3 classification.py 2
    ```
    - Run the following command to train the classification model using learnable function:
    ```
    python3 classification.py 3
    ```
    - Respectively, the models will be saved in the folder.

### Models 
 
 All the classification models and the pretrained ELMO can be downloaded from this [link.](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/aditya_pavani_students_iiit_ac_in/Es2NDTpBiwJPsXJtA_V6mVQBav-CfFjenK35Fwlh4Gj9hw?e=AKn6t0)

 

