# INLP Assignment 2

## Files
### Classes.py:
- This file contains the code for the classes used in the assignment.
- The classes are:
    - create_Dataset - This class is used to create the dataset for the models. The argument 'N' is used to specify which neural network model to use. If N=1, the dataset for the Feed Forward Neural Network is created. If N=2, the dataset for the LSTM model is created.
    - FFNN - This class is used to create the Feed Forward Neural Network model.
    - LSTM - This class is used to create the LSTM model.

### model_1_train.py:
- This file contains the code for training the **Feed Forward Neural Network**.
- The model is trained on the training data and the trained model is saved in the file model1.pth.
- Vocabularies are also saved in the file vocab1.npy, upos_tags1.npy.

Run the file using the command: 
```bash
python3 model_1_train.py
```

### model_1_tuning.py:
- This file contains the code for tuning the hyperparameters of the **Feed Forward Neural Network**.
- The hyperparameters used for tuning:
    - Activation function - ReLU, Tanh
    - Learning rate - 0.1, 0.01, 0.001
    - Number of hidden layers - 1, 2, 3
    - Hidden Dimension - 64, 128, 256
    - Embedding Dimension - 64, 128, 256
    - Epochs - 5, 10, 15

Run the file using the command:
```bash
python3 model_1_tuning.py
```

### model_1_test.py:
- This file contains the code for testing the **Feed Forward Neural Network**.
- Loads the trained model from the file model1.pth and tests it on the test data.
- Metrics used for evaluation are:
    - Accuracy
    - Precision (Macro and Micro)
    - Recall (Macro and Micro)
    - F1 Score (Macro and Micro)
    - Confusion Matrix

Run the file using the command:
```bash
python3 model_1_test.py
```

### model_2_train.py:
- This file contains the code for training the **LSTM** model.
- The model is trained on the training data and the trained model is saved in the file model2.pth.
- Vocabularies are also saved in the file vocab2.npy, upos_tags2.npy.
- This file also contains the code for plotting the graph of epochs vs accuracy and epochs vs loss for **LSTM** model on dev data.

Run the file using the command:
```bash
python3 model_2_train.py
```

### model_2_tuning.py:
- This file contains the code for tuning the hyperparameters of the **LSTM** model.
- The hyperparameters used for tuning:
    - Activation function - ReLU, Tanh
    - Learning rate - 0.1, 0.01, 0.001
    - Number of LSTM layers - 1, 2, 3
    - Hidden Dimension - 64, 128, 256
    - Embedding Dimension - 64, 128, 256
    - Epochs - 5, 10, 15
    - Bidirectional - True, False

Run the file using the command:
```bash
python3 model_2_tuning.py
```

### model_2_test.py:
- This file contains the code for testing the **LSTM** model.
- Loads the trained model from the file model2.pth and tests it on the test data.
- Metrics used for evaluation are:
    - Accuracy
    - Precision (Macro and Micro)
    - Recall (Macro and Micro)
    - F1 Score (Macro and Micro)
    - Confusion Matrix

Run the file using the command:
```bash
python3 model_2_test.py
```

### graphs.py:
- This file contains the code for plotting the graph of p vs loss and p vs accuracy for **FFNN** model on dev data.

Run the file using the command:
```bash
python3 graphs.py
```

### pos_tagger.py:
To run the file, use the following command:
```bash
python3 pos_tagger.py -method <method>
```
- method: method to be used for prediction. It can be -f for FFFN or -r for LSTM.
- On running the file, it prompts the user to enter the sentence and then predicts the POS tags for the sentence.


