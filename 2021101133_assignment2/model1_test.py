# testing the model1
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from conllu import parse_incr
from classes import Create_Dataset, FFNN
from torch.utils.data import DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
p = 1
s = 1
vocab = np.load("vocab1.npy", allow_pickle=True).item()
upos_tags = np.load("upos_tags1.npy", allow_pickle=True).item()
model = torch.load("model1.pth")

file = open("UD_English-Atis/en_atis-ud-test.conllu", "r", encoding="utf-8")
sentences = list(parse_incr(file))
file.close()
test_data = Create_Dataset(sent= sentences,p=p,s=s,vocab=vocab,upos_tags=upos_tags,N=0)
test_loader = DataLoader(test_data, batch_size=32, shuffle=True)

file = open("UD_English-Atis/en_atis-ud-dev.conllu", "r", encoding="utf-8")
sentences = list(parse_incr(file))
file.close()
validation_data = Create_Dataset(sent= sentences,p=p,s=s,vocab=vocab,upos_tags=upos_tags,N=0)
validation_loader = DataLoader(validation_data, batch_size=32, shuffle=True)

model.eval()
model.to(device)

def get_values(targets,preds ,true_p, false_p, false_n, confusion_matrix):
    
    for i in range(len(upos_tags)):
        for j in range(len(targets)):
            if targets[j] == i:
                if preds[j] == i:
                    true_p[i] += 1
                else:
                    false_n[i] += 1
            else:
                if preds[j] == i:
                    false_p[i] += 1
    for i in range(len(targets)):
        confusion_matrix[targets[i]][preds[i]] += 1

    return true_p, false_p, false_n, confusion_matrix

def calculate_metrics(true_p, false_p, false_n):

    precision = np.where((true_p + false_p) == 0, 0, true_p / (true_p + false_p))
    recall = np.where((true_p + false_n) == 0, 0, true_p / (true_p + false_n))
    f1 = np.where((precision + recall) == 0, 0, 2 * (precision * recall) / (precision + recall))

    precision = precision[precision != 0]
    recall = recall[recall != 0]
    f1 = f1[f1 != 0]
    
    precision_macro = np.mean(precision)
    recall_macro = np.mean(recall)
    f1_macro = np.mean(f1)
    
    precision_micro = np.sum(true_p) / (np.sum(true_p) + np.sum(false_p))
    recall_micro = np.sum(true_p) / (np.sum(true_p) + np.sum(false_n))
    f1_micro = 2 * (precision_micro * recall_micro) / (precision_micro + recall_micro)

    return precision_macro, recall_macro, f1_macro, precision_micro, recall_micro, f1_micro

def test_FFNN(data):
    correct = 0
    total = 0
    avg_loss = 0

    true_p, false_p , false_n = np.zeros(len(upos_tags)) , np.zeros(len(upos_tags)), np.zeros(len(upos_tags))
    confusion_matrix = np.zeros((len(upos_tags), len(upos_tags)))

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data):
            inputs = inputs.to(device) 
            targets = targets.to(device)
            # print(inputs.shape)
            preds = model(inputs)
            loss = criterion(preds, targets)
            
            val1 = torch.argmax(preds, 1)
            val2 = torch.argmax(targets, 1)

            correct += (val1==val2).sum().item()
            total += targets.size(0)
            avg_loss += loss.item()

            targets = torch.argmax(targets, 1)
            preds = torch.argmax(preds, 1)

            true_p, false_p, false_n, confusion_matrix = get_values(targets,preds,true_p, false_p, false_n, confusion_matrix)

        accuracy = correct / total
        precision_macro, recall_macro, f1_macro, precision_micro, recall_micro, f1_micro = calculate_metrics(true_p, false_p, false_n)
        return avg_loss / len(test_loader), accuracy, precision_macro, recall_macro, f1_macro, precision_micro, recall_micro, f1_micro, confusion_matrix

avg_loss, accuracy, precision_macro, recall_macro, f1_macro, precision_micro, recall_micro, f1_micro, confusion_matrix = test_FFNN(test_loader)
val_loss, accuracy_val, precision_macro_val, recall_macro_val, f1_macro_val, precision_micro_val, recall_micro_val, f1_micro_val, confusion_matrix_val = test_FFNN(validation_loader)

print("TEST RESULTS")
print(f"Average loss\t:\t  {avg_loss}")
print(f"Accuracy\t:\t  {accuracy}")
print("Macro")
print(f"Precision\t:\t  {precision_macro}")
print(f"Recall   \t:\t  {recall_macro}")
print(f"F1 score\t:\t  {f1_macro}")
print("Micro")
print(f"Precision\t:\t  {precision_micro}")
print(f"Recall   \t:\t  {recall_micro}")
print(f"F1 score\t:\t  {f1_micro}")
print("---------------------------------------------------")

plt.figure(figsize=(25, 10))
sns.heatmap(confusion_matrix, fmt='f',annot=True, xticklabels=upos_tags.keys(), yticklabels=upos_tags.keys())
plt.show()

print("VALIDATION RESULTS")
print(f"Average loss\t:\t  {val_loss}")
print(f"Accuracy\t:\t  {accuracy_val}")
print("Macro")
print(f"Precision\t:\t  {precision_macro_val}")
print(f"Recall   \t:\t  {recall_macro_val}")
print(f"F1 score\t:\t  {f1_macro_val}")
print("Micro")
print(f"Precision\t:\t  {precision_micro_val}")
print(f"Recall   \t:\t  {recall_micro_val}")
print(f"F1 score\t:\t  {f1_micro_val}")
print("---------------------------------------------------")

plt.figure(figsize=(25, 10))
sns.heatmap(confusion_matrix_val, fmt='f',annot=True, xticklabels=upos_tags.keys(), yticklabels=upos_tags.keys())
plt.show()







 