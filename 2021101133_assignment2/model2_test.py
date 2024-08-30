#testing model2 :LSTM
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from conllu import parse_incr
from classes import Create_Dataset, LSTM
from torch.utils.data import DataLoader
from model1_test import get_values, calculate_metrics
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab = np.load("vocab2.npy", allow_pickle=True).item()
upos_tags = np.load("upos_tags2.npy", allow_pickle=True).item()
model = torch.load("model2.pth")

file = open("UD_English-Atis/en_atis-ud-test.conllu", "r", encoding="utf-8")
sentences = list(parse_incr(file))
file.close()
test_data = Create_Dataset(sentences,p=2,s=2,vocab=vocab,upos_tags=upos_tags, N=1)  
test_loader = DataLoader(test_data, batch_size=32, shuffle=True)

file = open("UD_English-Atis/en_atis-ud-dev.conllu", "r", encoding="utf-8")
sentences = list(parse_incr(file))
file.close()
validation_data = Create_Dataset(sentences,p=2,s=2,vocab=vocab,upos_tags=upos_tags, N=1)
validation_loader = DataLoader(validation_data, batch_size=32, shuffle=True)

model.eval()
model.to(device)

def test_LSTM(data):
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
            preds = model(inputs)
            preds_ = preds.view(-1, preds.shape[-1])
            targets_ = targets.view(-1, targets.shape[-1])
            loss = criterion(preds_, targets_)

            avg_loss += loss.item()
            _, val1 = torch.max(targets, 2)
            _, val2 = torch.max(preds, 2)

            correct += (val1 == val2).sum().item()
            total += targets.size(0) * targets.size(1)

            targets = targets.view(-1)
            preds = preds.view(-1)

            true_p, false_p, false_n, confusion_matrix = get_values(targets,preds,true_p, false_p, false_n, confusion_matrix)

        accuracy = correct / total
        precision_macro, recall_macro, f1_macro, precision_micro, recall_micro, f1_micro = calculate_metrics(true_p, false_p, false_n)
        return avg_loss / len(test_loader), accuracy, precision_macro, recall_macro, f1_macro, precision_micro, recall_micro, f1_micro, confusion_matrix

avg_loss, accuracy, precision_macro, recall_macro, f1_macro, precision_micro, recall_micro, f1_micro, confusion_matrix = test_LSTM(test_loader)
val_loss, accuracy_val, precision_macro_val, recall_macro_val, f1_macro_val, precision_micro_val, recall_micro_val, f1_micro_val, confusion_matrix_val = test_LSTM(validation_loader)

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



