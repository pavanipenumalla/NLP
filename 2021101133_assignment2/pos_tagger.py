import torch
import numpy as np
import sys

method = ""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
p = 1
s = 1
START_TOKEN = "<s>"
END_TOKEN =  "</s>"


def get_pos_tags_ffnn(sent):
    model = torch.load("model1.pth").to(device)
    vocab = np.load("vocab1.npy", allow_pickle=True).item()
    upos_vocab = np.load("upos_tags1.npy", allow_pickle=True).item()
    model.eval()

    st = sent.split()
    sent = sent.lower().split()
    sent = [START_TOKEN] * p + sent + [END_TOKEN] * s

    tokens = []
    for i in range(p, len(sent) - s):
        tokens.append([sent[j] for j in range(i - p, i + s + 1)])

    tokens = [[vocab[token] if token in vocab else vocab["<unk>"] for token in x] for x in tokens]
    tokens = torch.tensor(tokens).to(device)
    # print(tokens.shape)
   
    outputs = model(tokens)
    preds = torch.argmax(outputs, 1)
    preds = preds.numpy()
    for i in range(len(st)):
        print("{} {}".format(st[i], list(upos_vocab.keys())[list(upos_vocab.values()).index(preds[i])]))

def get_pos_tags_lstm(sent):
    model = torch.load("model2.pth").to(device)
    vocab = np.load("vocab2.npy", allow_pickle=True).item()
    upos_vocab = np.load("upos_tags2.npy", allow_pickle=True).item()
    model.eval()

    st = sent.split()
    sent = sent.lower().split()
    sent = [START_TOKEN] + sent + [END_TOKEN]

    tokens = [vocab[token] if token in vocab else vocab["<unk>"] for token in sent]
    tokens = torch.tensor(tokens).to(device)
    tokens = tokens.view(1, -1)

    outputs = model(tokens).to("cpu")
    preds = torch.argmax(outputs, 2)
    preds = preds.numpy()
    preds = preds[0]
    for i in range(len(st)):
        print("{} {}".format(st[i], list(upos_vocab.keys())[list(upos_vocab.values()).index(preds[i+1])]))

def main():
    if sys.argv[1] == "-f":
        method = "FFN"
    elif sys.argv[1] == "-r":
        method = "RNN"
    else:
        print("Invalid method")
        sys.exit()
    sentence = input("Enter a sentence: ")
    if method == "FFN":
        get_pos_tags_ffnn(sentence)
    elif method == "RNN":
        get_pos_tags_lstm(sentence)

if __name__ == "__main__":
    main()


         