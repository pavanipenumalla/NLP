import re

def tokenize(text):
    sentence_pattern = re.compile(r'(?<!\w{1}\.\w{1}\.)(?<![A-Z][a-z]\.)(?<![A-Z]\.)(?<!Mrs\.)(?<=\.\"|\?\"|\!\")\s|(?<=\.|\?|\!)\s')
    number_pattern = re.compile(r'\d+')
    email_pattern = re.compile(r"[^@]+@[^@]+\.[^@]+")
    url_pattern = re.compile(r"https?:\/\/[a-zA-Z0-9./]+")
    mention_pattern = re.compile(r"@[a-zA-Z0-9]+")
    hashtag_pattern = re.compile(r"#[a-zA-Z0-9]+")
    punct_pattern = re.compile(r"<NUM>|<MAILID>|<URL>|<MENTION>|<HASHTAG>|Mr\.|Mrs\.|Dr\.|e.g.|\w+-\w+|[A-Z].\s|[^\w\s]|\w+")

    sentences = sentence_pattern.split(text)
    # print(sentences)

    words = []
    for sentence in sentences:
        words.append(re.findall(r"\S*\w+\S*", sentence))
    # print(words)

    for i in range(len(words)):
        for j in range(len(words[i])):
            if number_pattern.match(words[i][j]):
                words[i][j] = "<NUM>"
            if email_pattern.match(words[i][j]):
                words[i][j] = "<MAILID>"
            if url_pattern.match(words[i][j]):
                words[i][j] = "<URL>"
            if mention_pattern.match(words[i][j]):
                words[i][j] = "<MENTION>"
            if hashtag_pattern.match(words[i][j]):
                words[i][j] = "<HASHTAG>"

    tokens = []
    for i in range(len(words)):
        t = []
        for j in range(len(words[i])):
            t +=  punct_pattern.findall(words[i][j])
        tokens.append(t)
    # print(tokens) 
        
    return tokens

if __name__ == "__main__":
    text = input("Enter text: ")
    print(tokenize(text))