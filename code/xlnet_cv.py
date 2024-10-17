import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
import jieba
import re
import ast
from tqdm import tqdm

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='train', type=str)
args = parser.parse_args()



def preprocess_documents(documents):
    stop_words = set(stopwords.words('chinese'))
    processed_docs = []
    for doc in documents:
        # Tokenize and remove stopwords
        tokens = jieba.lcut(doc)
        processed_docs.append([token for token in tokens if token.isalnum() and token not in stop_words])
    return processed_docs

def preprocess_query(query):
    stop_words = set(stopwords.words('chinese'))
    tokens = jieba.lcut(query)
    return [token for token in tokens if token.isalnum() and token not in stop_words]


class TrainDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # get item
        item = self.df.iloc[idx]
        claim = item['claim']
        title = item['evidence']
        sent = item['evidence_list']

        # concat claim & title & sent
        try:
            text = claim + " [SEP] " + title[0][0][2] + " [SEP] " + sent[0]
        except:
            text = claim + " [SEP] " + sent[0]
        # encode text
        encoding = self.tokenizer(text, padding="max_length", max_length=128, truncation=True, return_tensors="pt")
        # remove batch dimension which the tokenizer automatically adds
        encoding = {k:v.squeeze() for k,v in encoding.items()}
        # add label
        label = item['label']
        encoding["labels"] = torch.tensor(LABEL2ID[label])

        return encoding


class TestDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # get item
        item = self.df.iloc[idx]
        claim = item['claim']
        title = item['predicted_evidence']
        # sent = item['evidence_list']
        sent = item['preprocessed_evidence_list']

        # concat claim & title & sent
        try:
            titles = ""
            for t in title:
                titles += t
            # text = claim + " [SEP] " + titles + " [SEP] " + sent[0]
            text = claim + " [SEP] " + titles + " [SEP] " + str(sent)
        except:
            # text = claim + " [SEP] " + sent[0]
            text = claim + " [SEP] " + "這個句子沒有東西欸"

        # encode text
        encoding = self.tokenizer(text, padding="max_length", max_length=128, truncation=True, return_tensors="pt")
        # remove batch dimension which the tokenizer automatically adds
        encoding = {k:v.squeeze() for k,v in encoding.items()}

        return encoding


MODEL_NAME = "hfl/chinese-xlnet-base"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3, ignore_mismatched_sizes=True)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001)

LABEL2ID = {
    "supports": 0,
    "refutes": 1,
    "NOT ENOUGH INFO": 2,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


if args.mode == "train":
    train_df = pd.read_csv("exp/train_data.csv")
    train_dataset = TrainDataset(df=train_df, tokenizer=tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # for batch in train_dataloader:
    #     print(batch)
    #     break


    n_epochs = 3
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        for batch in train_dataloader:
            # put batch on device
            batch = {k:v.to(device) for k,v in batch.items()}

            # forward pass
            outputs = model(**batch)
            loss = outputs.loss
            train_loss += loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f"Loss after epoch {epoch+1}:", train_loss)

    torch.save(model.state_dict(), 'checkpoint/claim_verify/xlnet.ckpt')


if args.mode == 'test':
    model.load_state_dict(torch.load('checkpoint/claim_verify/xlnet.ckpt'))
    # predict
    test_df = pd.read_csv("exp/test_data.csv")
    test_df['predicted_evidence'] = test_df['predicted_evidence'].apply(ast.literal_eval)

    # clean evidence list
    corpus = test_df["evidence_list"].tolist()
    # Sentence splitting pattern
    split_pattern = r'[，。]'

    # Splitting documents into sentences
    corpus_sentences = []
    for doc in corpus:
        sentences = re.split(split_pattern, doc)
        corpus_sentences.append(sentences)

    # BM25 ranking
    queries = test_df["claim"].tolist()
    new_corpus = []
    for i, query in enumerate(tqdm(queries)):
        tokenized_corpus = preprocess_documents(corpus_sentences[i])
        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(preprocess_query(query))
        if len(scores[scores>1]) > 1:
            indices = np.where(scores > 1)
            sentences = [tokenized_corpus[i] for i in indices[0]]
            # sentences = list(np.array(tokenized_corpus)[scores>1])
        else:
            indices = np.where(scores > 0)
            sentences = [tokenized_corpus[i] for i in indices[0]]
            # sentences = list(np.array(tokenized_corpus)[scores>0])
        # print(sentences)
        nc = "，".join(["".join(sen) for sen in sentences])
        # print(nc)
        new_corpus.append(nc)

    test_df['preprocessed_evidence_list'] = new_corpus
    # test_df['evidence_list'] = test_df['evidence_list'].apply(ast.literal_eval)
    test_dataset = TestDataset(df=test_df, tokenizer=tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=32)

    model.eval()
    preds = []
    for batch in test_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        output = model(**batch)
        pred = torch.argmax(output.logits, dim=1)
        preds.extend(pred.tolist())


    test_df["predicted_label"] = list(map(ID2LABEL.get, preds))
    print(test_df["predicted_label"].value_counts())
    test_df[["id", "predicted_label", "predicted_evidence"]].to_json(
        "sub/0602sub_xlnet.jsonl",
        orient="records",
        lines=True,
        force_ascii=False,
    )
