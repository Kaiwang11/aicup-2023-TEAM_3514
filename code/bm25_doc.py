import json
import numpy as np
import pandas as pd
from tqdm import tqdm

from rank_bm25 import BM25Okapi
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
import jieba

from utils import load_json, jsonl_dir_to_df, generate_evidence_to_wiki_pages_mapping
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True, verbose=0, nb_workers=10)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='train', type=str)
args = parser.parse_args()

# prepare wiki data
wiki_pages = jsonl_dir_to_df("data/wiki-pages")
wiki_pages = wiki_pages.loc[wiki_pages["text"]!="",["id", "text"]]
titles = wiki_pages["id"]
corpus = list(wiki_pages["text"])
del wiki_pages


# prepare preprocess function
def preprocess_documents(documents):
    stop_words = set(stopwords.words('chinese'))
    processed_docs = []
    for doc in tqdm(documents):
        # Tokenize and remove stopwords
        tokens = jieba.lcut(doc)
        processed_docs.append([token for token in tokens if token.isalnum() and token not in stop_words])
    return processed_docs

def preprocess_query(query):
    stop_words = set(stopwords.words('chinese'))
    tokens = jieba.lcut(query)
    return [token for token in tokens if token.isalnum() and token not in stop_words]


print("##############################")
print("Start tokenizing wiki docs")
print("##############################")
tokenized_corpus = preprocess_documents(corpus)
del corpus
bm25 = BM25Okapi(tokenized_corpus)


# train mode
if args.mode == "train":
    TRAIN_DATA = load_json("data/concat_train.jsonl")
    index = list(x["id"] for x in TRAIN_DATA)
    claims = list(x["claim"] for x in TRAIN_DATA)
    labels = list(x["label"] for x in TRAIN_DATA)
    evidences = list(x["evidence"] for x in TRAIN_DATA)
    del TRAIN_DATA

    print("##############################")
    print("Start retriveing doc with BM25")
    print("##############################")
    recall = 0
    predicted_pages = []
    for i in tqdm(range(len(claims))):

        # get top10 bm25 docs
        query = preprocess_query(claims[i])
        top10 = set(titles.iloc[np.argpartition(bm25.get_scores(query), -10)[-10:]])
        predicted_pages.append(list(top10))

        if labels[i] == "NOT ENOUGH INFO":
            recall += 1
            continue

        # compute recall
        answers = []
        for j in range(len(evidences[i][0])):
            answers.append(evidences[i][0][j][2])

        hits = len(top10.intersection(set(answers)))
        counts = len(set(answers))
        if hits == counts:
            recall += 1


    print("Recall = ",recall/len(index))
    pd.DataFrame({"id": index, "predicted_pages": predicted_pages}).to_csv("exp/train_bm25_doc.csv", index=False)



# test mode
if args.mode == "test":
    TEST_DATA = load_json("data/concat_test.jsonl")
    index = list(x["id"] for x in TEST_DATA)
    claims = list(x["claim"] for x in TEST_DATA)
    del TEST_DATA

    print("##############################")
    print("Start retriveing doc with BM25")
    print("##############################")
    predicted_pages = []
    for i in tqdm(range(len(claims))):

        # get top10 bm25 docs
        query = preprocess_query(claims[i])
        top10 = set(titles.iloc[np.argpartition(bm25.get_scores(query), -10)[-10:]])
        predicted_pages.append(list(top10))

    pd.DataFrame({"id": index, "predicted_pages": predicted_pages}).to_csv("exp/test_bm25_doc.csv", index=False)
