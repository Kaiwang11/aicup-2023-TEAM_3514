import pandas as pd
from utils import load_json, save_doc
import ast

# get link
import wikipedia
from bs4 import BeautifulSoup as bs
import csv
from tqdm import tqdm
from pandas.core.common import flatten

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='train', type=str)
args = parser.parse_args()


def get_pred(d_ls):
    pred = []
    for item in d_ls:
        pred.append(item.strip())

    # retrieval wiki link from doc_predict top3
    for j in d_ls[:3]:
        try:
            result = wikipedia.search(j.strip())
            soup = bs(str(wikipedia.page(result[0]).html()),'html.parser')
        except:
            continue
        count = 0
        for section in soup.find('div',class_='mw-parser-output').find_all('p',recursive=False):
            for item in section.find_all('a'):
                if 'title' not in item.attrs.keys(): continue
                link = item['title']
                # get 5 wiki_link each doc_pred
                if count < 5 :
                    # check link in wiki_data
                    if link in wiki_page and link not in pred:
                        pred.append(link)
                        count+=1
                else :
                    break

    return([*set(pred)])

def load_csv_list(path):
    with open(path,'r',encoding='utf-8') as f:
        rows = list(csv.reader(f, delimiter="\n"))
    return rows

wikipedia.set_lang('zh')
# read wiki data id (主辦提供的dataset)
wiki_page = list(flatten(load_csv_list('data/wiki_page.csv')))




if args.mode == "train":
    npm_doc10 = pd.read_csv("exp/train_npm_doc.csv")
    npm_doc10['predicted_pages'] = npm_doc10['predicted_pages'].apply(ast.literal_eval)
    bm25_doc10 = pd.read_csv("exp/train_bm25_doc.csv")
    bm25_doc10['predicted_pages'] = bm25_doc10['predicted_pages'].apply(ast.literal_eval)

    TRAIN_DATA = load_json("data/concat_train.jsonl")
    ids = list(x["id"] for x in TRAIN_DATA)
    labels = list(x["label"] for x in TRAIN_DATA)
    evidences = list(x["evidence"] for x in TRAIN_DATA)

    # compute recall before link
    recall = 0
    for i in range(len(labels)):
        pages = npm_doc10.loc[npm_doc10["id"]==ids[i]].iloc[0]["predicted_pages"] + bm25_doc10.loc[bm25_doc10["id"]==ids[i]].iloc[0]["predicted_pages"]

        if labels[i] == "NOT ENOUGH INFO":
            recall += 1
            continue

        answers = []
        for j in range(len(evidences[i][0])):
            answers.append(evidences[i][0][j][2])
        hits = len(set(pages).intersection(set(answers)))
        counts = len(set(answers))
        if hits == counts:
            recall += 1
    print("npm + bm25 recall:", recall/len(ids))


    # compute recall after link
    recall = 0
    predicted_pages = []
    for i in tqdm(range(len(labels))):
        pages = npm_doc10.loc[npm_doc10["id"]==ids[i]].iloc[0]["predicted_pages"] + bm25_doc10.loc[bm25_doc10["id"]==ids[i]].iloc[0]["predicted_pages"]
        pages = list(set(get_pred(pages)))
        predicted_pages.append(pages)
        # print(predicted_pages[i])

        if labels[i] == "NOT ENOUGH INFO":
            recall += 1
            continue

        answers = []
        for j in range(len(evidences[i][0])):
            answers.append(evidences[i][0][j][2])
        hits = len(set(pages).intersection(set(answers)))
        counts = len(set(answers))
        if hits == counts:
            recall += 1
    print("npm + bm25 + link recall:", recall/len(ids))
    # save train_doc.jsonl
    save_doc(TRAIN_DATA, predicted_pages, "train")


if args.mode == "test":
    npm_doc10 = pd.read_csv("exp/test_npm_doc.csv")
    npm_doc10['predicted_pages'] = npm_doc10['predicted_pages'].apply(ast.literal_eval)
    bm25_doc10 = pd.read_csv("exp/test_bm25_doc.csv")
    bm25_doc10['predicted_pages'] = bm25_doc10['predicted_pages'].apply(ast.literal_eval)

    TEST_DATA = load_json("data/concat_test.jsonl")
    ids = list(x["id"] for x in TEST_DATA)

    predicted_pages = []
    for i in tqdm(range(len(ids))):
        pages = npm_doc10.loc[npm_doc10["id"]==ids[i]].iloc[0]["predicted_pages"] + bm25_doc10.loc[bm25_doc10["id"]==ids[i]].iloc[0]["predicted_pages"]
        predicted_pages.append(list(set(get_pred(pages))))

    save_doc(TEST_DATA, predicted_pages, "test")
