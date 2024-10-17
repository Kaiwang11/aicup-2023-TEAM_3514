# built-in libs
import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple, Union

# 3rd party libs
import hanlp
import opencc
import pandas as pd
import wikipedia
from hanlp.components.pipeline import Pipeline
from pandarallel import pandarallel

# our own libs
from utils import load_json

pandarallel.initialize(progress_bar=True, verbose=0, nb_workers=10)
wikipedia.set_lang("zh")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='train', type=str)
args = parser.parse_args()


CONVERTER_T2S = opencc.OpenCC("t2s.json")
CONVERTER_S2T = opencc.OpenCC("s2t.json")

def do_st_corrections(text: str) -> str:
    simplified = CONVERTER_T2S.convert(text)
    return CONVERTER_S2T.convert(simplified)

def get_nps_hanlp(predictor, d):
    claim = d["claim"]
    tree = predictor(claim)["con"]
    nps = [
        do_st_corrections("".join(subtree.leaves()))
        for subtree in tree.subtrees(lambda t: t.label() == "NP")
    ]
    return nps

predictor = (hanlp.pipeline().append(
    hanlp.load("FINE_ELECTRA_SMALL_ZH"),
    output_key="tok",
).append(
    hanlp.load("CTB9_CON_ELECTRA_SMALL"),
    output_key="con",
    input_key="tok",
))

def get_pred_pages(series_data):
    results = []
    tmp_muji = []
    mapping = {}
    claim = series_data["claim"]
    nps = series_data["hanlp_results"]
    first_wiki_term = []

    for i, np in enumerate(nps):
        wiki_search_results = [
            do_st_corrections(w) for w in wikipedia.search(np)
        ]

        wiki_set = [re.sub(r"\s\(\S+\)", "", w) for w in wiki_search_results]
        wiki_df = pd.DataFrame({
            "wiki_set": wiki_set,
            "wiki_results": wiki_search_results
        })

        grouped_df = wiki_df.groupby("wiki_set", sort=False).first()
        candidates = grouped_df["wiki_results"].tolist()
        muji = grouped_df.index.tolist()

        for prefix, term in zip(muji, candidates):
            if prefix not in tmp_muji:
                matched = False

                if i == 0:
                    first_wiki_term.append(term)

                if (((new_term := term) in claim) or
                        ((new_term := term.replace("·", "")) in claim) or
                        ((new_term := term.split(" ")[0]) in claim) or
                        ((new_term := term.replace("-", " ")) in claim)):
                    matched = True

                elif "·" in term:
                    splitted = term.split("·")
                    for split in splitted:
                        if (new_term := split) in claim:
                            matched = True
                            break

                if matched:
                    term = term.replace(" ", "_")
                    term = term.replace("-", "")
                    results.append(term)
                    mapping[term] = claim.find(new_term)
                    tmp_muji.append(new_term)


    if len(results) > 10:
        assert -1 not in mapping.values()
        results = sorted(mapping, key=mapping.get)[:10]
    elif len(results) < 1:
        results = first_wiki_term

    return results


if args.mode == "train":
    TRAIN_DATA = load_json("data/concat_train.jsonl")

    hanlp_results = [get_nps_hanlp(predictor, d) for d in TRAIN_DATA]
    train_df = pd.DataFrame(TRAIN_DATA)
    train_df.loc[:, "hanlp_results"] = hanlp_results
    predicted_results = train_df.parallel_apply(get_pred_pages, axis=1)
    train_df['predicted_pages'] = predicted_results
    train_df[['id', 'predicted_pages']].to_csv("exp/train_npm_doc.csv", index=False)

    labels = list(x["label"] for x in TRAIN_DATA)
    evidences = list(x["evidence"] for x in TRAIN_DATA)
    recall = 0
    for i in range(len(evidences)):
        if labels[i] == "NOT ENOUGH INFO":
            recall += 1
            continue

        # compute recall
        answers = []
        for j in range(len(evidences[i][0])):
            answers.append(evidences[i][0][j][2])

        hits = len(set(predicted_results[i]).intersection(set(answers)))
        counts = len(set(answers))
        if hits == counts:
            recall += 1
    print("recall:", recall/len(evidences))

if args.mode == "test":
    TEST_DATA = load_json("data/concat_test.jsonl")

    hanlp_results = [get_nps_hanlp(predictor, d) for d in TEST_DATA]
    test_df = pd.DataFrame(TEST_DATA)
    test_df.loc[:, "hanlp_results"] = hanlp_results
    test_results = test_df.parallel_apply(get_pred_pages, axis=1)
    test_df['predicted_pages'] = test_results
    test_df[['id', 'predicted_pages']].to_csv("exp/test_npm_doc.csv", index=False)
