from typing import Dict, Tuple
import json

import numpy as np
import pandas as pd
from pandarallel import pandarallel

from utils import (
    generate_evidence_to_wiki_pages_mapping,
    jsonl_dir_to_df,
    load_json,
)

pandarallel.initialize(progress_bar=True, verbose=0, nb_workers=4)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='train', type=str)
args = parser.parse_args()


def join_with_topk_evidence(
    df: pd.DataFrame,
    mapping: dict,
    mode: str = "train",
    topk: int = 5,
) -> pd.DataFrame:

    # format evidence column to List[List[Tuple[str, str, str, str]]]
    if "evidence" in df.columns:
        df["evidence"] = df["evidence"].parallel_map(
            lambda x: [[x]] if not isinstance(x[0], list) else [x]
            if not isinstance(x[0][0], list) else x)

    print(f"Extracting evidence_list for the {mode} mode ...")
    if mode == "eval":
        # extract evidence
        df["evidence_list"] = df["predicted_evidence"].parallel_map(lambda x: [
            mapping.get(evi_id, {}).get(str(evi_idx), "")
            for evi_id, evi_idx in x  # for each evidence list
        ][:topk] if isinstance(x, list) else [])
        print(df["evidence_list"][:5])
    else:
        # extract evidence
        df["evidence_list"] = df["evidence"].parallel_map(lambda x: [
            " ".join([  # join evidence
                mapping.get(evi_id, {}).get(str(evi_idx), "")
                for _, _, evi_id, evi_idx in evi_list
            ]) if isinstance(evi_list, list) else ""
            for evi_list in x  # for each evidence list
        ][:1] if isinstance(x, list) else [])

    return df

wiki_pages = jsonl_dir_to_df("data/wiki-pages")
mapping = generate_evidence_to_wiki_pages_mapping(wiki_pages,)
del wiki_pages



if args.mode == "train":

    TRAIN_DATA = load_json("data/concat_train.jsonl")
    train_df = join_with_topk_evidence(
        pd.DataFrame(TRAIN_DATA),
        mapping,
        topk=5,
    )

    # clean repeated id
    targets = 1
    while targets > 0:
        targets = 0
        for id0, index in enumerate(train_df['id']):
          n_repeat = len(train_df.loc[train_df['id']==index,'id'])
          if n_repeat > 1:
            targets += 1
            train_df.loc[train_df['id']==index,'id'] = np.arange(index, index+n_repeat)


    # random sample evidence for NEI
    for index in train_df['id']:
      if len(train_df.loc[train_df['id']==index,'evidence_list'].iloc[0][0]) != 0:
        rs = train_df.loc[train_df['id']==index,'evidence_list'].values
      if len(train_df.loc[train_df['id']==index,'evidence_list'].iloc[0][0]) == 0:
        train_df.loc[train_df['id']==index,'evidence_list'] = rs


    # check evidence is not None
    for x in train_df['evidence_list'].values:
      if type(x) != list:
        print(x)
        print(type(x))

    train_df.to_csv("exp/train_data.csv", index=False)


if args.mode == "test":

    TEST_DATA = load_json("exp/test_sent.jsonl")
    test_df = join_with_topk_evidence(
            pd.DataFrame(TEST_DATA),
            mapping,
            mode="eval",
            topk=5,
    )
    test_df.to_csv("exp/test_data.csv", index=False)
