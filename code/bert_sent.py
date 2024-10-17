# built-in libs
from pathlib import Path
from typing import Dict, List, Set, Tuple, Union

# third-party libs
import json
import numpy as np
import pandas as pd
from pandarallel import pandarallel
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_scheduler,
)

from dataset import BERTDataset, Dataset

# local libs
from utils import (
    generate_evidence_to_wiki_pages_mapping,
    jsonl_dir_to_df,
    load_json,
    load_model,
    set_lr_scheduler,
)

pandarallel.initialize(progress_bar=True, verbose=0, nb_workers=10)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='train', type=str)
args = parser.parse_args()


def evidence_macro_recall(
    instance: Dict,
    top_rows: pd.DataFrame,
) -> Tuple[float, float]:
    """Calculate recall for sentence retrieval
    This function is modified from fever-scorer.
    https://github.com/sheffieldnlp/fever-scorer/blob/master/src/fever/scorer.py

    Args:
        instance (dict): a row of the dev set (dev.jsonl) of test set (test.jsonl)
        top_rows (pd.DataFrame): our predictions with the top probabilities

        IMPORTANT!!!
        instance (dict) should have the key of `evidence`.
        top_rows (pd.DataFrame) should have a column `predicted_evidence`.

    Returns:
        Tuple[float, float]:
        [1]: relevant and retrieved (numerator of recall)
        [2]: relevant (denominator of recall)
    """
    # We only want to score F1/Precision/Recall of recalled evidence for NEI claims
    if instance["label"].upper() != "NOT ENOUGH INFO":
        # If there's no evidence to predict, return 1
        if len(instance["evidence"]) == 0 or all(
            [len(eg) == 0 for eg in instance]):
            return 1.0, 1.0

        claim = instance["claim"]

        predicted_evidence = top_rows[top_rows["claim"] == claim]["predicted_evidence"].tolist()

        for evidence_group in instance["evidence"]:
            evidence = [[e[2], e[3]] for e in evidence_group]
            if all([item in predicted_evidence for item in evidence]):
                # We only want to score complete groups of evidence. Incomplete
                # groups are worthless.
                return 1.0, 1.0
        return 0.0, 1.0
    return 0.0, 0.0

def evaluate_retrieval(
    probs: np.ndarray,
    df_evidences: pd.DataFrame,
    ground_truths: pd.DataFrame,
    top_n: int = 5,
    cal_scores: bool = True,
    save_name: str = None,
) -> Dict[str, float]:
    """Calculate the scores of sentence retrieval

    Args:
        probs (np.ndarray): probabilities of the candidate retrieved sentences
        df_evidences (pd.DataFrame): the candiate evidence sentences paired with claims
        ground_truths (pd.DataFrame): the loaded data of dev.jsonl or test.jsonl
        top_n (int, optional): the number of the retrieved sentences. Defaults to 2.

    Returns:
        Dict[str, float]: F1 score, precision, and recall
    """
    df_evidences["prob"] = probs
    top_rows = (
        df_evidences.groupby("claim").apply(
        lambda x: x.nlargest(top_n, "prob"))
        .reset_index(drop=True)
    )

    if cal_scores:
        # macro_precision = 0
        # macro_precision_hits = 0
        macro_recall = 0
        macro_recall_hits = 0

        for i, instance in enumerate(ground_truths):
            # macro_prec = evidence_macro_precision(instance, top_rows)
            # macro_precision += macro_prec[0]
            # macro_precision_hits += macro_prec[1]

            macro_rec = evidence_macro_recall(instance, top_rows)
            macro_recall += macro_rec[0]
            macro_recall_hits += macro_rec[1]

        # pr = (macro_precision /
        #       macro_precision_hits) if macro_precision_hits > 0 else 1.0
        rec = (macro_recall /
               macro_recall_hits) if macro_recall_hits > 0 else 0.0
        # f1 = 2.0 * pr * rec / (pr + rec)

    if save_name is not None:
        # write doc7_sent5 file
        with open(f"exp/{save_name}", "w") as f:
            for instance in ground_truths:
                claim = instance["claim"]
                predicted_evidence = top_rows[
                    top_rows["claim"] == claim]["predicted_evidence"].tolist()
                instance["predicted_evidence"] = predicted_evidence
                f.write(json.dumps(instance, ensure_ascii=False) + "\n")

    if cal_scores:
        # return {"F1 score": f1, "Precision": pr, "Recall": rec}
        return {"Recall": rec}

def get_predicted_probs(
    model: nn.Module,
    dataloader: Dataset,
    device: torch.device,
) -> np.ndarray:
    """Inference script to get probabilites for the candidate evidence sentences

    Args:
        model: the one from HuggingFace Transformers
        dataloader: devset or testset in torch dataloader

    Returns:
        np.ndarray: probabilites of the candidate evidence sentences
    """
    model.eval()
    probs = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            probs.extend(torch.softmax(logits, dim=1)[:, 1].tolist())

    return np.array(probs)

class SentRetrievalBERTDataset(BERTDataset):
    """AicupTopkEvidenceBERTDataset class for AICUP dataset with top-k evidence sentences."""

    def __getitem__(
        self,
        idx: int,
        **kwargs,
    ) -> Tuple[Dict[str, torch.Tensor], int]:
        item = self.data.iloc[idx]
        sentA = item["claim"]
        sentB = item["text"]

        # claim [SEP] text
        concat = self.tokenizer(
            sentA,
            sentB,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )
        concat_ten = {k: torch.tensor(v) for k, v in concat.items()}
        if "label" in item:
            concat_ten["labels"] = torch.tensor(item["label"])

        return concat_ten

def pair_with_wiki_sentences(
    mapping: Dict[str, Dict[int, str]],
    df: pd.DataFrame,
    negative_ratio: float,
) -> pd.DataFrame:
    """Only for creating train sentences."""
    claims = []
    sentences = []
    labels = []

    # positive
    for i in range(len(df)):
        if df["label"].iloc[i] == "NOT ENOUGH INFO":
            continue

        claim = df["claim"].iloc[i]
        evidence_sets = df["evidence"].iloc[i]
        for evidence_set in evidence_sets:
            sents = []
            for evidence in evidence_set:
                # evidence[2] is the page title
                page = evidence[2].replace(" ", "_")
                # the only page with weird name
                if page == "臺灣海峽危機#第二次臺灣海峽危機（1958）":
                    continue
                # evidence[3] is in form of int however, mapping requires str
                sent_idx = str(evidence[3])
                sents.append(mapping[page][sent_idx])

            whole_evidence = " ".join(sents)

            claims.append(claim)
            sentences.append(whole_evidence)
            labels.append(1)

    # negative
    for i in range(len(df)):
        if df["label"].iloc[i] == "NOT ENOUGH INFO":
            continue
        claim = df["claim"].iloc[i]

        evidence_set = set([(evidence[2], evidence[3])
                            for evidences in df["evidence"][i]
                            for evidence in evidences])
        predicted_pages = df["predicted_pages"][i]
        for page in predicted_pages:
            page = page.replace(" ", "_")
            try:
                page_sent_id_pairs = [
                    (page, sent_idx) for sent_idx in mapping[page].keys()
                ]
            except KeyError:
                # print(f"{page} is not in our Wiki db.")
                continue

            for pair in page_sent_id_pairs:
                if pair in evidence_set:
                    continue
                text = mapping[page][pair[1]]
                # `np.random.rand(1) <= 0.05`: Control not to add too many negative samples
                #if text != "" and np.random.rand(1) <= negative_ratio:
                if len(text) >= 10 and np.random.rand(1) <= negative_ratio:
                    claims.append(claim)
                    sentences.append(text)
                    labels.append(0)

    return pd.DataFrame({"claim": claims, "text": sentences, "label": labels})


def pair_with_wiki_sentences_eval(
    mapping: Dict[str, Dict[int, str]],
    df: pd.DataFrame,
    is_testset: bool = False,
) -> pd.DataFrame:
    """Only for creating dev and test sentences."""
    claims = []
    sentences = []
    evidence = []
    predicted_evidence = []

    # negative
    for i in range(len(df)):
        # if df["label"].iloc[i] == "NOT ENOUGH INFO":
        #     continue
        claim = df["claim"].iloc[i]

        predicted_pages = df["predicted_pages"][i]
        for page in predicted_pages:
            page = page.replace(" ", "_")
            try:
                page_sent_id_pairs = [(page, k) for k in mapping[page]]
            except KeyError:
                # print(f"{page} is not in our Wiki db.")
                continue

            for page_name, sentence_id in page_sent_id_pairs:
                text = mapping[page][sentence_id]
                if text != "":
                    claims.append(claim)
                    sentences.append(text)
                    if not is_testset:
                        evidence.append(df["evidence"].iloc[i])
                    predicted_evidence.append([page_name, int(sentence_id)])

    return pd.DataFrame({
        "claim": claims,
        "text": sentences,
        "evidence": evidence if not is_testset else None,
        "predicted_evidence": predicted_evidence,
    })


MODEL_NAME = "bert-base-chinese"
NUM_EPOCHS = 1
LR = 2e-5
TRAIN_BATCH_SIZE = 128
TEST_BATCH_SIZE = 256
NEGATIVE_RATIO = 0.09
VALIDATION_STEP = 250
TOP_N = 5
SEED = 42

LABEL2ID: Dict[str, int] = {
    "supports": 0,
    "refutes": 1,
    "NOT ENOUGH INFO": 2,
}
ID2LABEL: Dict[int, str] = {v: k for k, v in LABEL2ID.items()}

wiki_pages = jsonl_dir_to_df("data/wiki-pages")
mapping = generate_evidence_to_wiki_pages_mapping(wiki_pages)
del wiki_pages


if args.mode == "train":
    TRAIN_DATA = load_json("data/concat_train.jsonl")
    DOC_DATA = load_json("exp/train_doc.jsonl")

    _y = [LABEL2ID[data["label"]] for data in TRAIN_DATA]
    TRAIN_GT, DEV_GT = train_test_split(
        DOC_DATA,
        test_size=0.2,
        random_state=SEED,
        shuffle=True,
        stratify=_y,
    )

    train_df = pair_with_wiki_sentences(
        mapping,
        pd.DataFrame(TRAIN_GT),
        NEGATIVE_RATIO,
    )
    counts = train_df["label"].value_counts()
    print("Now using the following train data with 0 (Negative) and 1 (Positive)")
    print(counts)

    dev_evidences = pair_with_wiki_sentences_eval(mapping, pd.DataFrame(DEV_GT))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = SentRetrievalBERTDataset(train_df, tokenizer=tokenizer)
    val_dataset = SentRetrievalBERTDataset(dev_evidences, tokenizer=tokenizer)

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=TRAIN_BATCH_SIZE,
    )
    eval_dataloader = DataLoader(val_dataset, batch_size=TEST_BATCH_SIZE)
    del train_df

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=LR)
    num_training_steps = NUM_EPOCHS * len(train_dataloader)
    lr_scheduler = set_lr_scheduler(optimizer, num_training_steps)
    writer = SummaryWriter("checkpoint/sent_retrieve/")

    progress_bar = tqdm(range(num_training_steps))
    current_steps = 0

    for epoch in tqdm(range(NUM_EPOCHS)):
        model.train()

        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            #print("OUTPUTS: ", outputs)
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            writer.add_scalar("training_loss", loss.item(), current_steps)

            y_pred = torch.argmax(outputs.logits, dim=1).tolist()
            y_true = batch["labels"].tolist()
            #print("ypred: ", y_pred)
            #print("ytrue: ", y_true)

            current_steps += 1

            if current_steps % VALIDATION_STEP == 0 and current_steps > 0:
                print("Start validation")
                print("current_steps=", current_steps)
                probs = get_predicted_probs(model, eval_dataloader, device)

                val_results = evaluate_retrieval(
                    probs=probs,
                    df_evidences=dev_evidences,
                    ground_truths=DEV_GT,
                    top_n=TOP_N,
                )
                print(val_results)

                # log each metric separately to TensorBoard
                for metric_name, metric_value in val_results.items():
                    writer.add_scalar(
                        f"dev_{metric_name}",
                        metric_value,
                        current_steps,
                    )

    torch.save(model.state_dict(), 'checkpoint/sent_retrieve/bert.ckpt')
    print("Finished training!")

    print("Start validation")
    probs = get_predicted_probs(model, eval_dataloader, device)
    val_results = evaluate_retrieval(
        probs=probs,
        df_evidences=dev_evidences,
        ground_truths=DEV_GT,
        top_n=TOP_N,
    )
    print(f"Validation scores => {val_results}")

if args.mode == "test":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(device)
    model.load_state_dict(torch.load('checkpoint/sent_retrieve/bert.ckpt'))

    test_data = load_json("exp/test_doc.jsonl")
    test_evidences = pair_with_wiki_sentences_eval(
        mapping,
        pd.DataFrame(test_data),
        is_testset=True,
    )
    test_set = SentRetrievalBERTDataset(test_evidences, tokenizer)
    test_dataloader = DataLoader(test_set, batch_size=TEST_BATCH_SIZE)

    print("Start predicting the test data")
    probs = get_predicted_probs(model, test_dataloader, device)
    evaluate_retrieval(
        probs=probs,
        df_evidences=test_evidences,
        ground_truths=test_data,
        top_n=TOP_N,
        cal_scores=False,
        save_name=f"test_sent.jsonl",
    )
