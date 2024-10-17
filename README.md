# AICUP-2023 真相只有一個: 事實文字檢索與查核競賽

## TEAM_3514
Public LB Score: 0.407482
Private LB Score: 0.522138

## Run our code
- Use our fine-tune model to get the test result
```bash
./quick_run.sh
```
- Rerun the whole expirement (takes some hours)
```bash
./run.sh
```

## Folder explanation
- data: we put concated training & testing data and wiki data here
- exp: we save the output of each stage here
- code: the code we used to get the final score on leaderboard
- checkpoint: we save 2 fine-tune model.ckpt here
- sub: the final submission data is saved here

## Python file explanation
- `utils.py` `dataset.py` : some useful function and dataset class

- `npm_doc.py` : document retrival by Noun Phrase Matching
```bash
python npm_doc.py --mode train
python npm_doc.py --mode test
```
| input file | output file | Note |
| :-----: | :----: | :----: |
| concat_train.jsonl, wiki-pages | train_npm_doc.csv | take about 2 hours |
| concat_test.jsonl, wiki-pages | test_npm_doc.csv | take about 2 hours |

- `bm25_doc.py` : document retrival by BM25
```bash
python bm25_doc.py --mode train
python bm25_doc.py --mode test
```
| input file | output file | Note |
| :-----: | :----: | :----: |
| concat_train.jsonl, wiki-pages | train_bm25_doc.csv | take 20+ hours |
| concat_test.jsonl, wiki-pages | test_bm25_doc.csv | take 20+ hours |

- `link_doc.py` : concat results of NPM & BM25 then retrieve more docs by Link method
```bash
python link_doc.py --mode train
python link_doc.py --mode test
```
| input file | output file | Note |
| :-----: | :----: | :----: |
| concat_train.jsonl, wiki_page.csv, train_npm_doc.csv, train_bm25_doc.csv | train_doc.jsonl | take 20+ hours |
| concat_test.jsonl, wiki_page.csv, test_npm_doc.csv, test_bm25_doc.csv | test_doc.jsonl | take 20+ hours |

- `bert_sent.py` : sentence retrieval by bert model
```bash
python bert_sent.py --mode train
python bert_sent.py --mode test
```
| input file | output file | Note |
| :-----: | :----: | :----: |
| concat_train.jsonl, wiki-pages, train_doc.jsonl | bert.ckpt | take about 2 hours |
| wiki-pages, test_doc.jsonl, bert.ckpt | test_sent.jsonl | take about 2 hours |

- `get_p3_data.py` : get train_data.csv & test_data.csv for next stage
```bash
python get_p3_data.py --mode train
python get_p3_data.py --mode test
```

- `xlnet_cv.py` : claim verification by xlnet model
```bash
python xlnet_cv.py --mode train
python xlnet_cv.py --mode test
```
| input file | output file | Note |
| :-----: | :----: | :----: |
| train_data.csv | xlet.ckpt | take about ? hours |
| test_data.csv, xlnet.ckpt | 0602xlnet_sub.jsonl | - |









