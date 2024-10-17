# Part1: Doc Retrieval
python code/npm_doc.py --mode train
python code/npm_doc.py --mode test
python code/bm25_doc.py --mode train
python code/bm25_doc.py --mode test
python code/link_doc.py --mode train
python code/link_doc.py --mode test

# Part2: Sent Retrieval
python code/bert_sent.py --mode train
python code/bert_sent.py --mode test

# Part3: Claim Verification
python code/get_p3_data.py --mode train
python code/get_p3_data.py --mode test
python code/xlnet_cv.py --mode train
# Final Result
python code/xlnet_cv.py --mode test
