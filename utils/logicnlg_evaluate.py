import json
import nltk
import sys
from multiprocessing import Pool
import warnings
import argparse
warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, type=str)
parser.add_argument("--reference", required=True, type=str)
args = parser.parse_args() 

with open(args.input, 'r') as f:
    hypothesis = json.load(f)

with open(args.reference, 'r') as f:
    reference = json.load(f)


def get_reference(table_id, option='test'):
    assert option == 'test'
    entry = reference[table_id]
    return [_[0].lower().split(' ') for _ in entry]


def func_compute_bleu(f):
    sent_bleus_1, sent_bleus_2, sent_bleus_3, sent_bleus_4 = [], [], [], []
    reference = get_reference(f)
    for hyp in hypothesis[f]:
        hyps = hyp.lower().split()       
        sent_bleus_1.append(nltk.translate.bleu_score.sentence_bleu(reference, hyps, weights=(1, 0, 0)))
        sent_bleus_2.append(nltk.translate.bleu_score.sentence_bleu(reference, hyps, weights=(0.5, 0.5, 0)))
        sent_bleus_3.append(nltk.translate.bleu_score.sentence_bleu(reference, hyps, weights=(0.33, 0.33, 0.33)))    
        sent_bleus_4.append(nltk.translate.bleu_score.sentence_bleu(reference, hyps, weights=(0.25, 0.25, 0.25, 0.25)))    

    return sent_bleus_1, sent_bleus_2, sent_bleus_3, sent_bleus_4


if __name__ == "__main__":
    assert len(hypothesis) == len(reference)

    # calculate sentence-level BLEU score
    pool = Pool(64)
    bleus = pool.map(func_compute_bleu, hypothesis.keys())

    sent_bleus_1, sent_bleus_2, sent_bleus_3, sent_bleus_4 = [], [], [], []
    for _ in bleus:
        sent_bleus_1.extend(_[0])
        sent_bleus_2.extend(_[1])
        sent_bleus_3.extend(_[2])
        sent_bleus_4.extend(_[3])

    bleu_1 = sum(sent_bleus_1) / len(sent_bleus_1)
    bleu_2 = sum(sent_bleus_2) / len(sent_bleus_2)
    bleu_3 = sum(sent_bleus_3) / len(sent_bleus_3)
    bleu_4 = sum(sent_bleus_4) / len(sent_bleus_4) 

    print("Sentence BLEU: {}/{}/{}/{}".format(bleu_1, bleu_2, bleu_3, bleu_4))
