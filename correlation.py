import numpy as np
import copy
from scipy.stats import pearsonr, spearmanr, kendalltau
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import corpus_bleu
import os
from utils import get_pure_correlation, get_correlation_func, read, _type, data_dir, get_score

os.system("rm -r ./figure")
os.system("mkdir ./figure")

eval_story, human_score, reference, reference_ipt, reference_opt = get_score()
print("length of stories:", len(eval_story))
get_correlation = get_correlation_func(human_score)

metric_dir = "%s/metric_output"%data_dir
metric = {
    "human-score": human_score,

    "bleu": read("bleu.txt", metric_dir),
    "moverscore": read("moverscore.txt", metric_dir),
    "ruber-ref": read("ruber_ref_score.txt", metric_dir),

    "ppl": read("ppl.txt", metric_dir),
    "ruber-unref": read("ruber_unref_score.txt", metric_dir),
    "dis": read("dis.txt", metric_dir),
    "union": read("union.txt", metric_dir),
    "union-recon": read("union_recon.txt", metric_dir),

    "ruber": read("ruber_score.txt", metric_dir),
    "bleurt": read("bleurt.txt", metric_dir),
}


metric_list = list(metric.keys())
for a in sorted(metric_list):
    if a != "human-score":
        p, s ,t = get_correlation(metric[a], name=a)
        print(a, "(%+.4f, %+.4f, %+.4f)"%(p[0], s[0], t[0]), p[1], s[1], t[1])