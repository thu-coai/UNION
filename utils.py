from scipy.stats import pearsonr, spearmanr, kendalltau
from matplotlib import pyplot as plt
import numpy as np
import os
import sys
_type = sys.argv[1]

if "roc" in _type:
    data_dir = "./Data/ROCStories"
elif "wp" in _type:
    data_dir = "./Data/WritingPrompts"

def get_pure_correlation(h, m):
    pear = pearsonr(m, h)
    spear = spearmanr(m, h)
    tau = kendalltau(m, h)
    return pear, spear, tau

pointer = 0
def get_correlation_func(human):
    human_perb = (np.array(human) + np.random.normal(0,0.05,size=(len(human)))).tolist()
    def get_correlation(metric, dir_name="", name=""):
        global pointer
        pear = pearsonr(metric, human)
        spear = spearmanr(metric, human)
        tau = kendalltau(metric, human)
        plt.figure(pointer)
        plt.plot(human_perb, metric, ".")
        plt.xlabel("human")
        plt.ylabel(dir_name+"_"+name)
        figure_dir = "./figure/%s"%(dir_name)
        if not os.path.exists(figure_dir):
            os.system("mkdir %s"%figure_dir)
        plt.savefig("%s/result_%s.png"%(figure_dir, name))
        plt.close(pointer)
        pointer += 1
        return pear, spear, tau
    return get_correlation

def read(name, dir_=None):
    dir_name = data_dir if dir_ is None else dir_
    l = []
    with open("%s/%s"%(dir_name, name), "r") as fin:
        for line in fin:
            try:
                l.append(float(line.strip()))
            except:
                l.append(line.strip())
    return l

def get_score():
    ant_data_dir = "%s/ant_data/"%data_dir
    reference = read("reference.txt", ant_data_dir)
    reference_ipt = read("reference_ipt.txt", ant_data_dir)
    reference_opt = read("reference_opt.txt", ant_data_dir)
    eval_story = []

    with open("%s/ant_data.txt"%(ant_data_dir), "r") as fin:
        with open("%s/ant_data_all.txt"%(ant_data_dir), "r") as fin_all:
            id_dict = {"Repeated Plots":[],
                    "Poor Coherence":[],
                    "Conflicting Logic":[],
                    "Chaotic Scenes":[],
                    "Others":[]}
            annotator = 7
            for iii in range(0,annotator+1):
                id_dict["%d/%d"%(iii, annotator)] = []
            for i, line in enumerate(fin):
                tmp = line.strip().split("|||")
                true_st_id = list(map(int, tmp[0].strip().split("///")))
                eval_st = tmp[1].strip()
                eval_st_opt = eval_st[len(reference_ipt[true_st_id[0]]):]
                score = list(map(float, tmp[2].strip().split()))
                line_all = fin_all.readline()
                score_all = list(map(int, line_all.strip().split("|||")[2].strip().split()))
                mean_score = np.mean(score)
                eval_story.append({"true_st": true_st_id, "st":eval_st, "st_opt":eval_st_opt, "score":score, "human":mean_score})
                for iii in range(0, annotator+1):
                    if mean_score == iii/annotator:
                        id_dict["%d/%d"%(iii, annotator)].append(i)
                if "roc" in _type:
                    if score_all.count(1)>=1: id_dict["Repeated Plots"].append(i)
                    if score_all.count(2)>=1: id_dict["Poor Coherence"].append(i)
                    if score_all.count(3)>=1: id_dict["Conflicting Logic"].append(i)
                    if score_all.count(4)>=1: id_dict["Chaotic Scenes"].append(i)
                    if score_all.count(5)>=1: id_dict["Others"].append(i)

        human_score = [s["human"] for s in eval_story]
        for kkk in id_dict:
            print(kkk, len(id_dict[kkk]))
    return eval_story, human_score, reference, reference_ipt, reference_opt
