import numpy as np
import random
import re
import copy
from nltk.corpus import stopwords
import nltk
pos_tag = nltk.pos_tag
from nltk.stem import WordNetLemmatizer
lemma = WordNetLemmatizer().lemmatize
import sys

function_word = [".", ",", "!", "?", "male", "female", "neutral"]
def get_avail_phrases():
    sw = set(stopwords.words('english'))
    avail_phrases = set()
    fin = open("./conceptnet_entity.csv", 'r')
    for i, line in enumerate(fin):
        avail_phrases.add(' '.join(line.strip().split("|||")[:-1]))
    avail_phrases = avail_phrases - sw
    fin.close()

    fin = open("./negation.txt", 'r')
    negation_word = []
    for i, line in enumerate(fin):
        word = ' '.join(line.strip().split()[1:])
        negation_word.append(word)
        avail_phrases.add(word)
    fin.close()

    for w in function_word:
        avail_phrases.add(w)

    with open("avail_phrases.txt", "w") as fout:
        for w in avail_phrases:
            fout.write(w+"\n")
    return avail_phrases, negation_word

avail_phrases, negation_word = get_avail_phrases()

def output(st, fout):
    if "w" in data_dir:
        fout.write(" ".join(st)+"\n")
    else:
        for sen in st:
            fout.write(sen+"\n")
        fout.write("-"*5+"\n")

def repeat_sentence(st):
    # repeat one sentence and delete the original sentence
    idx = np.random.choice(np.arange(len(st))[1:], 1 + int(len(st)/2), replace=False).tolist()
    s = min(idx)
    tmp_st = copy.deepcopy(st)
    for l in idx:
        tmp_st[l] = copy.deepcopy(tmp_st[s])
    return tmp_st

def repeat_ngram(st):
    # repeat ngram in one sentence 1~4
    def repeat_sen_gram(st):
        flag = True
        for _ in range(10):
            try:
                idx = np.random.choice(np.arange(len(st))[1:])
                gram_num = np.random.choice(np.arange(5)[1:])
                split_sen = st[idx].strip().split()
                pointer_st = np.random.choice(np.arange(len(split_sen)))
                pointer_ed = pointer_st + gram_num
                if pointer_ed > len(split_sen):
                    pointer_ed = pointer_st
                    pointer_st = pointer_ed - gram_num
                    if pointer_st < 0:
                        continue
                    else:
                        flag = False
                        break
            except:
                continue
        if flag:
            return copy.deepcopy(st)
        sen1, sen2, sen3 = " ".join(split_sen[:pointer_st]), " ".join(split_sen[pointer_st:pointer_ed]), " ".join(split_sen[pointer_ed:])
        tmp_st = copy.deepcopy(st)
        tmp_st[idx] = " ".join([sen1, sen2, sen2, sen3]).strip()
        return tmp_st
    for i in range(int(len(st)/2)):
        st = repeat_sen_gram(st)
    return st

def replace_sentence(st):
    flag = True
    for _ in range(10):
        try:
            tmp_st = copy.deepcopy(st)
            idxs = np.random.choice(np.arange(len(st))[1:], np.random.choice(np.arange(1, len(st))), replace=False)
            replace_st_id = np.random.choice(np.arange(len(story)))
            for idx in idxs:
                tmp_st[idx] = np.random.choice(story[replace_st_id])
            flag = False
            break
        except:
            continue
    if flag:
        return copy.deepcopy(st)
    return tmp_st

def change_neg_helper(sen):
    def pro(s):
        final_sen = " ".join(s)
        return final_sen
    sen = sen.strip().split()        
    for i, n in enumerate(sen):
        if n in negation_word:
            del sen[i]
            return pro(sen)
    neg_list = ["not", "n't"]
    for i, n in enumerate(sen):
        if n in ["would", "will", "can", "could", "may", "might", "shall", "should", "do", "does", "did", "am", "is", "are", "was", "were", "be", "been"]:
            sen.insert(i+1, np.random.choice(neg_list))
            return pro(sen)
    pos_sen = pos_tag(sen)
    for i, n in enumerate(pos_sen):
        if n[1] == "VB":
            sen.insert(i, "do " + np.random.choice(neg_list))
            return pro(sen)
        elif n[1] == "VBD":
            sen[i] = lemma(sen[i], "v")
            sen.insert(i, "did " + np.random.choice(neg_list))
            return pro(sen)
        elif n[1] == "VBG":
            sen.insert(i, np.random.choice(neg_list))
            return pro(sen)
        elif n[1] == "VBN":
            sen.insert(i, np.random.choice(neg_list))
            return pro(sen)
        elif n[1] == "VBP":
            sen.insert(i, "do " + np.random.choice(neg_list))
            return pro(sen)
        elif n[1] == "VBZ":
            sen[i] = lemma(sen[i], "v")
            sen.insert(i, "does " + np.random.choice(neg_list))
            return pro(sen)
    print("VERB ERROR")
    return None

anotomy_word = {}
all_num, anotomy_num = 0, 0
with open("./conceptnet_antonym.txt", "r") as fin:
    for line in fin:
        tmp = line.strip().split("|||")
        if len(tmp) == 3:
            h, t = tmp[0], tmp[2].split()
            if h in anotomy_word:
                anotomy_word[h] += t
            else:
                anotomy_word[h] = t[:]

def change_neg_sentence(st):
    flag = True
    for _ in range(10):
        try:
            tmp_st = copy.deepcopy(st)
            idxs = np.random.choice(np.arange(len(st))[1:], np.random.choice(np.arange(1, len(st))), replace=False)
            for idx in idxs:
                tmp_st_idx = change_neg_helper(st[idx])
                if tmp_st_idx is not None: 
                    tmp_st[idx] = tmp_st_idx
                    flag = False
            if flag == False:
                break
        except:
            continue
    if flag:
        return copy.deepcopy(st)
    return tmp_st

def replace_word(st):
    global all_num, anotomy_num
    def replace_one_word(st):
        anotomy = False
        flag = True
        for _ in range(100):
            tmp_st = copy.deepcopy(st)
            idx = np.random.choice(np.arange(len(st))[1:])
            split_sen = tmp_st[idx].split()
            pos_split_sen = pos_tag(split_sen)
            avail_w_id = []
            for w_id, w in enumerate(split_sen):
                if (w in avail_phrases and w not in function_word and "[" not in w):
                    avail_w_id.append(w_id)
            if len(avail_w_id) == 0: continue
            word_id = np.random.choice(avail_w_id)
            if pos_split_sen[word_id][1] not in pos_vocab_entity: continue
            lemma_word = lemma(pos_split_sen[word_id][0], 'v' if pos_split_sen[word_id][1][0] == 'V' else 'n')
            if lemma_word in anotomy_word:
                replace_word = np.random.choice(anotomy_word[lemma_word])
                anotomy = True
            else:
                word_freq = pos_vocab_entity[pos_split_sen[word_id][1]]
                replace_word = ""
                flag_in = True
                for _ in range(10):
                    replace_word = np.random.choice(word_freq["word"], p=word_freq["freq"]/np.sum(word_freq["freq"]))
                    if len(word_freq["word"]) == 1 or replace_word != pos_split_sen[word_id][0]:
                        flag_in = False
                        break
                if flag_in:
                    replace_word = pos_split_sen[word_id][0]
                anotomy = False
            tmp_split_sen = copy.deepcopy(split_sen)
            split_sen[word_id] = replace_word
            tmp_st[idx] = " ".join(split_sen)
            flag = False
            break
        if flag:
            return copy.deepcopy(st), False
        return tmp_st, anotomy
    num = 0
    for idx in np.arange(len(st))[1:]:
        for word in st[idx].split():
            if word in avail_phrases:
                num += 1
    try:
        final_num = np.random.choice(np.arange(1, int(num*0.15+1)))
    except:
        final_num = 1
    for _ in range(final_num):
        st, anotomy = replace_one_word(st)
        all_num += 1
        if anotomy: anotomy_num += 1
    return st

def shuffle_sentence(st, n_sentence):
    def exchange(l, ids, target_ids):
        tmp_l = copy.deepcopy(l)
        for o_id, t_id in zip(ids, target_ids):
            tmp_l[o_id] = copy.deepcopy(l[t_id])
        return tmp_l
    # exchange n sentences
    flag = True
    for _ in range(10):
        sen_ids = np.random.choice(np.arange(len(st))[1:], n_sentence, replace=False)
        target_ids = np.random.permutation(sen_ids)
        tmp_st = exchange(st, sen_ids, target_ids)
        if st != tmp_st:
            flag = False
            break       
    if flag:
        return copy.deepcopy(st)
    return tmp_st
def get_pos_vocab(dir):
    pos_vocab_entity = {}
    with open("%s/entity_vocab.txt"%dir, "r") as fin:
        for line in fin:
            tmp = line.strip().split("|||")
            word = tmp[0].split()[0]
            pos = tmp[1:]
            for p in pos:
                pp = p.split()
                if pp[0] in pos_vocab_entity:
                    pos_vocab_entity[pp[0]]["word"].append(word)
                    pos_vocab_entity[pp[0]]["freq"].append(float(pp[1]))
                else:
                    pos_vocab_entity[pp[0]] = {"word":[word], "freq":[float(pp[1])]}
    return pos_vocab_entity
# ========================================================================================

name_list = ["test", "dev", "train"]
data_dir = "./%s/ini_data"%("WritingPrompts" if "w" in sys.argv[1] else "ROCStories")
output_dir = "%s/train_data"%("WritingPrompts" if "w" in sys.argv[1] else "ROCStories")

# type_dict = {"repeat":0.6, "replace":0.15, "shuffle":0.15, "neg":0.1}
type_dict = {"repeat":0.1, "replace":0.3, "shuffle":0.4, "neg":0.2}
type_list = list(type_dict.keys())
type_prob_list = []
for t in type_list:
    type_prob_list.append(type_dict[t])

time_list = [1,2,3,4]
# time_prob_list = [0.2,0.4,0.3,0.1]
time_prob_list = [0.5,0.2,0.2,0.1]

pos_vocab_entity = get_pos_vocab(data_dir)
for name in name_list:
    if "w" in data_dir.lower():
        with open("%s/%s.wp_source"%(data_dir, name), "r") as fin1:
            with open("%s/%s.wp_target"%(data_dir, name), "r") as fin2:
                story, tmp = [], []
                for k, line in enumerate(fin2):
                    src = fin1.readline().strip()
                    if src[-1].isalpha():
                        src = src + " ."
                    tmp.append(src)
                    for sen in line.strip().split(".")[:-1]:
                        if sen.strip() != "":
                            tmp.append(sen.strip()+" .")
                    if len(tmp) >= 4:
                        story.append(tmp)
                    tmp = []
    else:
        with open("%s/%s.txt"%(data_dir, name), "r") as fin:
            story, tmp = [], []
            for k, line in enumerate(fin):
                i = k + 1
                if i % 6 == 0:
                    story.append(tmp)
                    tmp = []
                else:
                    sen = line.strip()
                    tmp.append(sen+" ." if sen[-1].isalpha() else sen)

    with open("%s/%s_human.txt"%(output_dir, name), "w") as fout:
        for st_id, st in enumerate(story):
            output(st, fout)

    prefix = "%s/%s_negative"%(output_dir, name)
    with open("%s.txt"%(prefix), "w") as fout:
        for st_id, st in enumerate(story):
            chaotic_list = np.random.choice(type_list, 
                np.random.choice(time_list, p=time_prob_list), replace=False, p=type_prob_list/np.sum(type_prob_list)).tolist()
            print(chaotic_list)
            for c in chaotic_list:
                if c == "repeat":
                    if random.random() < 0.7:
                        st = repeat_sentence(st)
                    else:
                        st = repeat_ngram(st)
                if c == "replace":
                    if random.random() < 0.5:
                        # replace one sentence
                        st = replace_sentence(st)
                    else:
                        # replace one word
                        st = replace_word(st)
                if c == "shuffle":
                    n_sentence = int(np.random.choice(np.arange(1,len(st)-1)+1))
                    st = shuffle_sentence(st, n_sentence)
                if c == "neg":
                    st = change_neg_sentence(st)
            output(st, fout)



    print("Anotomy:", anotomy_num)
    print("All:", all_num)

