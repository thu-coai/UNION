from nltk.stem import WordNetLemmatizer
lemma = WordNetLemmatizer().lemmatize
import nltk
pos_tag = nltk.pos_tag
from nltk.corpus import stopwords
import sys

mode = sys.argv[1]
file_dir = "./WritingPrompts/ini_data/" if "w" in mode else "./ROCStories/ini_data/"
file_name = "train.wp_target" if "w" in mode else "train.txt"

def get_avail_phrases():
    sw = set(stopwords.words('english'))
    avail_phrases = set()
    fin = open("./conceptnet_entity.csv", 'r')
    for i, line in enumerate(fin):
        avail_phrases.add(' '.join(line.strip().split("|||")[:-1]))
    avail_phrases = avail_phrases - sw
    fin.close()

    fin = open("./negation.txt", 'r')
    for i, line in enumerate(fin):
        avail_phrases.add(' '.join(line.strip().split()[1:]))
    fin.close()

    for w in [".", ",", "!", "?", "male", "female", "neutral"]:
        avail_phrases.add(w)

    return avail_phrases

avail_phrases = get_avail_phrases()

vocab = {}
with open("%s/%s"%(file_dir, file_name), "r") as fin1:
    for kkk, line in enumerate(fin1):
        if kkk % 1000 == 0:
            print(kkk)
        tmp = line.strip().split()
        pos = pos_tag(tmp)
        for word_pos in pos:
            if lemma(word_pos[0], 'v' if word_pos[1][0] == 'V' else 'n') not in avail_phrases:
                continue
            if word_pos[0] in vocab:
                vocab[word_pos[0]]["number"] += 1
                if word_pos[1] in vocab[word_pos[0]]:
                    vocab[word_pos[0]][word_pos[1]] += 1
                else:
                    vocab[word_pos[0]][word_pos[1]] = 1
            else:
                vocab[word_pos[0]] = {word_pos[1]:1, "number":1}
vocab_list = sorted(vocab, key=lambda x: vocab[x]["number"], reverse=True)
with open("%s/entity_vocab.txt"%file_dir, "w") as fout:
    for v in vocab_list:
        pos_list = sorted(vocab[v], key=vocab[v].get, reverse=True)
        pos_list.remove("number")
        fout.write("%s %d|||"%(v, vocab[v]["number"]) + "|||".join(["%s %d"%(p, vocab[v][p]) for p in pos_list]) + "\n")