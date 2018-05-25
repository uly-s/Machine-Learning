import os
import re
import word2toke
import PreEmbed

def path2words(file, path="C:/Users/Grant/PycharmProjects/Machine-Learning/Random Encounter/", encoding='utf-8', reg="\w+|[^\w\s]"):
    path = os.path.join(path, file)
    file = open(path, 'r', encoding=encoding)
    words = []
    for line in file:
        line = line.lower()
        entry = re.findall(r"".join(reg), line)
        words.extend(entry)

    return words

def words2vocab(words, n=10000, prune=True, prune_rare=False, f=0.00002):
    return word2toke.words2vocab(words, n, prune=prune, prune_rare=prune_rare, f=f)

def words2sens(words):
    seqs = []
    index, start, end = 0, 0, 0

    for i, word in enumerate(words):
        end +=1
        if word == "." or word == "?" or word == "!" and i < len(words) and words[i+1] != ".":
            seq = words[start:end]
            seqs.append(seq)
            start = end

    return seqs

def uniqueSens(sens):
    return list(set([" ".join(word for word in sen) for sen in sens]))


def maxlen(seqs):
    max = 0
    for seq in seqs:
        n = len(seq)
        if n > max:
            max = n

    return max








def string2sents(text):
    caps = "([A-Z])"
    prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
    suffixes = "(Inc|Ltd|Jr|Sr|Co)"
    starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
    acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
    websites = "[.](com|net|org|io|gov)"

    text = " " + text + "  "
    text = text.replace("\n", " ")
    text = re.sub(prefixes, "\\1<prd>", text)
    text = re.sub(websites, "<prd>\\1", text)
    if "Ph.D" in text: text = text.replace("Ph.D.", "Ph<prd>D<prd>")
    text = re.sub("\s" + caps + "[.] ", " \\1<prd> ", text)
    text = re.sub(acronyms + " " + starters, "\\1<stop> \\2", text)
    text = re.sub(caps + "[.]" + caps + "[.]" + caps + "[.]", "\\1<prd>\\2<prd>\\3<prd>", text)
    text = re.sub(caps + "[.]" + caps + "[.]", "\\1<prd>\\2<prd>", text)
    text = re.sub(" " + suffixes + "[.] " + starters, " \\1<stop> \\2", text)
    text = re.sub(" " + suffixes + "[.]", " \\1<prd>", text)
    text = re.sub(" " + caps + "[.]", " \\1<prd>", text)
    if "”" in text: text = text.replace(".”", "”.")
    if "\"" in text: text = text.replace(".\"", "\".")
    if "!" in text: text = text.replace("!\"", "\"!")
    if "?" in text: text = text.replace("?\"", "\"?")
    text = text.replace(".", ".<stop>")
    text = text.replace("?", "?<stop>")
    text = text.replace("!", "!<stop>")
    text = text.replace("<prd>", ".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences
