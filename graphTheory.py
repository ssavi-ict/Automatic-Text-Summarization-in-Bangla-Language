# -*- coding: utf-8 -*-

# author : Sujit Pal

# Note: this is a python3 updated version of http://sujitpal.blogspot.fr/2014/12/semantic-similarity-for-short-sentences.html
# by mathieu Chrétien (mchretien.pro@mail.com)

# contributor : Mathieu Chrétien
#from nltk.tokenize import sent_tokenize,word_tokenize


from __future__ import division
import nltk
from nltk.tokenize import sent_tokenize,word_tokenize
from heapq import nlargest

from nltk.corpus import wordnet as wn
from nltk.corpus import brown
import math
import numpy as np
import sys
import time
from datetime import datetime

time1 = time.time()

# Parameters to the algorithm. Currently set to values that was reported
# in the paper to produce "best" results.
ALPHA = 0.2
BETA = 0.45
ETA = 0.4
PHI = 0.2
DELTA = 0.85

brown_freqs = dict()
N = 0


######################### word similarity ##########################

def get_best_synset_pair(word_1, word_2):
    """
    Choose the pair with highest path similarity among all pairs.
    Mimics pattern-seeking behavior of humans.
    """
    max_sim = -1.0
    synsets_1 = wn.synsets(word_1)
    synsets_2 = wn.synsets(word_2)
    if len(synsets_1) == 0 or len(synsets_2) == 0:
        return None, None
    else:
        max_sim = -1.0
        best_pair = None, None
        for synset_1 in synsets_1:
            for synset_2 in synsets_2:
                sim = wn.path_similarity(synset_1, synset_2)
                if sim == None:
                    sim = 0
                if sim > max_sim:
                    max_sim = sim
                    best_pair = synset_1, synset_2
        return best_pair


def length_dist(synset_1, synset_2):
    """
    Return a measure of the length of the shortest path in the semantic
    ontology (Wordnet in our case as well as the paper's) between two
    synsets.
    """
    l_dist = sys.maxsize
    if synset_1 is None or synset_2 is None:
        return 0.0
    if synset_1 == synset_2:
        # if synset_1 and synset_2 are the same synset return 0
        l_dist = 0.0
    else:
        wset_1 = set([str(x.name()) for x in synset_1.lemmas()])
        wset_2 = set([str(x.name()) for x in synset_2.lemmas()])
        if len(wset_1.intersection(wset_2)) > 0:
            # if synset_1 != synset_2 but there is word overlap, return 1.0
            l_dist = 1.0
        else:
            # just compute the shortest path between the two
            l_dist = synset_1.shortest_path_distance(synset_2)
            if l_dist is None:
                l_dist = 0.0
    # normalize path length to the range [0,1]
    return math.exp(-ALPHA * l_dist)


def hierarchy_dist(synset_1, synset_2):
    """
    Return a measure of depth in the ontology to model the fact that
    nodes closer to the root are broader and have less semantic similarity
    than nodes further away from the root.
    """
    h_dist = sys.maxsize
    if synset_1 is None or synset_2 is None:
        return h_dist
    if synset_1 == synset_2:
        # return the depth of one of synset_1 or synset_2
        h_dist = max([x[1] for x in synset_1.hypernym_distances()])
    else:
        # find the max depth of least common subsumer
        hypernyms_1 = {x[0]: x[1] for x in synset_1.hypernym_distances()}
        hypernyms_2 = {x[0]: x[1] for x in synset_2.hypernym_distances()}
        lcs_candidates = set(hypernyms_1.keys()).intersection(
            set(hypernyms_2.keys()))
        if len(lcs_candidates) > 0:
            lcs_dists = []
            for lcs_candidate in lcs_candidates:
                lcs_d1 = 0
                if lcs_candidate in hypernyms_1:
                    lcs_d1 = hypernyms_1[lcs_candidate]
                lcs_d2 = 0
                if lcs_candidate in hypernyms_2:
                    lcs_d2 = hypernyms_2[lcs_candidate]
                lcs_dists.append(max([lcs_d1, lcs_d2]))
            h_dist = max(lcs_dists)
        else:
            h_dist = 0
    return ((math.exp(BETA * h_dist) - math.exp(-BETA * h_dist)) /
            (math.exp(BETA * h_dist) + math.exp(-BETA * h_dist)))


def word_similarity(word_1, word_2):
    synset_pair = get_best_synset_pair(word_1, word_2)
    return (length_dist(synset_pair[0], synset_pair[1]) *
            hierarchy_dist(synset_pair[0], synset_pair[1]))


######################### sentence similarity ##########################

def most_similar_word(word, word_set):
    """
    Find the word in the joint word set that is most similar to the word
    passed in. We use the algorithm above to compute word similarity between
    the word and each word in the joint word set, and return the most similar
    word and the actual similarity value.
    """
    max_sim = -1.0
    sim_word = ""
    for ref_word in word_set:
        sim = word_similarity(word, ref_word)
        if sim > max_sim:
            max_sim = sim
            sim_word = ref_word
    return sim_word, max_sim


def info_content(lookup_word):
    """
    Uses the Brown corpus available in NLTK to calculate a Laplace
    smoothed frequency distribution of words, then uses this information
    to compute the information content of the lookup_word.
    """
    global N
    if N == 0:
        # poor man's lazy evaluation
        for sent in brown.sents():
            for word in sent:
                word = word.lower()
                if word not in brown_freqs:
                    brown_freqs[word] = 0
                brown_freqs[word] = brown_freqs[word] + 1
                N = N + 1
    lookup_word = lookup_word.lower()
    n = 0 if lookup_word not in brown_freqs else brown_freqs[lookup_word]
    return 1.0 - (math.log(n + 1) / math.log(N + 1))


def semantic_vector(words, joint_words, info_content_norm):
    """
    Computes the semantic vector of a sentence. The sentence is passed in as
    a collection of words. The size of the semantic vector is the same as the
    size of the joint word set. The elements are 1 if a word in the sentence
    already exists in the joint word set, or the similarity of the word to the
    most similar word in the joint word set if it doesn't. Both values are
    further normalized by the word's (and similar word's) information content
    if info_content_norm is True.
    """
    sent_set = set(words)
    semvec = np.zeros(len(joint_words))
    i = 0
    for joint_word in joint_words:
        if joint_word in sent_set:
            # if word in union exists in the sentence, s(i) = 1 (unnormalized)
            semvec[i] = 1.0
            if info_content_norm:
                semvec[i] = semvec[i] * math.pow(info_content(joint_word), 2)
        else:
            # find the most similar word in the joint set and set the sim value
            sim_word, max_sim = most_similar_word(joint_word, sent_set)
            semvec[i] = PHI if max_sim > PHI else 0.0
            if info_content_norm:
                semvec[i] = semvec[i] * info_content(joint_word) * info_content(sim_word)
        i = i + 1
    return semvec


def semantic_similarity(sentence_1, sentence_2, info_content_norm):
    """
    Computes the semantic similarity between two sentences as the cosine
    similarity between the semantic vectors computed for each sentence.
    """
    words_1 = nltk.word_tokenize(sentence_1)
    words_2 = nltk.word_tokenize(sentence_2)
    joint_words = set(words_1).union(set(words_2))
    vec_1 = semantic_vector(words_1, joint_words, info_content_norm)
    vec_2 = semantic_vector(words_2, joint_words, info_content_norm)
    return np.dot(vec_1, vec_2.T) / (np.linalg.norm(vec_1) * np.linalg.norm(vec_2))


######################### word order similarity ##########################

def word_order_vector(words, joint_words, windex):
    """
    Computes the word order vector for a sentence. The sentence is passed
    in as a collection of words. The size of the word order vector is the
    same as the size of the joint word set. The elements of the word order
    vector are the position mapping (from the windex dictionary) of the
    word in the joint set if the word exists in the sentence. If the word
    does not exist in the sentence, then the value of the element is the
    position of the most similar word in the sentence as long as the similarity
    is above the threshold ETA.
    """
    wovec = np.zeros(len(joint_words))
    i = 0
    wordset = set(words)
    for joint_word in joint_words:
        if joint_word in wordset:
            # word in joint_words found in sentence, just populate the index
            wovec[i] = windex[joint_word]
        else:
            # word not in joint_words, find most similar word and populate
            # word_vector with the thresholded similarity
            sim_word, max_sim = most_similar_word(joint_word, wordset)
            if max_sim > ETA:
                wovec[i] = windex[sim_word]
            else:
                wovec[i] = 0
        i = i + 1
    return wovec


def word_order_similarity(sentence_1, sentence_2):
    """
    Computes the word-order similarity between two sentences as the normalized
    difference of word order between the two sentences.
    """
    words_1 = nltk.word_tokenize(sentence_1)
    words_2 = nltk.word_tokenize(sentence_2)
    joint_words = list(set(words_1).union(set(words_2)))
    windex = {x[1]: x[0] for x in enumerate(joint_words)}
    r1 = word_order_vector(words_1, joint_words, windex)
    r2 = word_order_vector(words_2, joint_words, windex)
    return 1.0 - (np.linalg.norm(r1 - r2) / np.linalg.norm(r1 + r2))


######################### overall similarity ##########################

def similarity(sentence_1, sentence_2, info_content_norm):
    """
    Calculate the semantic similarity between two sentences. The last
    parameter is True or False depending on whether information content
    normalization is desired or not.
    """
    return DELTA * semantic_similarity(sentence_1, sentence_2, info_content_norm) + \
           (1.0 - DELTA) * word_order_similarity(sentence_1, sentence_2)
"""
sentence_pairs = [
    ["I like that bachelor.", "I like that bachelor.", 1.0],
    ["John is very nice.", "Is John very nice?", 0.977],
    ["Red alcoholic drink.", "A bottle of wine.", 0.585],
    ["Red alcoholic drink.", "Fresh orange juice.", 0.611],
    ["Red alcoholic drink.", "An English dictionary.", 0.0],
    ["Red alcoholic drink.", "Fresh apple juice.", 0.420],
    ["A glass of cider.", "A full cup of apple juice.", 0.678],
    ["It is a dog.", "That must be your dog.", 0.739],
    ["It is a dog.", "It is a log.", 0.623],
    ["It is a dog.", "It is a pig.", 0.790],
    ["Dogs are animals.", "They are common pets.", 0.738],
    ["Canis familiaris are animals.", "Dogs are common pets.", 0.362],
    ["I have a pen.", "Where do you live?", 0.0],
    ["I have a pen.", "Where is ink?", 0.129],
    ["I have a hammer.", "Take some nails.", 0.508],
    ["I have a hammer.", "Take some apples.", 0.121]
]
for sent_pair in sentence_pairs:
    print ("%s\t%s\t%.3f\t%.3f\t%.3f" % (sent_pair[0], sent_pair[1], sent_pair[2],
        similarity(sent_pair[0], sent_pair[1], False),
        similarity(sent_pair[0], sent_pair[1], True)))

"""

fi = open('Input5.txt', encoding="utf8")
strss = ""
for line in fi:
    strss = strss + line
#print(strss)

class my_struct():
    def __init__(self, i, j, sim):
        self.i = i
        self.j = j
        self.sim = sim

sents=sent_tokenize(strss)
#print(sents)
l=len(sents)
#print(l)

sim = []
for i in range(l):
    for j in range(l):
        if (j>i):
            cal = similarity(sents[i], sents[j], False)
            sim.append(my_struct(i, j, cal))
            #print("%d %d %.3f" %(i, j, cal))
            #print("%s\t%s\t%.3f\t%.3f" % (sents[i], sents[j], similarity(sents[i], sents[j], False), similarity(sents[i], sents[j], True)))

lsm = len(sim)
sim.sort(key=lambda x: x.sim, reverse=True)

#for i in range(lsm):
        #print("%d\t%d\t%.3f" % (sim[i].i, sim[i].j, sim[i].sim))

hmlines = int(input(print("How many lines you want as summary from " + str(l) + " lines ")))

my_nodup_list = []

tot_line_printed = 0;
if(hmlines>=len(set(sents))):
    print(strss)
else:
    for i in range(lsm):
        if sents[sim[i].i] not in my_nodup_list:
            print(sents[sim[i].i], end = " ")
            my_nodup_list.append(sents[sim[i].i])
            tot_line_printed = tot_line_printed + 1

        if(tot_line_printed==hmlines):
            break

        if sents[sim[i].j] not in my_nodup_list:
            print(sents[sim[i].j], end = " ")
            my_nodup_list.append(sents[sim[i].j])
            tot_line_printed = tot_line_printed + 1

        if (tot_line_printed == hmlines):
            break

time2 = time.time()
total_time=(time2 - time1)
no_of_docs = 1
ind_time=(total_time / no_of_docs)
print('')
print ("Process ended: ", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
print ("Total time required for ", no_of_docs, " articles to be summarized: ", round(total_time,3) , "seconds")
print ("Average time for each article ",round(ind_time,3)," seconds")
