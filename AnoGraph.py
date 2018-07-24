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

# Parameters to the algorithm. Currently set to values that was reported
# in the paper to produce "best" results.
time1=time.time()
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
            semvec[i] = max_sim  if max_sim > PHI else 0.0
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
#input1='Bangladesh is a developing country. Agriculture is main culture of it. It earns more money by exporting jute and prawn.',time=10sec
"""#output1=Automatic text summarization is a text- mining task that extracts essential sentences to cover almost all the concepts of a document.
It is to reduce users’ consuming time in document reading without losing the general issues for users’ comprehension.
With document summary available, users can easily decide its relevancy to their interests and acquire desired documents with much less mental loads involved.


ouput2=The region witnessed the Bengali Language Movement in 1952 and the Bangladesh Liberation War in 1971.
After independence was achieved, a parliamentary republic was established.
A presidential government was in place between 1975 and 1990, followed by a return to parliamentary democracy.
The country continues to face challenges in the areas of poverty, education, healthcare and corruption.Bangladesh is a middle power and a developing nation.
Listed as one of the Next Eleven, its economy ranks 46th in terms of nominal gross domestic product (GDP) and 29th in terms of purchasing power parity (PPP).
It is one of the largest textile exporters in the world.
Its major trading partners are the European Union, the United States, China, India, Japan, Malaysia and Singapore.
With its strategically vital location between Southern, Eastern and Southeast Asia, Bangladesh is an important promoter of regional connectivity and cooperation.
It is a founding member of SAARC, BIMSTEC, the Bangladesh-China-India-Myanmar Forum for Regional Cooperation and the Bangladesh Bhutan India Nepal Initiative.
It is also a member of the Commonwealth of Nations, the Developing 8 Countries, the OIC, the Non Aligned Movement, the Group of 77 and the World Trade Organization.
Bangladesh is one of the largest contributors to United Nations peacekeeping forces.
Time : 1090.401 sec
"""

#stress=' The country of Bengal officially the People Republic of Bangladesh  is a country in South Asia. It shares land borders with India and Myanmar (Burma). Nepal, Bhutan and China are located near Bangladesh but do not share a border with it. The country maritime territory in the Bay of Bengal is roughly equal to the size of its land area.[11] Bangladesh is the worlds eighth most populous country. Dhaka is its capital and largest city, followed by Chittagong, which has the country largest port. Bangladesh forms the largest and easternmost part of the Bengal region.[12] Bangladeshis include people from a range of ethnic groups and religions. Bengalis, who speak the official Bengali language, make up 98% of the population.[2][3] The politically dominant Bengali Muslims make the nation the world third largest Muslim-majority country. Most of Bangladesh is covered by the Bengal delta, the largest delta on Earth. The country has 700 rivers and 8,046 km (5,000 miles) of inland waterways. Highlands with evergreen forests are found in the northeastern and southeastern regions of the country. Bangladesh has many islands and a coral reef. The longest unbroken sea beach, Cox Bazar Beach is located here. It is home to the Sundarbans, the largest mangrove forest in the world. The country biodiversity includes a vast array of plant and wildlife, including endangered Bengal tigers, the national animal. The Greeks and Romans identified the region as Gangaridai, a powerful kingdom ofthe historical subcontinent, in the 3rd century BCE. Archaeological research has unearthed several ancient cities in Bangladesh, which enjoyed international trade links for millennia.[13] The Bengal Sultanate and Mughal Bengal transformed the region into a cosmopolitan Islamic imperial power between the 14th and 18th centuries. The region was home to many principalities that made use of their inland naval prowess.[14][15] It was also a notable center of the global muslin and silk trade. As part of British India, the region was influenced by the Bengali renaissance and played an important role in anti-colonial movements. The Partition of British India made East Bengal a part of the Dominion of Pakistan; and renamed it as East Pakistan. The region witnessed the Bengali Language Movement in 1952 and the Bangladesh Liberation War in 1971. After independence was achieved, a parliamentary republic was established. A presidential government was in place between 1975 and 1990, followed by a return to parliamentary democracy. The country continues to face challenges in the areas of poverty, education, healthcare and corruption.Bangladesh is a middle power and a developing nation. Listed as one of the Next Eleven, its economy ranks 46th in terms of nominal gross domestic product (GDP) and 29th in terms of purchasing power parity (PPP). It is one of the largest textile exporters in the world. Its major trading partners are the European Union, the United States, China, India, Japan, Malaysia and Singapore. With its strategically vital location between Southern, Eastern and Southeast Asia, Bangladesh is an important promoter of regional connectivity and cooperation. It is a founding member of SAARC, BIMSTEC, the Bangladesh-China-India-Myanmar Forum for Regional Cooperation and the Bangladesh Bhutan India Nepal Initiative. It is also a member of the Commonwealth of Nations, the Developing 8 Countries, the OIC, the Non Aligned Movement, the Group of 77 and the World Trade Organization. Bangladesh is one of the largest contributors to United Nations peacekeeping forces.'
stress = "I Love Bangladesh. Cz it's my country. And peoples are awesome. and the animals are also. Scenic beauty is too good."
sents=sent_tokenize(stress)
l=len(sents)
print(l)
tot={}
ok={}
for i in range(l):
    tot[i]=0
    ok[i]=0


for i in range(l):
    for j in range(l):
        if j>i:#here condition is for reduce over calculation
            a=similarity(sents[i],sents[j],True)
           # b=similarity(sents[i],sents[j],False)
            b=0
            d=(a+b)
            #print(d);
            #total similarity of first sentence with second sentence
            tot[i]=tot[i]+d

            #to store same result i.e. total similarity of second sentence with first sentence
            tot[j]=tot[j]+d

lst=nlargest((int)(l/3),tot)

for i in lst:
    ok[i]=1
for i in range(l):
    if ok[i]==1:
        print(sents[i])

time2=time.time()
print(round(time2-time1,3))