# -*- coding: utf-8 -*-

from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
from string import punctuation
from heapq import nlargest
from bs4 import BeautifulSoup
from datetime import datetime
import time
import urllib.request
from many_stop_words import get_stop_words
from collections import Counter

time1 = time.time()

class FrequencySummarizer:
  def __init__(self, min_cut=0.1, max_cut=0.9):
    """
     Initilize the text summarizer.
     Words that have a frequency term lower than min_cut
     or higer than max_cut will be ignored.
    """
    self._min_cut = min_cut
    self._max_cut = max_cut
    self._stopwords = set(get_stop_words('bn'))

  def _compute_frequencies(self, word_sent):
    """
      Compute the frequency of each of word.
      Input:
       word_sent, a list of sentences already tokenized.
      Output:
       freq, a dictionary where freq[w] is the frequency of w.
    """
    freq = defaultdict(int)
    for s in word_sent:
      for word in s:
        if word not in self._stopwords:
          freq[word] += 1
    # frequencies normalization and filtering
    m = float(max(freq.values()))

    for w in list(freq.keys()):
      freq[w] = freq[w]/m
      if freq[w] >= self._max_cut or freq[w] <= self._min_cut:
        del freq[w]
    return freq

  def summarize(self, text, n):
    """
      Return a list of n sentences
      which represent the summary of text.
      print('text = ')
      print(text)
    """
    # Sentences Tokenizer Custom

    stcs = [];
    tmpstr = "";
    words = word_tokenize(text)
    for w in words:
      if (w == "।" or w == "?" or w == ";" or w=='!'):
        tmpstr = tmpstr + w;
        stcs.append(tmpstr);
        tmpstr = "";
      else:
        tmpstr = tmpstr + w + " ";
    sents = stcs

    # Sentences Tokenizer Custom

    assert n <= len(sents)

    word_sent = [word_tokenize(s.lower()) for s in sents]
    self._freq = self._compute_frequencies(word_sent)
    ranking = defaultdict(int)
    for i,sent in enumerate(word_sent):

      for w in sent:
        if w in self._freq:
          ranking[i] += self._freq[w]
    sents_idx = self._rank(ranking, n)

    return [sents[j] for j in sents_idx]

  def _rank(self, ranking, n):
    """ return the first n sentences with highest ranking """
    return nlargest(n, ranking, key=ranking.get)




def get_only_text(url):
 """
  return the title and the text of the article
  at the specified url
 """
 page = urllib.request.urlopen(url).read().decode('utf8')
 soup = BeautifulSoup(page)
 text = ' '.join(map(lambda p: p.text, soup.find_all('p')))
 return soup.title.text, text


"""
feed_xml = urllib.request.urlopen('http://feeds.bbci.co.uk/news/rss.xml').read()
feed = BeautifulSoup(feed_xml.decode('utf8'))
to_summarize = map(lambda p: p.text, feed.find_all('guid'))


for article_url in list(to_summarize):
  title, text = get_only_text(article_url)
  print ('----------------------------------')
  print (title)
  """
fs = FrequencySummarizer()


fi = open('Input5.txt', encoding="utf8")
strss = ""
for line in fi:
    strss = strss + line

for s in (fs.summarize(strss, 10)):
   print (s, end = ' ')

time2 = time.time()
total_time=(time2 - time1)
no_of_docs = 1
ind_time=(total_time / no_of_docs)
print('')
print ("Process ended: ", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
print ("Total time required for ", no_of_docs, " articles to be summarized: ", round(total_time,3) , "seconds")
print ("Average time for each article ",round(ind_time,3)," seconds")
