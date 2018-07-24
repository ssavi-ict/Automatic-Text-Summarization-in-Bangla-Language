from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from many_stop_words import get_stop_words

str = "পাইথন খুব ভাল একটা ভাষা ।"

words = word_tokenize(str)
print(words)
print(get_stop_words('bn'))