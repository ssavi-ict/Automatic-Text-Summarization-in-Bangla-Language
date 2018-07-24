from many_stop_words import get_stop_words
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter


stopwords = get_stop_words('bn')
print(len(stopwords))
print(stopwords)
wfilter = []
freq = {}
str = "ও আমার দেশের মাটি । তোমার পরে ঠেকাই মাথা । যা হবার হবে । যা হবে তা দেখা যাবে । তিনি কি করছেন তাতে আমার কোন মাথা ব্যাথা নেই । "
words = word_tokenize(str)
print (words)

print ("যে শব্দগুলো stopwords না সেগুলো হচ্ছে - ")

for w in words:
    if w in stopwords and (w!="।"):
        if w in freq:
            freq[w] = freq[w] + 1
        else:
            freq[w] = 1
            wfilter.append(w)

    elif(w!="।"):
        print(w)

#print(wfilter)
print("Stopwords এবং তাদের Frequency গুলো হচ্ছে - ")
for w in wfilter:
    print(w, freq[w])

stcs = [];
tmpstr = "";

for w in words:
    if(w=="।"):
       tmpstr = tmpstr + "।";
       stcs.append(tmpstr) ;
       tmpstr = "";
    else:
        tmpstr = tmpstr + w + " ";

for ln in stcs:
    print(ln);