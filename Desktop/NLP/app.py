from flask import Flask, render_template, request
import nltk
import os
import re
from nltk.corpus import shakespeare, stopwords
from nltk.probability import FreqDist
from nltk.util import bigrams, trigrams, ngrams

app = Flask(__name__)

if __name__ == '__main__':
    app.debug = True
    app.run()

def get_trigram_freq(tokens):
    tgs = list(nltk.trigrams(tokens))
    a,b,c = list(zip(*tgs))
    bgs = list(zip(a,b))
    
    return nltk.ConditionalFreqDist(list(zip(bgs, c)))

def get_bigram_freq(tokens):
    bgs = list(nltk.bigrams(tokens))

    return nltk.ConditionalFreqDist(bgs)

shakespeare.fileids()
tokens = shakespeare.words('hamlet.xml') + shakespeare.words('dream.xml') + shakespeare.words('r_and_j.xml')

punctuation = re.compile(r"[-.?!',:;()|0-9]")
stopwords.words('english')
post_punctuation = []
for words in tokens:
    word = punctuation.sub("", words)
    if len(word) > 0:
        post_punctuation.append(word)

NLPDoc_tokens = []
for word in post_punctuation:
    if word not in stopwords.words('english'):
        NLPDoc_tokens.append(word)
for i in range(len(NLPDoc_tokens)):
    NLPDoc_tokens[i] = NLPDoc_tokens[i].lower()

bgs_freq = get_bigram_freq(NLPDoc_tokens)
tgs_freq = get_trigram_freq(NLPDoc_tokens)

@app.route('/')
def index():
    return render_template("index.html")


@app.route('/', methods=["POST"])
def result():
    text = str(request.form["text"])
    text.lower()
    words = text.split()
    n = len(words)
    if n==1:
        result = bgs_freq[(text)].most_common(5)
    if n>1:
        result = tgs_freq[(words[n-2],words[n-1])].most_common(10)
    
    return render_template("index.html", result=result)
