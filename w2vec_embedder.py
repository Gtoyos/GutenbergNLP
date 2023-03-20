#This script creates the word embeddings for the books.
#It saves a matrices in the working directory with the respective wordvector embeddings of the test and train books. Please note that it takes a couple of hours to create this embeddings.

from sklearn.model_selection import train_test_split
from gutenberg import Gutenberg
import torch
import string
import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import gensim.downloader as api
import nltk
nltk.download('punkt')
nltk.download('stopwords')
import string
from nltk.tokenize import word_tokenize
from sklearn.cluster import KMeans
from tqdm import tqdm
from gutenberg import Gutenberg
import numpy as np
from collections import Counter
from multiprocessing import Pool
import os
wv=api.load("glove-twitter-25")

g = Gutenberg()
train,test = train_test_split(g.getCatalog(),test_size=0.2,random_state=42)
train,test = Gutenberg(train),Gutenberg(test)

def embed(book,max_paragraphs=100):
    def tk(paragraph):
        tokens = word_tokenize(paragraph, language='english')
        tokens = list(filter(lambda token: token not in string.punctuation
            and token not in ["``",'""',"--","''","**","'s","'d","'ll"]
            and token not in nltk.corpus.stopwords.words('english')
            and token in wv, tokens))
        s = np.zeros(25)
        if(len(tokens)<5):
            return None
        for t in tokens:
            s += wv[t]
        return s/len(tokens)         
    ph = book.split("\n\n")
    ph = list(filter(lambda sentence: len(sentence.split(" "))>10,ph))
    gap = int(len(ph)*0.1)
    ph = ph[gap:min(len(ph),gap+max_paragraphs)]
    x = np.zeros((max_paragraphs,25))
    j=0
    for i in range(len(ph)):
        v = tk(ph[i])
        if(v is None):
            j-=1
            continue
        x[i+j,:] = v
    return x.flatten()

subs = {}
rows = 0
for c in test.getCatalog().columns:
    tmp = test.getCatalog()[test.getCatalog()[c]==1]
    subs[c] = tmp.index
    rows += len(tmp.index)
subs.pop("Title")
w=100
X_test = np.zeros((rows,25*w))
i=0
for label,vectors in subs.items():
    for j in tqdm(range(len(vectors))):
        X_test[i,:] = embed(train.getBook(vectors[j]))
        i+=1
i=0
l=[]
for label,vectors in subs.items():
    l.append(np.full(len(vectors),i))
    i+=1
Y_test = np.concatenate(l)

subs = {}
rows = 0
for c in train.getCatalog().columns:
    tmp = train.getCatalog()[train.getCatalog()[c]==1]
    subs[c] = tmp.index
    rows += len(tmp.index)
subs.pop("Title")
w=100
X = np.zeros((rows,25*w))
i=0
for label,vectors in subs.items():
    for j in tqdm(range(len(vectors))):
        X[i,:] = embed(train.getBook(vectors[j]))
        i+=1
i=0
l=[]
for label,vectors in subs.items():
    l.append(np.full(len(vectors),i))
    i+=1
Y = np.concatenate(l)

np.save("embeddings/X.npy",X)
np.save("embeddings/Y.npy",Y)
np.save("embeddings/X_test.npy",X_test)
np.save("embeddings/Y_test.npy",Y_test)#This script creates the word embeddings for the books.
#It saves a matrices in the working directory with the respective wordvector embeddings of the test and train books. Please note that it takes a couple of hours to create this embeddings.

from sklearn.model_selection import train_test_split
from gutenberg import Gutenberg
import torch
import string
import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import gensim.downloader as api
import nltk
nltk.download('punkt')
nltk.download('stopwords')
import string
from nltk.tokenize import word_tokenize
from sklearn.cluster import KMeans
from tqdm import tqdm
from gutenberg import Gutenberg
import numpy as np
from collections import Counter
from multiprocessing import Pool
import os
wv=api.load("glove-twitter-25")

g = Gutenberg()
train,test = train_test_split(g.getCatalog(),test_size=0.2,random_state=42)
train,test = Gutenberg(train),Gutenberg(test)

def embed(book,max_paragraphs=100):
    def tk(paragraph):
        tokens = word_tokenize(paragraph, language='english')
        tokens = list(filter(lambda token: token not in string.punctuation
            and token not in ["``",'""',"--","''","**","'s","'d","'ll"]
            and token not in nltk.corpus.stopwords.words('english')
            and token in wv, tokens))
        s = np.zeros(25)
        if(len(tokens)<5):
            return None
        for t in tokens:
            s += wv[t]
        return s/len(tokens)         
    ph = book.split("\n\n")
    ph = list(filter(lambda sentence: len(sentence.split(" "))>10,ph))
    gap = int(len(ph)*0.1)
    ph = ph[gap:min(len(ph),gap+max_paragraphs)]
    x = np.zeros((max_paragraphs,25))
    j=0
    for i in range(len(ph)):
        v = tk(ph[i])
        if(v is None):
            j-=1
            continue
        x[i+j,:] = v
    return x.flatten()

subs = {}
rows = 0
for c in test.getCatalog().columns:
    tmp = test.getCatalog()[test.getCatalog()[c]==1]
    subs[c] = tmp.index
    rows += len(tmp.index)
subs.pop("Title")
w=100
X_test = np.zeros((rows,25*w))
i=0
for label,vectors in subs.items():
    for j in tqdm(range(len(vectors))):
        X_test[i,:] = embed(train.getBook(vectors[j]))
        i+=1
i=0
l=[]
for label,vectors in subs.items():
    l.append(np.full(len(vectors),i))
    i+=1
Y_test = np.concatenate(l)

subs = {}
rows = 0
for c in train.getCatalog().columns:
    tmp = train.getCatalog()[train.getCatalog()[c]==1]
    subs[c] = tmp.index
    rows += len(tmp.index)
subs.pop("Title")
w=100
X = np.zeros((rows,25*w))
i=0
for label,vectors in subs.items():
    for j in tqdm(range(len(vectors))):
        X[i,:] = embed(train.getBook(vectors[j]))
        i+=1
i=0
l=[]
for label,vectors in subs.items():
    l.append(np.full(len(vectors),i))
    i+=1
Y = np.concatenate(l)

np.save("embeddings/X.npy",X)
np.save("embeddings/Y.npy",Y)
np.save("embeddings/X_test.npy",X_test)
np.save("embeddings/Y_test.npy",Y_test)