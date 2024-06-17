# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.stem import PorterStemmer
import pandas as pd
import numpy as np
import string
import enchant
from tqdm import tqdm
ps = PorterStemmer();
d = enchant.Dict("en-US")
news = pd.read_csv("news.csv")

#%%
"""
    NLP preprocessing
    
    - remove punctuations
    - lowercase
    - remove stopwords
    - remove numbers
    - stem
    - remove meaningless words
    - tokenize
"""
stopwords = stopwords.words('english')
puncs = string.punctuation

rpunc_news = list()

for i in range(news.count()["text"]):
    temp = list()
    for l in news.iloc[i]["text"] :
        if l in puncs:
            temp.append(' ')
        else:
            temp.append(l)
            
    rpunc_news.append(''.join(temp))
    
del(temp)
del(i)
del(l)

processed_news = list()

for n in tqdm(rpunc_news):
    tok_n = word_tokenize(n)
    proc_n = list()
    for w in tok_n:
        if w not in stopwords:
            if d.check(w) and len(w)>1:
                if not w.isnumeric():
                    proc_n.append(ps.stem(w.lower()))
                    
    processed_news.append(proc_n)
    
del(tok_n)
del(n)
del(proc_n)
del(w)

news["Processed text"] = processed_news

del(puncs)
del(stopwords)
del(rpunc_news)
del(processed_news)
del(d)
del(ps)

#%%
#calculate each word count in a news
vectors = list()
for i in tqdm(range(news.count()["Processed text"])):
    vectors.append(FreqDist(news.iloc[i]["Processed text"]))
    
del(i)
#%%
#build a voacbulary of all words and sort alphabetically
vocab = list()
for vector in tqdm(vectors):
    vocab += vector
vocab = list(set(vocab))
vocab.sort()
del(vector)
#%%
#build the frequency matrix
terms_matrix = np.empty((len(vocab),len(vectors)))

for i in tqdm(range(len(vectors))):
    for term in vectors[i]:
        terms_matrix[vocab.index(term)][i] = 1
        
del(i)
del(term)
#%%
#count the number of all news that has each word 
terms_occurence = np.zeros((len(vocab),))
for w_index in tqdm(range(len(vocab))):
    terms_occurence[w_index] = np.sum(terms_matrix[w_index])

del(w_index)
#%%
#delete most general words
treashold = 400
for i in tqdm(range(len(terms_occurence)-1,-1,-1)):
    if terms_occurence[i] > treashold:
        del(vocab[i])
        terms_occurence = np.delete(terms_occurence,i,0)
        terms_matrix = np.delete(terms_matrix,i,0)
del(i)
del(treashold)
#%%
#count coocurrence of all words in all documents
t_terms_matrix = np.transpose(terms_matrix)
terms_co = np.matmul(terms_matrix, t_terms_matrix)
for i in range(terms_co.shape[0]):
    terms_co[i,i] = 0
del(i)
del(t_terms_matrix)

#%%
news_count = news.count()["text"]
def p(w1, w2="", u=1):
    if w2=="":
        w1_count = terms_occurence[vocab.index(w1)]
        w1_tf = (w1_count+0.5)/(news_count+1)
    else:
        w1_ind = vocab.index(w1)
        w2_ind = vocab.index(w2)
        co_count = terms_co[w1_ind, w2_ind]
        co_tf = (co_count+0.25)/(news_count+1)
        return co_tf
    if u==1:
        return w1_tf
    else:
        return (1-w1_tf)

#%%
def simplify_P(pw1, pw2, pw1w2, u, v):
    if u==1 and v==1:
        return pw1w2
    elif u==1 and v==0:
        return (pw1-pw1w2)
    elif u==0 and v==1:
        return (pw2-pw1w2)
    else:
        return (1-(pw1w2 + simplify_P(pw1, pw2, pw1w2, 0, 1) + simplify_P(pw1, pw2, pw1w2, 1, 0)))
#%%
import math

import heapq

def mutual_information(w1):
    w1_mi = dict()
    w1_index = vocab.index(w1)
    pw1 = p(w1)
    w1_vocab = [i for i in range(len(vocab)) if terms_co[w1_index,i] != 0]
    for w2_index in w1_vocab:
        w2 = vocab[w2_index]
        pw2 = p(w2)
        pco = p(w1,w2)
        mi=0
        for u in [0,1]:
            for v in [0,1]:
                puv = simplify_P(pw1, pw2, pco, u, v)
                mi += puv*math.log2( (puv) / (p(w1,u=u)*p(w2,u=u)) )
        if mi>0:
            w1_mi[w2] = mi
        else:
            w1_mi[w2] = 0
        
    return w1_mi


def top_10(d):
    return heapq.nlargest(10, d, key=d.get)
    
#%%
#count top 10 syntagmatic relations with iran word
iran_context = mutual_information("iran")
iran_top10_synt = top_10(iran_context)

#%%
#count top 10 syntagmatic relations with teacher word
teacher_context = mutual_information("teacher")
teacher_top10_synt = top_10(teacher_context)
#%%
#calculate mutual information for each pair of words
mi_matrix = list()
for w in tqdm(vocab):
    mi_matrix.append(mutual_information(w))
del(w)
#%%
#convert list of lists to numpy array
mi_matrix = list(mi_matrix)
#%%
pd.DataFrame(mi_matrix).to_csv("mi.csv")
#%%
mi_df = pd.read_csv('mi.csv')
#%%
mi_m_head = list(mi_df.columns.values)
del(mi_m_head[0])
mi_m = mi_df.to_numpy()
mi_m = np.delete(mi_m,0,axis=1)
mi_m = np.nan_to_num(mi_m)
#%%
from scipy import spatial
def dict_non_zero_keys(d):
    nz_keys = list()
    for k in d.keys():
        if not d[k] == 0:
            nz_keys.append(k)
    return nz_keys

def paradigmatic(w):
    w_para = list()
    for i in tqdm(range(len(vocab))):    
        w_para.append(1-spatial.distance.cosine(mi_m[i], mi_m[vocab.index(w)]))           
    return np.array(w_para)

def topn_numpy_index(arr, n):
    return (-arr).argsort()[:n]
#%%
iran_para = paradigmatic('iran')
iran_top10_para = [ {vocab[t] :iran_para[t]} for t in topn_numpy_index(iran_para, 11)]
#%%
teacher_para = paradigmatic('teacher')
teacher_top10_para = [ {vocab[t] :teacher_para[t]} for t in topn_numpy_index(teacher_para, 11)]
#%%
