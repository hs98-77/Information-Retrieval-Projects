# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 21:41:13 2021

@author: SadOldMan
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.preprocessing import normalize
from sklearn import cluster
import pandas as pd
import numpy as np
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
from tqdm import tqdm
news = pd.read_csv("news.csv")
#%%
#Preprocess news
puncs = string.punctuation
stopwords = stopwords.words('english')
ps = PorterStemmer();
def preprocess(text):
    temp = list()
    for l in text :
        if l in puncs:
            temp.append(' ')
        else:
            temp.append(l)
    temp = ''.join(temp)
    
    t_temp = word_tokenize(temp)
    pp_temp = list()
    for w in t_temp:
        if w not in stopwords:
            if len(w)>1:
                if not w.isnumeric():
                    pp_temp.append(ps.stem(w.lower()))

    return " ".join(pp_temp)        
  
preprocessed_text = list()
for text in tqdm(news['text']):
    preprocessed_text.append(preprocess(text))

del(stopwords)
del(ps)
del(puncs)
del(text)
#%%
vectorizer = TfidfVectorizer(stop_words='english', max_features=3000 )
tfidf_text = vectorizer.fit_transform(preprocessed_text)
ntfidf_text = normalize(tfidf_text)
del(vectorizer)
#%%
#train kmeans(k=8) and fetch the predicted labels
km_model = KMeans(n_clusters=8, max_iter=100, n_init=1)
km_model.fit_predict(ntfidf_text)
km_labels = km_model.labels_

#%%
#calculates purity for a given label set
def purity(y_true, y_pred):
    cm = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(cm, axis=0)) / np.sum(cm) 
#converts predicted and true labels to the same type
def unify_labels(y_true, y_pred, kinds):
    cm = metrics.cluster.contingency_matrix(y_true, y_pred)
    dest_label = list()
    for i in range(cm.shape[1]):
        index = np.argmax(cm[:,i])
        dest_label.append(index)
    return [kinds[dest_label[l]] for l in y_pred]
    
#%%
#calculate the metrics for kmeans
km_purity = purity(news["label"], km_labels)
km_nmi = metrics.normalized_mutual_info_score(news["label"], km_labels)
km_ri = metrics.rand_score(news["label"], km_labels)
km_ul = unify_labels(news["label"], km_labels, ['acq', 'crude', 'earn', 'grain', 'interest', 'money-fx', 'ship', 'trade'])
km_cm = metrics.confusion_matrix(news["label"], km_ul)
km_f1 = metrics.f1_score(news["label"], km_ul, average='macro')
#%%
#calculate count of false negatives and false postivies for each class
fn_and_fp = np.zeros((8,2))
#index 0 is FN
#index 1 is FP
for i in range(8):
    fn_and_fp[i,0] = np.sum(km_cm[i])-km_cm[i,i]
    fn_and_fp[i,1] = np.sum(km_cm[:,i])-km_cm[i,i]
#%%
#train single link and fetch the predicted labels
sl_model = cluster.AgglomerativeClustering(n_clusters=8, linkage='single')
sl_model.fit(ntfidf_text.toarray())
sl_labels = sl_model.labels_
#%%
#train average link and fetch the predicted labels
al_model = cluster.AgglomerativeClustering(n_clusters=8, linkage='average')
al_model.fit(ntfidf_text.toarray())
al_labels = al_model.labels_
#%%
#train complete link and fetch the predicted labels
cl_model = cluster.AgglomerativeClustering(n_clusters=8, linkage='complete')
cl_model.fit(ntfidf_text.toarray())
cl_labels = cl_model.labels_
#%%
#calculate the metrics for single link
sl_purity = purity(news["label"], sl_labels)
sl_nmi = metrics.normalized_mutual_info_score(news["label"], sl_labels)
sl_ri = metrics.rand_score(news["label"], sl_labels)
sl_ul = unify_labels(news["label"], sl_labels, ['acq', 'crude', 'earn', 'grain', 'interest', 'money-fx', 'ship', 'trade'])
sl_f1 = metrics.f1_score(news["label"], sl_ul, average='macro')
#%%
#calculate the metrics for average link
al_purity = purity(news["label"], al_labels)
al_nmi = metrics.normalized_mutual_info_score(news["label"], al_labels)
al_ri = metrics.rand_score(news["label"], al_labels)
al_ul = unify_labels(news["label"], al_labels, ['acq', 'crude', 'earn', 'grain', 'interest', 'money-fx', 'ship', 'trade'])
al_f1 = metrics.f1_score(news["label"], al_ul, average='macro')
#%%
#calculate the metrics for complete link
cl_purity = purity(news["label"], cl_labels)
cl_nmi = metrics.normalized_mutual_info_score(news["label"], cl_labels)
cl_ri = metrics.rand_score(news["label"], cl_labels)
cl_ul = unify_labels(news["label"], cl_labels, ['acq', 'crude', 'earn', 'grain', 'interest', 'money-fx', 'ship', 'trade'])
cl_f1 = metrics.f1_score(news["label"], cl_ul, average='macro')
#%%