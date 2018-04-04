
# coding: utf-8

# In[1]:


import numpy as np
import random as rnd

from scipy.sparse import csr_matrix
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from pprint import pprint


# In[2]:


# Load train and test data set with class labels 
train_Xy = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
test_Xy = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))


# In[3]:


# Convert all text data into tf-idf vectors
vectorizer = TfidfVectorizer(stop_words='english', min_df=3, max_df=0.9)
train_vec = vectorizer.fit_transform(train_Xy.data)
test_vec = vectorizer.transform(test_Xy.data)


# In[4]:


# Divide train data set into labeled and unlabeled data sets
n_train_data = train_vec.shape[0]
split_ratio = 0.2 # labeled vs unlabeled
X_l, X_u, y_l, y_u = train_test_split(train_vec, train_Xy.target, train_size=split_ratio)
print (X_l.shape, X_u.shape)


# In[6]:


# Train Naive Bayes classifier (imported) 
# using both labeled and unlabeled data set
clf = MultinomialNB(alpha=1e-8)
clf.fit(X_l, y_l)
# clf.fit(train_vec, train_Xy.target)


# In[7]:


# Evaluate NB classifier using test data set
pred = clf.predict(test_vec)
print(metrics.classification_report(test_Xy.target, pred, target_names=test_Xy.target_names))


# In[8]:


pprint(metrics.confusion_matrix(test_Xy.target, pred))


# In[9]:


print(metrics.accuracy_score(test_Xy.target, pred))


# In[10]:


# from scipy.linalg import get_blas_funcs
# b_w_d = (X_u > 0).T.toarray()
# lp_w_c = (clf.feature_log_prob_)
# lp_d_c = get_blas_funcs("gemm", (lp_w_c, b_w_d))
# print type(lp_w_c), type(b_w_d), type(lp_d_c)
# # lp_d_c(alpha=1.0, a=lp_w_c, b=b_w_d.T, trans_a=True, trans_b=True)
# lp_d_c(alpha=1.0, a=lp_w_c, b=b_w_d).shape


# In[11]:


# find the most informative features 
import numpy as np
def show_topK(classifier, vectorizer, categories, K=10):
    feature_names = np.asarray(vectorizer.get_feature_names())
    for i, category in enumerate(categories):
        topK = np.argsort(classifier.coef_[i])[-K:]
        print("%s: %s" % (category, " ".join(feature_names[topK])))


# In[12]:


show_topK(clf, vectorizer, train_Xy.target_names, K=20)

