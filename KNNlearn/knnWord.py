# -*- coding:utf-8 -*-

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import os
import scipy as sp
import sys
import nltk.stem

englist_stemmer = nltk.stem.SnowballStemmer("english")
class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer,self).build_analyzer()
        return lambda doc:(englist_stemmer.stem(w) for w in analyzer(doc))

class StemmedTifdVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedTifdVectorizer,self).build_analyzer()
        return lambda doc:(englist_stemmer.stem(w) for w in analyzer(doc))

def dist_raw(v1,v2):
    delta = v1-v2
    return sp.linalg.norm(delta.toarray())

def dist_norm(v1,v2):
    v1_norm = v1/sp.linalg.norm(v1.toarray())
    v2_norm = v2/sp.linalg.norm(v2.toarray())
    delta = v1_norm - v2_norm
    return sp.linalg.norm(delta.toarray())

vectorizer = StemmedTifdVectorizer(min_df=1,stop_words="english")
# content = ['how to format my disk',"hard disk format problems"]
# X= vectorizer.fit_transform(content)
# print(vectorizer.get_feature_names())
# print(X.toarray().transpose())
Dir = "../data1/toy"
posts = [open(os.path.join(Dir,f)).read() for f in os.listdir(Dir)]
X_train = vectorizer.fit_transform(posts)
num_samples,num_features = X_train.shape
print("#samples:%d,#features:%d"%(num_samples,num_features))
print(vectorizer.get_feature_names())
new_post = "imaging databases"
new_post_vec = vectorizer.transform([new_post])
print(new_post_vec)
print(new_post_vec.toarray())
best_doc = None
best_dict = sys.maxsize
best_i = None
for i in range(0,num_samples):
    post = posts[i]
    if post == new_post:
        continue
    post_vec = X_train.getrow(i)
    d = dist_norm(post_vec,new_post_vec)
    print("*** post %i with dict=%.2f :%s"%(i,d,post))
    if d<best_dict:
        best_dict = d
        best_i = i
print("best post is %i with dist=%.2f"%(best_i,best_dict))