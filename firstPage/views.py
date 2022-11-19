from django.shortcuts import render
from django.http import HttpResponse

import pickle
import pandas as pd

#-------------------------------- START: Model-1 - MNB----------------------------------------------------------
import re
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

model_file_path = './models/model_mnb.pkl'
tfidf_file_path = './models/tfidf_vectorizer_1000.pkl'
pickled_model = pickle.load(open(model_file_path,'rb'))
pickled_tfidf = pickle.load(open(tfidf_file_path,'rb'))

model_mnb_cv_file_path = './models/model_mnb_cv_unibigram.pkl'
cv_file_path = './models/count_vectorizer_unibigram.pkl'
pickled_model_mnb_cv = pickle.load(open(model_mnb_cv_file_path,'rb'))
pickled_cv = pickle.load(open(cv_file_path,'rb'))

def preprocessing(df):
    # Selecting only alphabets:
    df["text"] = df["text"].apply(lambda x: re.sub('[^A-Za-z0-9]+', ' ', x))
    # Making all textual data lowercase:
    df["text"] = df["text"].apply(lambda x: x.lower())
    # Removing all extra white spaces:
    df["text"] = df["text"].apply(lambda x: x.strip())
    return df

# Function to get corpus of X data
def get_corpus(X):
    corpus = []
    ps = PorterStemmer()
    text = X['text']
    text = X['text'][0]
    text = text.split()
    text = [ps.stem(word) for word in text]
    text = " ".join(text)
    corpus.append(text)
    return corpus

def tfidf_vectorizer(corpus):
  tfidf = TfidfVectorizer(max_features=1000, binary=bool)
  X_tfidf = tfidf.fit_transform(corpus).todense()
  return X_tfidf

def test_model(sentence):
    sen = pickled_tfidf.transform([sentence]).toarray()
    res = pickled_model.predict(sen)[0]
    print("test_model : Sentiment -- ", res)
    if res == 1:
        return 'Positive'
    else:
        return 'Negative'

def analyse_sentiment(sentence):
    sen = pickled_cv.transform([sentence]).toarray()
    res = pickled_model_mnb_cv.predict(sen)[0]
    print("test_model : Sentiment -- ", res)
    if res == 1:
        return 'Positive'
    else:
        return 'Negative'
#-------------------------------- END: Model-1 - MNB----------------------------------------------------------

# Create your views here.
def index(request):
    context={}
    temp_label={}
    temp_label["text"]= "It was a good movie :) :)"
    context= {'temp_label':temp_label}
    return render(request,'index.html',context)

def predictSentiment(request):
    if request.method == 'POST':
        entered_review_text = request.POST.get('review_text')
        review_text = entered_review_text
        temp={}
        temp["entered_review_text"]=entered_review_text
        entered_test_data = pd.DataFrame({'X': temp}).transpose()

        entered_test_data2 = entered_test_data.copy()
        entered_test_data2['text'] = entered_test_data['entered_review_text']
        del entered_test_data2['entered_review_text']
        # preprocessing
        X = preprocessing(entered_test_data2)
        corpus = get_corpus(X)
        corpus =  X['text'][0]
        #scoreval = test_model(corpus)
        scoreval = analyse_sentiment(corpus)
    context = {'scoreval':scoreval, 'temp':temp, 'review_text':review_text}
    return render(request,'index.html',context )