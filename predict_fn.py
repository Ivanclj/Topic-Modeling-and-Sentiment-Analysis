import numpy as np
import pandas as pd
import re
import preprocessor as p
import spacy  # For preprocessing
from spacy.lang.en import English
p.set_options(p.OPT.URL, p.OPT.EMOJI)
import pickle
from textblob import TextBlob


nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser']) # disabling Named Entity Recognition for speed

## load count vectorizer
with open('count.pkl', 'rb') as f:
    v = pickle.load(f)
## load tokenizer
with open('lstm_tokenizer.pkl', 'rb') as f:
    tok = pickle.load(f)

with open('id2word.pkl', 'rb') as f:
    id2word = pickle.load(f)
with open('lda_model.pkl', 'rb') as f:
    lda_model = pickle.load(f)



def get_stopwords():
	spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
	spacy_stopwords = list(set(spacy_stopwords))
	neg_stopwords = ['nobody','beside','besides','otherwise','bottom','without','unless','though',
	                  'against','not','nothing','no','n’t','however','very','nevertheless','yet',
	                'cannot','none','less','nowhere','nor','neither', "n't",'over',
	                'least','never','although','except','but']

	return [x for x in spacy_stopwords if x not in neg_stopwords]

stopwords = get_stopwords()	

def clean_text(text):
    text = p.clean(text)
    text = re.sub(r'https?:\/\/.*\/\w*', '', text)
    text = text.replace('realDonaldTrump','')
    text = text.replace('U.S.A','USA')
    text = re.sub('@[^\s]+','',text)
    text = re.sub(r'\&\w*;', '', text)
    text = text.replace('Ûª',"'")
    return re.sub("[^A-Za-z']+", ' ', str(text)).lower()

def tokenizer(doc,stopwords):
    return [token.lemma_ for token in doc if token.text not in stopwords]


def clean_wp(text):
    flag = True
    while flag:
        try:
            text.remove(' ')
        except:
            flag = False
    return text


def get_tweet_sentiment(tweet): 
    ''' 
    Utility function to classify sentiment of passed tweet 
    using textblob's sentiment method 
    '''
    # create TextBlob object of passed tweet text 
    analysis = TextBlob(tweet)
    val = analysis.sentiment.polarity 
    # set sentiment 
    if analysis.sentiment.polarity > 0: 
        return {'positive':val}
#     elif analysis.sentiment.polarity == 0: 
#         return 'neutral'
    else: 
        return {'negative':val}

def preprocess(text):
	a = tokenizer(nlp(clean_text(text)),stopwords)
	return clean_wp(' '.join(a))

def make_features_rf(text):
	return v.transform(text).toarray()


def pad_text(text):
    miss = 19 - len(text)
    if miss > 0 :
        return [0 for i in range(miss)]+text
    elif miss == 0:
        return text
    else:
        return text[:19]
    
def make_features_lstm(text):
	txt = tok.texts_to_sequences(text)
	txt = [pad_text(x) for x in txt]
	return np.asarray(txt)


def make_prediction_rf(feature,model):
    prob = model.predict_proba(feature)
    label = ['negative','positive'] 
    return [{label[0]:x[0],label[1]:x[1]}for x in prob]
     


def make_prediction_lstm(feature,model):
    print(feature)
    print(model.summary())
    prob = model.predict(feature)
    label = ['negative','positive'] 
    return [{label[0]:str(x[0]),label[1]:str(x[1])}for x in prob]

def extract_topic(text):
    text = [tokenizer(nlp(x),stopwords) for x in text]
    bow_vector = [id2word.doc2bow(x) for x in text]
    topic_sort = [sorted(lda_model[x][0], key=lambda tup: -1*tup[1])[0] for x in bow_vector]
    topic_sort = [{'label_prob':str(a[1]),'topic':lda_model.print_topic(a[0],6)} for a in topic_sort]
    return topic_sort
