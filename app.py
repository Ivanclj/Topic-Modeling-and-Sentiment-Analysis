from flask import Flask,request
import logging
import sys
import time
from predict_fn import preprocess,make_features_rf,make_features_lstm, make_prediction_rf,make_prediction_lstm,get_tweet_sentiment,extract_topic
import json
import pickle
import tensorflow as tf
import datetime
from bert_fx import get_bert_features,get_bert_estimator,getPrediction

logging.basicConfig(filename='api_log',
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.ERROR)




app = Flask(__name__)

MODEL_PATH = 'model/'

with open(MODEL_PATH+'RF.pkl', 'rb') as f:
    rf = pickle.load(f)


# model = 'model path'
# graph1 = Graph()
# with graph1.as_default():
#     session1 = Session(graph=graph1)
#     with session1.as_default():
#         model_1.load_model(model) # depends on your model type


@app.route('/')
def index():
    return "Hello, World!"



@app.route('/predict', methods=['GET'])
def run_batch_prediction():
    t1 = datetime.datetime.now()
    args = request.args
    logging.info('args:%s'%args)
    texts = args.get('text', default = None,type = str).split(';')
    model = args.get('model', default = None,type = str)
    
    rst = {}
    i = 1
    if texts is not None and model is not None:
        polar = [get_tweet_sentiment(x) for x in texts]
        text_ls = [preprocess(x) for x in texts]
        topics = extract_topic(text_ls)

        if model == 'rf':
            features = make_features_rf(text_ls)
            pred = make_prediction_rf(features,rf)
        elif model == 'lstm':
            lstm = tf.keras.models.load_model(MODEL_PATH+'LSTM.h5')
            features = make_features_lstm(text_ls)
            pred = make_prediction_lstm(features,lstm)
        elif model == 'bert':
            features = get_bert_features(text_ls)
            estimator = get_bert_estimator(features)
            pred = getPrediction(features,estimator)


        else:
            return 'Error, received %s,%s'%(text_ls,model) 
         
    else:
        return 'Error, received %s,%s'%(text_ls,model)

    for text,prob,pol,topic in zip(texts,pred,polar,topics):
        rst[i] = {'body':text,'pred':prob,'truth':pol,'topic':topic}
        i = i+1
    print('time taken: %s'%(datetime.datetime.now()-t1))
    return rst
    

if __name__ == '__main__':
    app.run(debug=True,host = '0.0.0.0',port = '5000')