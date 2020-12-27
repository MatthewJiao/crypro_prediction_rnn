#Flask application to make use of the travel-time prediction model
from flask import Flask
import numpy as np
import tensorflow as tf


app = Flask(__name__)

rnn_btc_model = tf.keras.models.load_model('btc_model')
rnn_ltc_model = tf.keras.models.load_model('ltc_model')
rnn_eth_model = tf.keras.models.load_model('eth_model')
rnn_bch_model = tf.keras.models.load_model('bch_model')


#format for features in sample call: /1,2,3,4,5
@app.route('/btc/<features>')
def index1(features):
    try:
        temp = []
        final = []
        data = features.split(',')
        data = list(map(float, data))

        for i in range(len(data)):
            if (i+1)%8==0:
                temp.append(data[i])
                final.append(temp)
                temp = []
            else: 
                temp.append(data[i])

        #data = [[data[0],data[1],data[2],data[3],data[4]]]
        print(final)
        prediction = rnn_btc_model.predict([final])
        prediction = np.array2string(prediction, precision=2, separator=',',suppress_small=True)
        return prediction
    except:
        return "Error"

@app.route('/ltc/<features>')
def index2(features):
    try:
        temp = []
        final = []
        data = features.split(',')
        data = list(map(float, data))

        for i in range(len(data)):
            if (i+1)%8==0:
                temp.append(data[i])
                final.append(temp)
                temp = []
            else: 
                temp.append(data[i])

        #data = [[data[0],data[1],data[2],data[3],data[4]]]
        print(final)
        prediction = rnn_ltc_model.predict([final])
        prediction = np.array2string(prediction, precision=2, separator=',',suppress_small=True)
        return prediction
    except:
        return "Error"

@app.route('/eth/<features>')
def index3(features):
    try:
        temp = []
        final = []
        data = features.split(',')
        data = list(map(float, data))

        for i in range(len(data)):
            if (i+1)%8==0:
                temp.append(data[i])
                final.append(temp)
                temp = []
            else: 
                temp.append(data[i])

        #data = [[data[0],data[1],data[2],data[3],data[4]]]
        print(final)
        prediction = rnn_eth_model.predict([final])
        prediction = np.array2string(prediction, precision=2, separator=',',suppress_small=True)
        return prediction
    except:
        return "Error"

@app.route('/bch/<features>')
def index4(features):
    try:
        temp = []
        final = []
        data = features.split(',')
        data = list(map(float, data))

        for i in range(len(data)):
            if (i+1)%8==0:
                temp.append(data[i])
                final.append(temp)
                temp = []
            else: 
                temp.append(data[i])

        #data = [[data[0],data[1],data[2],data[3],data[4]]]
        print(final)
        prediction = rnn_bch_model.predict([final])
        prediction = np.array2string(prediction, precision=2, separator=',',suppress_small=True)
        return prediction
    except:
        return "Error"

@app.route('/')
def start():
    return "hello"


if __name__ == "__main__":
    app.run(debug=True)
