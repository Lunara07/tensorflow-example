import json

from PIL import Image
from numpy import where
from werkzeug.wrappers import Request, Response
import pandas as pd
from utils.image import predict_image, process_image
from utils.model import load_model
import pickle
from tensorflow import keras
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import RandomOverSampler
model = None



def predict(environ, start_response):
    # Load inputdata from the HTTP request
    request = Request(environ)
    print(request)
    if not request.files:
        return Response('no file uploaded', 400)(environ, start_response)
    csv_file = next(request.files.values())
    test = pd.read_csv(csv_file)
    
    train = pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vSEhmFQ0wZgYM7Q_aETO8dPwA0eqShcQ2v5ps8oBZJYzgUR7fg1pS5hX8RdVPsb3u7tHmwTahl-nYdG/pub?gid=733135457&single=true&output=csv')


    one_hot_encoded_train = pd.get_dummies(train, columns = ['Code'])
    one_hot_encoded_train = pd.get_dummies(one_hot_encoded_train, columns = ['Method'])
    
    train = one_hot_encoded_train.drop('Time', 1)
    y_train=train[["Y"]]
    X_train=train.drop("Y", 1)
    y_train.loc[y_train["Y"] == -1, "Y"] = 0
    ros = RandomOverSampler(random_state=0)
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)
    model = KNeighborsClassifier( weights='distance')

    model.fit(X_train_resampled, y_train_resampled['Y'])
    
    one_hot_encoded_test = pd.get_dummies(test, columns = ['Code'])
    one_hot_encoded_test = pd.get_dummies(one_hot_encoded_test, columns = ['Method'])
    test = one_hot_encoded_test.drop('Time', 1)
    y_test=test[["Y"]]
    X_test=test.drop("Y", 1)
    y_test.loc[y_test["Y"] == -1, "Y"] = 0
    

    

    # The predictor must be lazily instantiated;
    # the TensorFlow graph can apparently not be shared
    # between processes.
    #global model
    #if not model:
        #model = load_model('model.h5')
     #   with open('model.h5', 'rb') as f:
      #      model = pickle.load(f)

    pred = model.predict(X_test)

    
    
    #pred = pred.where(pred==-1)
    #prediction = predict_image(model, image, inverted)
    #prediction = {'anomalies': where(pred == -1)}
    prediction = pred.tolist()
    # The following line allows Valohai to track endpoint predictions
    # while the model is deployed. Here we remove the full predictions
    # details as we are only interested in tracking the rest of the results.
    #print(json.dumps({'vh_metadata': {k: v for k, v in prediction.items() if k != 'predictions'}}))

    # Return a JSON response
    response = Response(json.dumps(prediction), content_type='application/json')
    return response(environ, start_response)


# Run a local server for testing with `python deploy.py`
if __name__ == '__main__':
    from werkzeug.serving import run_simple
    run_simple('0.0.0.0', 8000, predict)
