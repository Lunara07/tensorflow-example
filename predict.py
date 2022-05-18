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

model = None

def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)        
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

def predict(environ, start_response):
    # Load inputdata from the HTTP request
    request = Request(environ)
    print(request)
    if not request.files:
        return Response('no file uploaded', 400)(environ, start_response)
    test_csv = pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vQD8yYdzF3wGm3E0prjgaTfV6YXHfhs2r1N6xSG9K9Z-uW0bFUahG_dI2XjmAAFDND2OYpaM4ZGZrxP/pub?gid=1714101272&single=true&output=csv')
    csv_file = next(request.files.values())
    df = pd.read_csv(csv_file)
    train_size = int(len(df) * 0.95)
    test_size = len(df) - train_size
    train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]
    scaler = StandardScaler()
    scaler = scaler.fit(train[['Delay']])
    train['Delay'] = scaler.transform(train[['Delay']])
    test['Delay'] = scaler.transform(test[['Delay']])
    test_csv['Delay'] = scaler.transform(test_csv[['Delay']])
    TIME_STEPS = 30

    # reshape to [samples, time_steps, n_features]

    X_train, y_train = create_dataset(train[['Delay']], train.Delay, TIME_STEPS)
    X_test, y_test = create_dataset(test[['Delay']], test.Delay, TIME_STEPS)
    X_test_csv, y_test_csv = create_dataset(test_csv[['Delay']], test_csv.Delay, TIME_STEPS)
    model = keras.Sequential()
    model.add(keras.layers.LSTM(units=64, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(keras.layers.Dropout(rate=0.2))
    model.add(keras.layers.RepeatVector(n=X_train.shape[1]))
    model.add(keras.layers.LSTM(units=64, return_sequences=True))
    model.add(keras.layers.Dropout(rate=0.2))
    model.add(keras.layers.TimeDistributed(keras.layers.Dense(units=X_train.shape[2])))
    model.compile(loss='mae', optimizer='adam')

    model.fit(X_train, y_train,epochs=10,batch_size=32,validation_split=0.1,shuffle=False)

    

    # The predictor must be lazily instantiated;
    # the TensorFlow graph can apparently not be shared
    # between processes.
    #global model
    #if not model:
     #   model = load_model('model.h5')

    #X_test_pred = model.predict(X_test)
    X_test_pred = model.predict(X_test_csv)
    test_mae_loss = np.mean(np.abs(X_test_pred - X_test_csv), axis=1)
    THRESHOLD = 0.65

    test_score_df = pd.DataFrame(index=test_csv[TIME_STEPS:].Time)
    test_score_df['loss'] = test_mae_loss
    test_score_df['threshold'] = THRESHOLD
    test_score_df['anomaly'] = test_score_df.loss > test_score_df.threshold
    test_score_df['delay'] = test[TIME_STEPS:].Delay

    anomalies = test_score_df[test_score_df.anomaly == True]
    
    #pred = pred.where(pred==-1)
    #prediction = predict_image(model, image, inverted)
    #prediction = {'anomalies': where(pred == -1)}
    prediction = test_score_df['anomaly'].tolist()
    integer_map = map(int, prediction)

    integer_prediction = list(integer_map)
    # The following line allows Valohai to track endpoint predictions
    # while the model is deployed. Here we remove the full predictions
    # details as we are only interested in tracking the rest of the results.
    #print(json.dumps({'vh_metadata': {k: v for k, v in prediction.items() if k != 'predictions'}}))

    # Return a JSON response
    response = Response(json.dumps(integer_prediction), content_type='application/json')
    return response(environ, start_response)


# Run a local server for testing with `python deploy.py`
if __name__ == '__main__':
    from werkzeug.serving import run_simple
    run_simple('0.0.0.0', 8000, predict)
