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
    csv_file = next(request.files.values())
    df = pd.read_csv(csv_file)
    train_size = int(len(df) * 0.95)
    test_size = len(df) - train_size
    train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]
    scaler = StandardScaler()
    scaler = scaler.fit(train[['Delay']])
    TIME_STEPS = 30

    # reshape to [samples, time_steps, n_features]
    
    X_test, y_test = create_dataset(test[['Delay']], test.Delay, TIME_STEPS)

    

    

    # The predictor must be lazily instantiated;
    # the TensorFlow graph can apparently not be shared
    # between processes.
    global model
    if not model:
        model = load_model('model.h5')

    X_test_pred = model.predict(X_test)

    test_mae_loss = np.mean(np.abs(X_test_pred - X_test), axis=1)
    THRESHOLD = 0.65

    test_score_df = pd.DataFrame(index=test[TIME_STEPS:].Time)
    test_score_df['loss'] = test_mae_loss
    test_score_df['threshold'] = THRESHOLD
    test_score_df['anomaly'] = test_score_df.loss > test_score_df.threshold
    test_score_df['delay'] = test[TIME_STEPS:].Delay

    anomalies = test_score_df[test_score_df.anomaly == True]
    
    #pred = pred.where(pred==-1)
    #prediction = predict_image(model, image, inverted)
    #prediction = {'anomalies': where(pred == -1)}
    prediction = anomalies['anomaly'].tolist()
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
