import json

from PIL import Image
from numpy import where
from werkzeug.wrappers import Request, Response
import pandas as pd
from utils.image import predict_image, process_image
from utils.model import load_model
import pickle
from sklearn.neighbors import LocalOutlierFactor
model = None


def predict(environ, start_response):
    # Load inputdata from the HTTP request
    request = Request(environ)
    print(request)
    if not request.files:
        return Response('no file uploaded', 400)(environ, start_response)
    csv_file = next(request.files.values())
    test = pd.read_csv(csv_file)
    one_hot_encoded_data2 = pd.get_dummies(test, columns = ['Code'])
    df2 = one_hot_encoded_data2[["Delay", "Code_200", "Code_201", "Code_204", "Code_302", "Code_400", "Code_404", "Code_500"]]
    #image, inverted = process_image(Image.open(image_file))

    # The predictor must be lazily instantiated;
    # the TensorFlow graph can apparently not be shared
    # between processes.
    #global model
    #if not model:
    #    with open('model.h5', 'rb') as f:
    #        model = pickle.load(f)
    data = pd.read_csv('http://www.testifytech.ml/Traffic_train.csv')
    one_hot_encoded_data = pd.get_dummies(data, columns = ['Code'])
    df = one_hot_encoded_data[["Delay", "Code_200", "Code_201", "Code_204", "Code_302", "Code_400", "Code_404", "Code_500"]]
    model = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=0.1)
    model.fit(df)
    pred = model.predict(df2)
    #pred = pred.where(pred==-1)
    #prediction = predict_image(model, image, inverted)
    #prediction = {'anomalies': where(pred == -1)}
    prediction = pred.tolist()
    prediction = list(map(lambda x: x.replace(-1, 0), prediction))
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
