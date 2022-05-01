import uuid
import pickle


import numpy as np
import tensorflow as tf
import valohai
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor

from tensorflow import keras
from sklearn.preprocessing import StandardScaler

# def log_metadata(epoch, logs):
#     """Helper function to log training metrics"""
#     with valohai.logger() as logger:
#         logger.log('epoch', epoch)
#         logger.log('accuracy', logs['accuracy'])
#         logger.log('loss', logs['loss'])

def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)        
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)
def main():

    valohai.prepare(
        step='train-model',
        image='tensorflow/tensorflow:2.6.0',
        default_inputs={
            'dataset': 'https://drive.google.com/file/d/1grwqu7MHB5CpiV9LmS9U8T53RQmV5Yqf/view?usp=sharing',
        }
    )

    RANDOM_SEED = 42

    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    # Read input files from Valohai inputs directory
    # This enables Valohai to version your training data
    # and cache the data for quick experimentation
    #print(valohai.inputs('dataset'))
    df = pd.read_csv(valohai.inputs('dataset').path())
    train_size = int(len(df) * 0.95)
    test_size = len(df) - train_size
    train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]
    

    scaler = StandardScaler()
    scaler = scaler.fit(train[['Delay']])

    train['Delay'] = scaler.transform(train[['Delay']])
    test['Delay'] = scaler.transform(test[['Delay']])
    TIME_STEPS = 30

    # reshape to [samples, time_steps, n_features]

    X_train, y_train = create_dataset(train[['Delay']], train.Delay, TIME_STEPS)
    X_test, y_test = create_dataset(test[['Delay']], test.Delay, TIME_STEPS)

    model = keras.Sequential()
    model.add(keras.layers.LSTM(units=64, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(keras.layers.Dropout(rate=0.2))
    model.add(keras.layers.RepeatVector(n=X_train.shape[1]))
    model.add(keras.layers.LSTM(units=64, return_sequences=True))
    model.add(keras.layers.Dropout(rate=0.2))
    model.add(keras.layers.TimeDistributed(keras.layers.Dense(units=X_train.shape[2])))
    model.compile(loss='mae', optimizer='adam')

    model.fit(X_train, y_train,epochs=10,batch_size=32,validation_split=0.1,shuffle=False)




    #input_path = valohai.inputs('dataset').path()
    #with np.load(input_path, allow_pickle=True) as f:
    #    x_train, y_train = f['x_train'], f['y_train']
    #    x_test, y_test = f['x_test'], f['y_test']

    #model = tf.keras.models.Sequential([
    #    tf.keras.layers.Flatten(input_shape=(28, 28)),
    #    tf.keras.layers.Dense(128, activation='relu'),
    #     tf.keras.layers.Dropout(0.2),
    #     tf.keras.layers.Dense(10),
    # ])

    # optimizer = tf.keras.optimizers.Adam(learning_rate=valohai.parameters('learning_rate').value)
    # loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # model.compile(optimizer=optimizer,
    #               loss=loss_fn,
    #               metrics=['accuracy'])

    # Print metrics out as JSON
    # This enables Valohai to version your metadata
    # and for you to use it to compare experiments

    ##callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_metadata)
    #model.fit(x_train, y_train, epochs=valohai.parameters('epochs').value, callbacks=[callback])

    # Evaluate the model and print out the test metrics as JSON

    #test_loss, test_accuracy = model.evaluate(x_test,  y_test, verbose=2)
    # with valohai.logger() as logger:
    #     logger.log('test_accuracy')
    #     logger.log('test_loss')

    # Write output files to Valohai outputs directory
    # This enables Valohai to version your data
    # and upload output it to the default data store

    suffix = uuid.uuid4()
    output_path = valohai.outputs().path(f'model-{suffix}.h5')
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)
    #model.save(output_path)


if __name__ == '__main__':
    main()
