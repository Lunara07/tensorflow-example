import uuid
import pickle


import numpy as np
import tensorflow as tf
import valohai
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor

from tensorflow import keras
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
#from imblearn.over_sampling import RandomOverSampler


# def log_metadata(epoch, logs):
#     """Helper function to log training metrics"""
#     with valohai.logger() as logger:
#         logger.log('epoch', epoch)
#         logger.log('accuracy', logs['accuracy'])
#         logger.log('loss', logs['loss'])


def main():

    valohai.prepare(
        step='train-model',
        image='tensorflow/tensorflow:2.6.0',
        default_inputs={
            'dataset': 'https://docs.google.com/spreadsheets/d/e/2PACX-1vSEhmFQ0wZgYM7Q_aETO8dPwA0eqShcQ2v5ps8oBZJYzgUR7fg1pS5hX8RdVPsb3u7tHmwTahl-nYdG/pub?gid=733135457&single=true&output=csv',
        }
    )

    
    # Read input files from Valohai inputs directory
    # This enables Valohai to version your training data
    # and cache the data for quick experimentation
    #print(valohai.inputs('dataset'))
    train = pd.read_csv(valohai.inputs('dataset').path())


    one_hot_encoded_train = pd.get_dummies(train, columns = ['Code'])
    one_hot_encoded_train = pd.get_dummies(one_hot_encoded_train, columns = ['Method'])
    
    train = one_hot_encoded_train.drop('Time', 1)
    y_train=train[["Y"]]
    X_train=train.drop("Y", 1)
    y_train.loc[y_train["Y"] == -1, "Y"] = 0
    #ros = RandomOverSampler(random_state=0)
    #X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)
    model = KNeighborsClassifier( weights='distance')

    model.fit(X_train, y_train['Y'])


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
    #model.save(output_path)
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)
    #model.save(output_path)


if __name__ == '__main__':
    main()
