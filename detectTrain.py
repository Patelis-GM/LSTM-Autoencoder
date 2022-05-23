import os
import sys
import random
import numpy
import tensorflow
from tensorflow.python.keras.layers import RepeatVector
from Utils.ArgumentParser import ArgumentParser
from tensorflow.keras.layers import LSTM, Dropout, Dense, TimeDistributed
from tensorflow.keras.models import Sequential
from Utils.Curve import Curve
from Utils.Parser import parse


# Utility function to ensure the reproducibility of the results
# The function's functionality is described in the following link :
# - https://keras.io/getting_started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
def experimentParameters():
    seed = 123
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tensorflow.random.set_seed(seed)
    numpy.random.seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    tensorflow.config.threading.set_inter_op_parallelism_threads(1)
    tensorflow.config.threading.set_intra_op_parallelism_threads(1)


if __name__ == '__main__':

    experimentParameters()

    argumentParser = ArgumentParser()
    argumentParser.addArgument(argument="-d", type="path", mandatory=True)

    if not argumentParser.parse(sys.argv):
        exit(1)

    path = argumentParser.getArgument("-d")

    curves = parse(path)

    # Keep 90% of the original Dataset
    curves = curves[:325]

    features = 1
    timesteps = 10
    batchSize = 64
    epochs = 50
    validationSplit = 0.15

    model = Sequential()

    # Encoder
    model.add(LSTM(64, input_shape=(timesteps, features), return_sequences=True))
    model.add(LSTM(64))

    # Encoder - Decoder bridge
    model.add(RepeatVector(timesteps))

    # Decoder
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(TimeDistributed(Dense(features)))
    model.compile(loss='mae', optimizer='adam')

    # Keep 80% of the Dataset as the Training Set
    trainPercentage = 0.8
    trainSet, _ = Curve.splitSet(curves, trainPercentage, asPercentage=True, shuffle=False)

    # The following way of training the model is verified by the Keras development team in the following link :
    # https://github.com/keras-team/keras/issues/4446
    for curve in trainSet:
        xTrain, yTrain = curve.sample(timesteps, length=len(curve), front=True, includeY=True, normalise=True)
        xTrain = numpy.reshape(xTrain, (xTrain.shape[0], timesteps, features))
        fitSummary = model.fit(xTrain, yTrain, batch_size=batchSize, epochs=epochs, verbose=0, validation_split=validationSplit, shuffle=False)
