import random
import sys
import numpy
import pandas
import seaborn as seaborn
from matplotlib import pyplot
from tensorflow.python.keras.models import load_model
from Utils.ArgumentParser import ArgumentParser
from Utils.Parser import parse

if __name__ == '__main__':

    argumentParser = ArgumentParser()
    argumentParser.addArgument(argument="-d", type="path", mandatory=True)
    argumentParser.addNumericArgument(argument="-n", type="int", floor=1, ceiling=359, mandatory=True)
    argumentParser.addNumericArgument(argument="-mae", type="float", floor=0.00001, ceiling=10.0, mandatory=True)
    argumentParser.addArgument(argument="-r", type="bool", mandatory=False)

    if not argumentParser.parse(sys.argv):
        exit(1)

    path = argumentParser.getArgument("-d")
    n = argumentParser.getArgument("-n")
    threshold = argumentParser.getArgument("-mae")
    reconstruct = argumentParser.getArgument("-r")

    if reconstruct is None:
        reconstruct = False

    curves = parse(path)

    if len(curves) == 0:
        exit(1)

    # Select at random the indices of the Curves that will be plotted and their anomalies will be detected
    indices = list(range(len(curves)))
    random.shuffle(indices)
    indices = indices[:n]

    subplots = 2
    timesteps = 10
    features = 1
    modelPath = "Model"
    model = load_model(modelPath)


    if reconstruct:
        subplots += 1

    for index in indices:

        curve = curves[index]

        x = curve.sample(timesteps=timesteps, length=len(curve), front=True, includeY=False, normalise=True)
        x = numpy.reshape(x, (x.shape[0], timesteps, features))

        modelPrediction = model.predict(x)

        # Calculate the MAE given the model's prediction and the actual data
        mae = numpy.mean(numpy.abs(x - modelPrediction), axis=1)

        dataframe = pandas.DataFrame()
        dataframe['Values'] = curve.getValues()[timesteps:]
        dataframe['Mean-Absolute-Error'] = mae
        dataframe['Threshold'] = threshold

        # Mark as anomalies the row(s) where the corresponding MAE is above the threshold defined
        dataframe['Anomaly'] = dataframe['Mean-Absolute-Error'] > dataframe['Threshold']

        # Keep the row(s) that are marked as anomalies
        anomalies = dataframe[dataframe['Anomaly'] == True]

        figures, axes = pyplot.subplots(subplots)

        axes[0].plot(dataframe['Mean-Absolute-Error'], label='Mean Absolute Error')
        axes[0].plot(dataframe['Threshold'], label='Threshold')
        axes[0].legend()

        axes[1].plot(dataframe['Values'], label=curve.getID())
        seaborn.scatterplot(ax=axes[1], x=anomalies['Anomaly'].index, y=anomalies['Values'], color=seaborn.color_palette()[3], s=52, label='Anomaly')
        axes[1].legend()

        # Given :
        # - Curve C with values V = [1,2,3,4,5,6]
        # - Timesteps = 3
        #
        # The C sampling should return the following :
        # - X = [[1,2,3],[2,3,4],[3,4,5]]
        # - Y = [[4],[5],[6]]
        #
        # Say the best case scenario stands and the RNN LSTM Autoencoder predicts the following :
        #  - Reconstructed-C = RC = model.predict(X)
        #  - RC = [[1,2,3],[2,3,4],[3,4,5]]
        #
        # Reshaping RC and assigning it to Samples should lead to the following :
        # - Samples = [1,2,3,2,3,4,3,4,5]
        # - Prediction = []
        #
        # Execution of the while loop :
        # - 1st iteration : Prediction = [1,2,3] & i = 3
        # - 2nd iteration : Prediction = [1,2,3,4] & i = 6
        # - 3rd iteration : Prediction = [1,2,3,4,5] & i = 9
        #
        #  Prediction is now the reconstruction of the original C Curve
        if reconstruct:
            samples = numpy.reshape(modelPrediction, (modelPrediction.shape[0] * timesteps))
            prediction = []
            i = 0
            while i < len(samples):
                if i == 0:
                    for j in range(timesteps):
                        prediction.append(samples[j])
                    i = timesteps
                else:
                    prediction.append(samples[i + (timesteps - 1)])
                    i += timesteps

            prediction = numpy.array(prediction)
            prediction = curve.denormalise(prediction)
            axes[2].plot(curve.getValues(), label=curve.getID())
            axes[2].plot(prediction, label=curve.getID() + " Reconstruction")
            axes[2].legend()

        pyplot.show()
