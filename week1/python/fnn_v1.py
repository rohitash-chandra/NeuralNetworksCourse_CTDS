# Rohitash Chandra, 2017 c.rohitash@gmail.conm

# !/usr/bin/python

# Feedforward Neural Network in Python (Classification Problem used for teaching and learning purpose)


# Sigmoid units used in hidden and output layer. gradient descent and stocastic gradient descent functions implemented
# this is best for teaching and learning as numpy arrays are not used   in Forward and Backward Pass.
# numpy implymentation with momemtum is given here: https://github.com/rohitash-chandra/VanillaFNN-Python
# corresponding C++ implementation which is generalised to any number of hidden layers is given here:
# https://github.com/rohitash-chandra/feedforward-neural-network

# problems: https://en.wikipedia.org/wiki/XOR_gate, (4 Bit Parity) https://en.wikipedia.org/wiki/Parity_bit
# wine classification: https://archive.ics.uci.edu/ml/datasets/wine

# infor: https://en.wikipedia.org/wiki/Feedforward_neural_network
# http://media.wiley.com/product_data/excerpt/19/04713491/0471349119.pdf





import matplotlib.pyplot as plt
import numpy as np
import random
import time

class Network:

    def __init__(self, layers, x_train, y_train, epochs, learning_rate, MinPer):
        self.layers = layers  # NN topology [input, hidden, output]
        self.epochs = epochs  # max epocs or training time
        self.num_samples = len(x_train)

        self.x_train = x_train
        self.y_train = y_train

        self.learning_rate = learning_rate  # will be updated later with BP call
        self.minPerf = MinPer
        # initialize weights ( W1 W2 ) and bias ( b1 b2 ) of the network
        np.random.seed()

        self.w1 = np.random.randn(self.layers[0], self.layers[1])
        self.b1 = np.random.randn(self.layers[1])  # bias first layer
        self.best_b1 = self.b1
        self.best_w1 = self.w1

        self.w2 = np.random.randn(self.layers[1], self.layers[2])
        self.b2 = np.random.randn(self.layers[2])  # bias second layer
        self.best_b2 = self.b2
        self.best_w2 = self.w2

        self.hidden_out = np.zeros(self.layers[1])  # output of first hidden layer
        self.hidden_delta = np.zeros(self.layers[1])  # gradient of first hidden layer

        self.output = np.zeros(self.layers[2])  # output last (output) layer
        self.output_delta = np.zeros(self.layers[2])  # gradient of  output layer

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def calculate_error(self, actual_out, output_layer_size=None):
        if output_layer_size is None:
            output_layer_size = self.layers[-1]

        error = np.sum(np.square(np.subtract(self.output, actual_out))) / output_layer_size
        return error

    def get_weighted_sum(self,weights, bias, input_layer_size,input, output_layer_size, output, activation_function):
        weighted_sum = 0
        for y in range(0, output_layer_size):
            for x in range(0, input_layer_size):
                weighted_sum += input[x] * weights[x][y]
                output[y] = activation_function(weighted_sum - bias[y])
            weighted_sum = 0

    def forward_pass(self, input):

        # This code expects a single hidden layer perceptron with input - hidden - output architecture
        # THIS SHOULD CHANGE WHEN IMPLEMENTING MLP WITH MULTIPLE LAYERS

        input_layer_size = self.layers[0]
        hidden_layer_size = self.layers[1]
        output_layer_size = self.layers[2]

        # Each Neuron on the INPUT layer has 'y' connections to the neurons on the HIDDEN layer
        # That's why the loop runs from the hidden layer to the input layer
        self.get_weighted_sum(self.w1, self.b1, input_layer_size, input, hidden_layer_size,
                              self.hidden_out, self.sigmoid)

        # Each Neuron on the HIDDEN layer has 'y' connections to the neurons on the OUTPUT layer
        # That's why the loop runs from the OUTPUT layer to the HIDDEN layer
        self.get_weighted_sum(self.w2, self.b2, hidden_layer_size, self.hidden_out, output_layer_size,
                              self.output, self.sigmoid)

    def get_output_layer_gradient(self, y, layer_size):
        """
        Compute gradients for the output layer which is defined by:
             (gradient of the cost w.r.t neuron activation) * (derivative of the sigmoid of the neuron weighted sum)
        
        It can bet interpreted as how much the cost and the activation is changing w.r.t to that neuron output
        :param y: the real output
        :param layer_size: the number of the neurons in the layer
        :return: a delta array for that layer
        """
        delta = np.zeros(layer_size)
        for x in range(0, layer_size):
            delta[x] = (y[x] - self.output[x]) * (self.output[x] * (1 - self.output[x]))

        return delta

    def get_hidden_layer_gradient(self, hidden_layer_size, hidden_output, next_layer_size, next_layer_delta, weights):
        temp = 0
        hidden_delta = np.zeros(hidden_layer_size)
        for x in range(0, hidden_layer_size):
            for y in range(0, next_layer_size):
                temp += (next_layer_delta[y] * weights[x, y])
                hidden_delta[x] = (hidden_output[x] * (1 - hidden_output[x])) * temp
                temp = 0
        return hidden_delta

    def update_weights_bias(self, weights, bias, hidden_layer_size, hidden_output, next_layer_size, next_layer_delta):
        for x in range(0, hidden_layer_size):
            for y in range(0, next_layer_size):
                weights[x, y] += self.learning_rate * next_layer_delta[y] * hidden_output[x]

        for y in range(0, next_layer_size):
            bias[y] += -1 * self.learning_rate * next_layer_delta[y]

    def backward_pass(self, input, y):

        input_layer_size = self.layers[0]
        hidden_layer_size = self.layers[1]
        output_layer_size = self.layers[2]

        self.output_delta = self.get_output_layer_gradient(y, output_layer_size)
        self.hidden_delta = self.get_hidden_layer_gradient(hidden_layer_size,
                                                           self.hidden_out, output_layer_size,
                                                           self.output_delta, self.w2)

        self.update_weights_bias(self.w2, self.b2, hidden_layer_size, self.hidden_out, output_layer_size,
                                 self.output_delta)

        self.update_weights_bias(self.w1, self.b1, input_layer_size, input, hidden_layer_size,
                                 self.hidden_delta)


    def TestNetwork(self, Data, erTolerance):

        clasPerf = 0
        sse = 0
        self.w1 = self.best_w1
        self.w2 = self.best_w2  # load best knowledge
        self.b1 = self.best_b1
        self.b2 = self.best_b2  # load best knowledge

        testSize = Data.shape[0]

        for s in range(0, testSize):

            Input = Data[s, 0:self.layers[0]]
            Desired = Data[s, self.layers[0]:]

            self.forward_pass(Input)
            sse = sse + self.calculate_error(Desired)

            if (np.isclose(self.output, Desired, atol=erTolerance).any()):
                clasPerf = clasPerf + 1

        return (sse / testSize, float(clasPerf) / testSize * 100)

    def save_model(self):
        self.best_w1 = self.w1
        self.best_w2 = self.w2
        self.best_b1 = self.b1
        self.best_b2 = self.b2

    def backpropagation(self, stochastic=True, train_tolerance=0):  # BP with Vanilla or SGD (Stocastic BP)

        error = []  # np.zeros((1, self.Max))
        epoch = 0
        best_mean_squared_error = 100
        bestTrain = 0

        input_layer_size = self.layers[0]

        while epoch < self.epochs and bestTrain < self.minPerf:

            sse = 0
            for s in range(0, self.num_samples):

                if stochastic:
                    idx = random.randint(0, self.num_samples - 1)
                else:
                    idx = s

                # Get the first 'input_layer_size' columns as features
                x = self.x_train[idx, 0:input_layer_size]
                # Get the last 'output_layer_size' columns (real outputs)
                y = self.x_train[idx, input_layer_size:]

                self.forward_pass(x)
                self.backward_pass(x, y)
                sse = sse + self.calculate_error(y)

            mean_squared_error = np.sqrt(sse / self.num_samples * self.layers[-1])


            if mean_squared_error < best_mean_squared_error:
                best_mean_squared_error = mean_squared_error
                self.save_model()
                (x, bestTrain) = self.TestNetwork(self.x_train, train_tolerance)

            error = np.append(error, mean_squared_error)

            epoch = epoch + 1


        return (error, best_mean_squared_error, bestTrain, epoch)


def normalisedata(data, inputsize, outsize):  # normalise the data between [0,1]. This is important for most problems.
    traindt = data[:, np.array(range(0, inputsize))]
    dt = np.amax(traindt, axis=0)
    tds = abs(traindt / dt)
    return np.concatenate((tds[:, range(0, inputsize)], data[:, range(inputsize, inputsize + outsize)]), axis=1)


def main():
    problem = 1  # [1,2,3] choose your problem (Iris classfication or 4-bit parity or XOR gate)

    if problem == 1:
        TrDat = np.loadtxt("train.csv", delimiter=',')  # Iris classification problem (UCI dataset)
        TesDat = np.loadtxt("test.csv", delimiter=',')  #
        Hidden = 6
        Input = 4
        Output = 2
        TrSamples = 110
        TestSize = 40
        learnRate = 0.1
        TrainData = normalisedata(TrDat, Input, Output)
        TestData = normalisedata(TesDat, Input, Output)
        MaxTime = 500
        MinCriteria = 95  # stop when learn 95 percent

    if problem == 2:
        TrainData = np.loadtxt("4bit.csv", delimiter=',')  # 4-bit parity problem
        TestData = np.loadtxt("4bit.csv", delimiter=',')  #
        Hidden = 4
        Input = 4
        Output = 1
        TrSamples = 16
        TestSize = 16
        learnRate = 0.9
        MaxTime = 3000
        MinCriteria = 95  # stop when learn 95 percent

    if problem == 3:
        TrainData = np.loadtxt("xor.csv", delimiter=',')  # 4-bit parity problem
        TestData = np.loadtxt("xor.csv", delimiter=',')  #
        Hidden = 3
        Input = 2
        Output = 1
        TrSamples = 4
        TestSize = 4
        learnRate = 0.9
        MaxTime = 500
        MinCriteria = 100  # stop when learn 95 percent

    print(TrainData)

    print('printed data. now we use FNN for training ...')

    layers = [Input, Hidden, Output]
    MaxRun = 10  # number of experimental runs

    trainTolerance = 0.2  # [eg 0.15 would be seen as 0] [ 0.81 would be seen as 1]
    testTolerance = 0.49

    trainPerf = np.zeros(MaxRun)
    testPerf = np.zeros(MaxRun)

    trainMSE = np.zeros(MaxRun)
    testMSE = np.zeros(MaxRun)
    Epochs = np.zeros(MaxRun)
    Time = np.zeros(MaxRun)

    stocastic = 1  # 0 if vanilla (batch mode)

    for run in range(0, MaxRun):
        print(run, 'is the experimental run')
        fnnSGD = Network(layers, TrainData, TestData, MaxTime, learnRate, MinCriteria)
        start_time = time.time()
        (erEp, trainMSE[run], trainPerf[run], Epochs[run]) = fnnSGD.backpropagation(stocastic, trainTolerance)
        Time[run] = time.time() - start_time
        (testMSE[run], testPerf[run]) = fnnSGD.TestNetwork(TestData, testTolerance)

    print(trainPerf, 'train perf % for n exp')
    print(testPerf, 'test  perf % for n exp')
    print(trainMSE, 'train mean squared error for n exp')
    print(testMSE, 'test mean squared error for n exp')

    print('mean and std for training perf %')
    print(np.mean(trainPerf), np.std(trainPerf))

    print('mean and std for test perf %')
    print(np.mean(testPerf), np.std(testPerf))

    print('mean and std for time in seconds')
    print(np.mean(Time), np.std(Time))

    plt.figure()
    plt.plot(erEp)
    plt.ylabel('error')
    plt.savefig(str(problem) + 'out.png')


if __name__ == "__main__": main()
