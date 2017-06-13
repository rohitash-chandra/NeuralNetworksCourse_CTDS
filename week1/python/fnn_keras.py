
import numpy as np
from keras.models import Sequential
from keras.layers import Dense


def normalisedata(data, inputsize, outsize):  # normalise the data between [0,1]. This is important for most problems.
    traindt = data[:, np.array(range(0, inputsize))]
    dt = np.amax(traindt, axis=0)
    tds = abs(traindt / dt)
    return np.concatenate((tds[:, range(0, inputsize)], data[:, range(inputsize, inputsize + outsize)]), axis=1)

def main():
    problem = 1  # [1,2,3] choose your problem (Iris classfication or 4-bit parity or XOR gate)

    if problem == 1:
        training_data = np.loadtxt("train.csv", delimiter=',')  # Iris classification problem (UCI dataset)
        test_data = np.loadtxt("test.csv", delimiter=',')  #
        hidden_size = 6
        input_size = 4
        output_size = 2
        x_train = training_data[:, 0:input_size]
        y_train = training_data[:, input_size:input_size+output_size]
        x_test = test_data[:, 0:input_size]
        y_test = test_data[:, input_size:input_size+output_size]


    if problem == 2:
        training_data = np.loadtxt("4bit.csv", delimiter=',')  # 4-bit parity problem
        test_data = np.loadtxt("4bit.csv", delimiter=',')  #
        hidden_size = 4
        input_size = 4
        output_size = 1

        x_train = training_data[:, 0:input_size]
        y_train = training_data[:, input_size:input_size+output_size]
        x_test = test_data[:, 0:input_size]
        y_test = test_data[:, input_size:input_size+output_size]

    if problem == 3:
        training_data = np.loadtxt("xor.csv", delimiter=',')  # 4-bit parity problem
        test_data = np.loadtxt("xor.csv", delimiter=',')  #
        hidden_size = 3
        input_size = 2
        output_size = 1
        x_train = training_data[:, 0:input_size]
        y_train = training_data[:, input_size:input_size+output_size]
        x_test = test_data[:, 0:input_size]
        y_test = test_data[:, input_size:input_size+output_size]

    model = Sequential()
    model.add(Dense(hidden_size, input_shape=(input_size,)))
    model.add(Dense(output_size, activation='sigmoid'))
    model.compile(optimizer='SGD', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=5, verbose=0, epochs=50)
    eval = model.evaluate(x_test, y_test)

    print("\nModel Loss: "+str(eval[0]))
    print("Model Accuracy: "+str(eval[1]))

if __name__ == "__main__": main()
