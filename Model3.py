#Extracted from Coursera Convolutional Neural Networks
import numpy as np
import tensorflow as tf
from Constants import *
import matplotlib.pyplot as plt


#Function from https://gist.github.com/tomokishii/0ce3bdac1588b5cca9fa5fbdf6e1c412
def batch_norm(x, n_out, phase_train):
    """
    Batch normalization on convolutional maps.
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope('bn'):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                     name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                      name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed


def create_placeholders():
    X = tf.placeholder(tf.float32, shape=(None, IMAGE_SIZE, IMAGE_SIZE, 3))
    Y = tf.placeholder(tf.float32, shape=(None, NUM_CLASSES))
    return X, Y


def initialize_parameters():
    # 32 filters with shape  [4,4,3]
    W1 = tf.get_variable("W1", [4, 4, 3, 128], initializer=tf.contrib.layers.xavier_initializer())
    # 64 filters with shape [2,2,32]
    W2 = tf.get_variable("W2", [2, 2, 128, 256], initializer=tf.contrib.layers.xavier_initializer())
    # 200 filters with shape [2,2,64]
    W3 = tf.get_variable("W3", [2,2,256,200],initializer=tf.contrib.layers.xavier_initializer())
    #Weights & bias layer 200 neurons
    W4 = tf.get_variable("W4", [200,200],initializer=tf.contrib.layers.xavier_initializer())
    B4 = tf.get_variable("B4", [200],initializer=tf.contrib.layers.xavier_initializer())
    #Weights & bias  layer 100 neurons
    W5 = tf.get_variable("W5", [200, 100], initializer=tf.contrib.layers.xavier_initializer())
    B5 = tf.get_variable("B5", [100], initializer=tf.contrib.layers.xavier_initializer())
    #Weights & bias  layer 50 neurons
    W6 = tf.get_variable("W6", [100, 50], initializer=tf.contrib.layers.xavier_initializer())
    B6 = tf.get_variable("B6", [50], initializer=tf.contrib.layers.xavier_initializer())
    #Weights & bias  layer 10 neurons (Output)
    W7 = tf.get_variable("W7", [50, 10], initializer=tf.contrib.layers.xavier_initializer())
    B7 = tf.get_variable("B7", [10], initializer=tf.contrib.layers.xavier_initializer())
    parameters = {"W1": W1,
                  "W2": W2,
                  "W3": W3,
                  "W4": W4,
                  "W5": W5,
                  "W6": W6,
                  "W7": W7,
                  "B4": B4,
                  "B5": B5,
                  "B6": B6,
                  "B7": B7}
    return parameters


def dropout_performing(data, prob, training = True):
    if training:
        return tf.nn.dropout(data, prob)
    else:
        return data


def forward_propagation(X, parameters, training = True):

    # Retrieve the parameters from the dictionary "parameters"
    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']
    W4 = parameters['W4']
    W5 = parameters['W5']
    W6 = parameters['W6']
    W7 = parameters['W7']
    B4 = parameters['B4']
    B5 = parameters['B5']
    B6 = parameters['B6']
    B7 = parameters['B7']
    # CONV2D: stride of 1, padding 'SAME'
    Z1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
    # FULLY-CONNECTED 50 neurons with RELUZ1= batch_norm(Z1, 128, tf.get_variable("Training",training, dtype=tf.bool))
    # RELU
    A1 = tf.nn.relu(Z1)
    # MAXPOOL: window 8x8, sride 8, padding 'SAME'
    P1 = tf.nn.max_pool(A1, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding='SAME')
    # CONV2D: filters W2, stride 1, padding 'SAME'
    Z2 = tf.nn.conv2d(P1, W2, strides=[1, 1, 1, 1], padding='SAME')
    #Z2 = batch_norm(Z2, 256, tf.get_variable("Training",training, dtype=tf.bool))
    # RELU
    A2 = tf.nn.relu(Z2)
    # MAXPOOL: window 4x4, stride 4, padding 'SAME'
    P2 = tf.nn.max_pool(A2, ksize=[1, 4, 4, 1], strides=[1, 2, 2, 1], padding='SAME')
    # CONV2D: filters W3, stride 1, padding 'VALID'
    Z3 = tf.nn.conv2d(P2, W3, strides=[1, 1, 1, 1], padding='VALID')
    # RELU
    A3 = tf.nn.relu(Z3)
    A3= tf.contrib.layers.flatten(A3)
    # FULLY-CONNECTED 200 neurons with RELU
    #
    A4 = tf.nn.relu(tf.matmul(A3,W4)+B4)
    #A4 = tf.contrib.layers.fully_connected(A3, 200)
    A4_drop = dropout_performing(A4, 0.5, training)
    # FULLY-CONNECTED 100 neurons with RELU
    A5 = tf.nn.relu(tf.matmul(A4_drop,W5)+B5)
    #A5 = tf.contrib.layers.fully_connected(A4, 100)
    A5_drop = dropout_performing(A5, 0.4, training)
    A6 = tf.nn.relu(tf.matmul(A5_drop,W6)+B6)
    #A6 = tf.contrib.layers.fully_connected(A5, 50)
    A6_dop = dropout_performing(A6, 0.2, training)
    # FULLY-CONNECTED 10 neurons with Softmax (Output Layer)
    A7 = tf.nn.softmax(tf.matmul(A6_dop,W7)+B7)
    #A7 = tf.contrib.layers.fully_connected(A6, NUM_CLASSES, activation_fn=None)

    return A7


def compute_cost(A, Y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=A, labels=Y))
    return cost


def create_minibatches (X_train, Y_train, num_minibatches, batch_size = MINIBATHC_SIZE):
    minibatches_X = []
    minibatches_Y = []
    for i in range(num_minibatches):
            minibatches_X.append(X_train[i*batch_size: batch_size*(i+1)])
            minibatches_Y.append(Y_train[i * batch_size: batch_size * (i + 1)])
    minibatches_X.append(X_train[num_minibatches*batch_size:])
    minibatches_Y.append(Y_train[num_minibatches * batch_size:])
    minibatches_X = np.array(minibatches_X)
    minibatches_Y = np.array(minibatches_Y)
    return minibatches_X, minibatches_Y



def model(X_train, Y_train, X_test, Y_test, learning_rate = LEARNING_RATE,
          num_epochs= EPOCHS, print_cost=True):
    """
    Implements a three-layer ConvNet in Tensorflow:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED

    Arguments:
    X_train -- training set, of shape (None, 32, 32, 3)
    Y_train -- test set, of shape (None, n_y = 10)
    X_test -- training set, of shape (None, 32, 32, 3)
    Y_test -- test set, of shape (None, n_y = 10)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    print_cost -- True to print the cost every 100 epochs

    Returns:
    train_accuracy -- real number, accuracy on the train set (X_train)
    test_accuracy -- real number, testing accuracy on the test set (X_test)
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    tf.reset_default_graph()  # to be able to rerun the model without overwriting tf variables
    costs = []  # To keep track of the cost

    # Create Placeholders of the correct shape
    X, Y = create_placeholders()

    # Initialize parameters
    parameters = initialize_parameters()

    # Forward propagation: Build the forward propagation in the tensorflow graph
    Z3 = forward_propagation(X, parameters)
    Z3_Test = forward_propagation(X, parameters, False)

    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(Z3, Y)

    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer that minimizes the cost.
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    # Initialize all the variables globally
    init = tf.global_variables_initializer()

    #Creating Minibatches
    num_batches = int(X_train.shape[0] / MINIBATHC_SIZE)
    minibatches_X, minibatches_Y = create_minibatches(X_train, Y_train, num_batches)

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:

        # Run the initialization
        sess.run(init)

        # Do the training loop
        for epoch in range(num_epochs):
            batch_cost= 0
            for minibatch in range(num_batches):
                # Select a minibatch
                batch_X = minibatches_X[minibatch]
                batch_Y = minibatches_Y[minibatch]

                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the optimizer and the cost, the feedict should contain a minibatch for (X,Y).
                _, temp_cost = sess.run([optimizer, cost], feed_dict={X: batch_X, Y: batch_Y})
                batch_cost += temp_cost / num_batches
            # Print the cost every epoch
            if print_cost == True and epoch % 5 == 0:
                print("Cost after epoch %i: %f" % (epoch, batch_cost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(batch_cost)

        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # Calculate the correct predictions
        predict_op = tf.argmax(Z3_Test, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(accuracy)
        train_accuracy = 0
        test_accuracy = 0
        num_batches_test = int(X_test.shape[0] / MINIBATHC_SIZE)
        minibatches_X_test, minibatches_Y_test = create_minibatches(X_test, Y_test, num_batches_test)
        for i in range(num_batches):
            train_accuracy = (train_accuracy + accuracy.eval({X: minibatches_X[i], Y: minibatches_Y[i]}))
        for i in range(num_batches_test):
            test_accuracy = (test_accuracy + accuracy.eval({X: minibatches_X_test[i], Y: minibatches_Y_test[i]}))
        train_accuracy = train_accuracy/num_batches
        test_accuracy = test_accuracy/num_batches_test
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)

        return train_accuracy, test_accuracy, parameters