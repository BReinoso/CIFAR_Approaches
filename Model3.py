# Author: Bryan Reinoso Cevallos
# Description: New model with the goal of allow the bilduing convolutional Neural Networks of different sizes and layers
#               This model will be based in the definition of the neural network based in an expected structured data
import tensorflow as tf
from Constants import *
import logging
import datetime
# Global Parameters definition
now = datetime.datetime.now()
parameters = {}
layers_operations = {}
# Creating a logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
# Creating a handler and setting a debug level
handler = logging.FileHandler(CIFART_LOG + now.strftime("%Y-%m-%d_%H-%M")+".log")
handler.setLevel(logging.DEBUG)
# Creating a formatter to manage message shape
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
# Adding the formatter to the handler
handler.setFormatter(formatter)
# Adding the handler to the logger
logger.addHandler(handler)


def log_and_console_printing(message, log=False, console=False):
    if log:
        logging.debug(message)
    if console:
        print(message)


def create_placeholders():
    xv = tf.placeholder(tf.float32, shape=(None, IMAGE_SIZE, IMAGE_SIZE, 3))
    yv = tf.placeholder(tf.float32, shape=(None, NUM_CLASSES))
    return xv, yv


# Basic operation to add Conv Filter/fully connected layer Weights to the parameters
def create_weights(name, shape, initializerv):
    parameters[name] = tf.get_variable(name, shape, initializer=initializerv)


# Function to create all parameters for the neural network
def create_layer_parameters(number_layer, layer_type, shape, initializerv, log=False, console=False):
    if layer_type == 'CONV':
        log_and_console_printing("Creating a convolutional layer with shape " + str(shape), log=log, console=console)
        create_weights("W"+str(number_layer), shape[0], initializerv[0])
    elif layer_type == 'FULL':
        log_and_console_printing("Creating a fully connected layer layer with shape " + str(shape), log=log,
                                 console=console)
        create_weights("W" + str(number_layer[0]), shape[0], initializerv[0])
        create_weights("B" + str(number_layer[0]), shape[1], initializerv[1])
    elif layer_type == "RESB":
        log_and_console_printing("Creating a residual block", log=log, console=console)
        create_layer_parameters([number_layer[0]], [shape[0]], [initializerv[0]])
        create_layer_parameters([number_layer[1]], [shape[1]], [initializerv[1]])
        create_layer_parameters([number_layer[2]], [shape[2]], [initializerv[2]])
        log_and_console_printing("End of the residual block", log=log, console=console)


def dropout_performing(data, prob, training=True):
    if training:
        return tf.nn.dropout(data, prob)
    else:
        return data


def create_neural_network(layers):
    logger.debug("debug message")
    logger.info("info message")
    logger.warn("warn message")
    logger.error("error message")
    logger.critical("critical message")



# Batch Norm
# Cost
# Forward propagation
# Model
if __name__ == "__main__":
    create_neural_network("Testing Log")