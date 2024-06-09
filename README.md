Implementation of a parallel programming application to a Deep Learning neural network.

Introduction
An artificial neural network (ANN) is a computational model inspired by the behavior observed in its biological counterpart. The brain can be considered a highly complex system
complex system, where it is estimated that there are approximately 100 billion neurons forming a network of more than 500 trillion neural connections. The goal of the neural network is to solve problems in the same way as the human brain, although neural networks are more abstract. Current neural networks typically contain from a few thousand to a few million neural units.
In an artificial neural network each neuron is connected to other neurons through links. In these links the output value of the previous neuron is multiplied by a weight value.

These link weights can increase or inhibit the activation state of adjacent neurons. These systems are able to discover and learn patterns in a set of data to train themselves.
of data to form themselves, rather than being explicitly programmed, and excel in areas where solution or feature detection is difficult to express with conventional programming.

Neural networks have been used to solve a wide variety of tasks, such as computer vision and speech recognition, that are difficult to solve using ordinary rule-based programming.


Context Problem

Today's companies are increasingly needing to increase their computational resources in order to be competitive in a highly technological marketplace.
in order to be competitive in a highly technological marketplace. As a result, the popularity of neural networks with Python is growing. Because the agility that
Python and the data processing power of ANNs make both elements together the most effective and efficient option for high-performance companies. In addition, neural networks and their subfield, deep learning, have achieved impressive results in areas such as robotics, robotic
impressive results in areas such as robotics, computer vision, Natural Language Processing (NLP) and Natural Language Understanding (NLU).

Methodology

1.	Tensorflow.Keras (tf.keras)

Keras is an open source Neural Networks library written in Python capable of running on top of TensorFlow, developed and maintained by François Chollet 2, a Google engineer. 

Model Definition

En tf.keras, se puede crear un modelo secuencial definiendo una serie de capas, que incluyen una capa de entrada, capas intermedias (ocultas) y una capa de salida. Por ejemplo, un modelo secuencial puede ser definido con una capa de entrada que recibe 500 parámetros, una capa oculta de 256.

