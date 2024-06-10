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

Tensorflow.Keras (tf.keras)

Keras is an open source Neural Networks library written in Python capable of running on top of TensorFlow, developed and maintained by François Chollet 2, a Google engineer. 

Model Definition

En tf.keras, se puede crear un modelo secuencial definiendo una serie de capas, que incluyen una capa de entrada, capas intermedias (ocultas) y una capa de salida. Por ejemplo, un modelo secuencial puede ser definido con una capa de entrada que recibe 500 parámetros, una capa oculta de 256.

Model training settings

Unlike tf.keras, there is no method equivalent to model.compile() in PyTorch. 
Instead, the programmer must explicitly define the loss function and the optimizerssEntropyLoss and the optimizer Stochastic Gradient Descent (SGD) with a learning rate of 0.01. These components are invoked manually during the training process.

Model Training

Training in PyTorch requires the developer to manually program the training loop. During each iteration of the training loop (epochs), these steps are followed:
- Load subsets of training data (batch).
- Reset the gradients to zero.
- Perform forward propagation to obtain predictions.
- Calculate loss using the loss function.
- Perform backward propagation to calculate gradients.
- Update the model weights using the optimizer.
This process is repeated for each batch of data and the accumulated loss is monitored to evaluate the training progress.

Evaluation and storage

To evaluate the model in PyTorch, an evaluation loop using the test data must be programmed. During this process:
- Gradient computation is disabled.
- Forward propagation is performed to obtain predictions.
- The predictions are compared with the actual labels to calculate the accuracy.
Once satisfied with the results, the trained model can be saved using torch.save(), specifying the path to the file where the model will be stored.

Parallelization with Horovod

In Horovod, each node communicates with its neighboring nodes in an iterative process. During the first iterations, the received values are added to the node's data buffer, and in the following iterations, the received values replace the values in the node's buffer.

Horovod about tf.keras

To use Horovod with tf.keras, the following modifications must be made:
- Horovod initialization: import the corresponding library and call the initialization function.
- Assign each GPU to a single process:

o For TensorFlow versions prior to 2.0.0, set the visible GPU for each process.
o For TensorFlow 2.0.0.0 and higher, GPU memory growth must be enabled and the visible GPU must be configured for each process.
o Learning rate scaling: Adjust the learning rate proportionally to the number of GPUs used.

- Optimizer distribution: Use a distributed optimizer that delegates the gradient calculation to the original optimizer and applies the resulting gradients.
- Propagation of the initial state of the variables: Broadcast the initial state of the global variables to all GPUs to ensure consistent initialization at the beginning of the training process.
- Other modifications: Identify and isolate code to be executed on a single GPU to avoid redundancies and ensure correct parallel execution.

Horovod about PyTorch

To use Horovod with PyTorch, the following modifications must be made:
- Horovod initialization: import the corresponding library and call the initialization function.
- Assign each GPU to a single process: Set the visible GPU for each process if GPUs are available.
- Learning rate scaling: Adjust the learning rate proportionally to the number of GPUs used.
- Data set partitioning: Use a distributed sampler to partition the dataset among the nodes, ensuring that each node works with its subset of data.
- Optimizer distribution: Use a distributed optimizer that delegates the gradient computation to the original optimizer and applies the resulting gradients.
- Propagation of the initial state of the variables: broadcast the initial state of the model variables to all GPUs to ensure consistent initialization at the beginning of the training process.
- Other modifications: Identify and isolate code to be executed on a single GPU to avoid redundancies and ensure correct parallel execution.

Results
  
The same Horovod codes as in the CTE tests were used for the AWS performance tests, however it is important to note that the G3 instances as mentioned above use NVIDIA M60 GPUs. These graphics cards are quite inferior to the NVIDIA V100 in several aspects:
Memory bus width: The V100 has a bus width of 4,096 bits, while the M60 has only 256 bits of memory bus width.
Memory capacity: An NVIDIA V100 has 16GB of HBM2 memory. An M60 has 8GB GDDR5 memory.
Memory bandwidth: Up to 900GB/s of bandwidth the NVIDIA V100 is capable of, much higher than the 160GB/s of the M60.
For these reasons, it is understandable that the results obtained on AWS are significantly worse than those of CTE. If you want to use NVIDIA V100-equipped instances, you would have to choose instances from the P39 family, whose cost is logically much higher.
As expected, the difference in training times between AWS and CTE is very noticeable.
the model takes almost seven times as long to train on the AWS G3 instances.
G3 instances of AWS. However, the accuracy of the model once trained is very similar for both tf.keras and pytorch.

