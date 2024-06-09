Implementation of a parallel programming application to a Deep Learning neural network.

Introduction
An artificial neural network (ANN) is a computational model inspired by the behavior observed in its biological counterpart. The brain can be considered a highly complex system
complex system, where it is estimated that there are approximately 100 billion neurons forming a network of more than 500 trillion neural connections. The goal of the neural network is to solve problems in the same way as the human brain, although neural networks are more abstract. Current neural networks typically contain from a few thousand to a few million neural units.
In an artificial neural network each neuron is connected to other neurons through links. In these links the output value of the previous neuron is multiplied by a weight value.

These link weights can increase or inhibit the activation state of adjacent neurons. These systems are able to discover and learn patterns in a set of data to train themselves.
of data to form themselves, rather than being explicitly programmed, and excel in areas where solution or feature detection is difficult to express with conventional programming.

Neural networks have been used to solve a wide variety of tasks, such as computer vision and speech recognition, that are difficult to solve using ordinary rule-based programming.


Context Problem

El entrenamiento en PyTorch requiere que el desarrollador programe manualmente el bucle de entrenamiento. Durante cada iteración del bucle de entrenamiento (épocas), se siguen estos pasos:
•	Cargar subconjuntos de datos de entrenamiento (batch).
•	Reiniciar los gradientes a cero.
•	Realizar la propagación hacia adelante (forward propagation) para obtener predicciones.
•	Calcular la pérdida (loss) utilizando la función de pérdida.
•	Realizar la propagación hacia atrás (backward propagation) para calcular los gradientes.
•	Actualizar los pesos del modelo mediante el optimizador.
Este proceso se repite para cada batch de datos y se monitoriza la pérdida acumulada para evaluar el progreso del entrenamiento.
