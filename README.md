El entrenamiento en PyTorch requiere que el desarrollador programe manualmente el bucle de entrenamiento. Durante cada iteración del bucle de entrenamiento (épocas), se siguen estos pasos:
•	Cargar subconjuntos de datos de entrenamiento (batch).
•	Reiniciar los gradientes a cero.
•	Realizar la propagación hacia adelante (forward propagation) para obtener predicciones.
•	Calcular la pérdida (loss) utilizando la función de pérdida.
•	Realizar la propagación hacia atrás (backward propagation) para calcular los gradientes.
•	Actualizar los pesos del modelo mediante el optimizador.
Este proceso se repite para cada batch de datos y se monitoriza la pérdida acumulada para evaluar el progreso del entrenamiento.
