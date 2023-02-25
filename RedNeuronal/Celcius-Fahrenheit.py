import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# Entradas
celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype = float)
# Resultados para aprender, salidas
fahrenheit = np.array([-40, 14, 22, 46, 59, 72, 100], dtype = float)
# FORMA 1
# Capas de salida, tipo densa. Unidades de neuronas 1, entrada con 1 neurona
capa = tf.keras.layers.Dense(units=1, input_shape=[1])
# Modelo secuancial es el simple
modelo = tf.keras.Sequential([capa])
# FORMA 2
#capaoculta1 = tf.keras.layers.Dense(units=3, input_shape=[1])
#capaoculta2 = tf.keras.layers.Dense(units=3)
#capasalida = tf.keras.layers.Dense(units=1)
#modelo = tf.keras.Sequential([capaoculta1, capaoculta2, capasalida])
# Compilacion para entrenarlo
# Modelo de Adam, tasa de aprendizaje
# Funcion de perdida, mean_squared_error
modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)
# Entrenamiento
print("Entrenando...")
historial = modelo.fit(celsius, fahrenheit, epochs=1000, verbose=False)
print("Modelo entrenado")
# Resultado de la funcion de perdida
plt.xlabel("# de Epocas")
plt.ylabel("Magnitud de perdida")
plt.plot(historial.history["loss"])
# Prediccion 
print("Prediciendo")
resultado = modelo.predict([100.0])
print("El resultado es " + str(resultado)+ " fahrenheit")
# Que datos se asignaron a la conexion y al sesgo
print("Variables internas del modelo")
print(capa.get_weights())
#print(capaoculta1.get_weights())
#print(capaoculta2.get_weights())
#print(capasalida.get_weights())