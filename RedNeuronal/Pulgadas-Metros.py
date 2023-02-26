import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# ......... ENTRADAS ..........
entrada = np.array([1, 6, 30, 7, 70, 43, 503, 201, 1005, 99], dtype=float)

# ......... RESULTADOS ..........
resultados = np.array([0.0254, 0.1524, 0.762, 0.1778, 1.778, 1.0922, 12.7762, 5.1054, 25.527, 2.5146], dtype=float)

# ......... TOPOGRAFIA ...........
capa1 = tf.keras.layers.Dense(units=1, input_shape=[1])

# ......... TIPO DE RED ...........
modelo = tf.keras.Sequential([capa1])

# ......... OPTIMIZADOR Y METRICA DE PERDIDA ...........
modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)

# ......... ENTRENAMIENTO ..........
print("ENTRENANDO")
entrenamiento = modelo.fit(entrada, resultados, epochs=100, verbose=False)

# ......... GUARDAR ..........
modelo.save("Pulgadas-Metros.h5")
modelo.save_weights("PesosPlg-Mts.h5")

# ......... OBSERVAR ........
plt.xlabel("Numero de epocas")
plt.ylabel("Magnitud de perdida")
plt.plot(entrenamiento.history["loss"])
plt.show()

# ......... VERIFICAR ........
print("TERMINO EL ENTRENAMIENTO")

# ......... PREDICCION ........
i = input("INGRESAR EL VALOR EN PULGADAS: ")
i = float(i)

prediccion = modelo.predict([i])
print("EL VALOR EN METROS ES: ", str(prediccion))
