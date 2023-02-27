import math
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
 
# ......... DATOS ........
datos, metadatos = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
datos_entrenamiento, datos_pruebas = datos['train'], datos['test']
# ......... CLASES ........
nombre_clases = metadatos.features['label'].names
print(nombre_clases)
# ......... NORMALIZAR DATOS ........
def normalizar(imagenes, etiquetas):
    imagenes = tf.cast(imagenes, tf.float32)
    imagenes /= 255
    return imagenes, etiquetas
datos_entrenamiento = datos_entrenamiento.map(normalizar)
datos_pruebas = datos_pruebas.map(normalizar)
# ......... GUARDAR DATOSA EN CACHE ........
datos_entrenamiento = datos_entrenamiento.cache()
datos_pruebas = datos_pruebas.cache()
# ......... MOSTRAR IMAGEN ........
for imagen, etiqueta in datos_entrenamiento.take(1):
    break
imagen = imagen.numpy().reshape((28,28)) # ... REDIMENSIONAR LA IMAGEN ...
# ......... DIBUJAR ........
plt.figure()
plt.imshow(imagen, cmap=plt.cm.binary)
plt.colorbar()
plt.grid(False)
plt.show()
plt.figure(figsize=(10,10))
for i, (imagen, etiqueta) in enumerate(datos_entrenamiento.take(25)):
    imagen = imagen.numpy().reshape((28,28)) # ... REDIMENSIONAR LA IMAGEN ...
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(imagen, cmap=plt.cm.binary)
    plt.xlabel(nombre_clases)
plt.show()
# ......... CREAR MODELO ........
modelo = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28,1)), # ... 1: BLANCO Y NEGRO ...
    tf.keras.layers.Dense(50, activation=tf.nn.relu),
    tf.keras.layers.Dense(50, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax) # ... PARA REDES DE CLASIFICACION ...
])
# ......... COMPILACION ..........
modelo.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)
# ......... ENTRENAMIENTO EN LOTES .........
num_ej_entrenamiento = metadatos.splits['train'].num_examples
num_ej_prueba = metadatos.splits['test'].num_examples
print(num_ej_entrenamiento, num_ej_prueba)
TAMAÑO_LOTE = 32
datos_entrenamiento = datos_entrenamiento.repeat().shuffle(num_ej_entrenamiento).batch(TAMAÑO_LOTE)
datos_pruebas = datos_pruebas.batch(TAMAÑO_LOTE)
# ......... ENTRENAR ..........
historial = modelo.fit(datos_entrenamiento, epochs=5, steps_per_epoch = math.ceil(num_ej_entrenamiento/TAMAÑO_LOTE))
# ......... OBSERVAR ........
plt.xlabel("Numero de epocas")
plt.ylabel("Magnitud de perdida")
plt.plot(historial.history["loss"])
plt.show()
# ......... TOMAR IMAGENES ........
for imagenes_prueba, etiquetas_prueba in datos_pruebas.take(1):
    imagenes_prueba=imagenes_prueba.numpy()
    etiquetas_prueba=etiquetas_prueba.numpy()
    predicciones=modelo.predict(imagenes_prueba)
def graficar_imagen(i, arr_predicciones, etiquetas_reales, imagenes):
    arr_predicciones, etiqueta_real, img=arr_predicciones[i], etiquetas_reales[i], imagenes[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    
    plt.imshow(img[...,0], cmap=plt.cm.binary)

    etiqueta_prediccion = np.argmax(arr_predicciones)
    if etiqueta_prediccion == etiqueta_real:
        color = 'blue' # ... SI LE PEGO ...
    else:
        color = 'red'  # ... SI NO LE PEGO ...
    plt.xlabel("{} {:20f}% ({})".format(
        nombre_clases[etiqueta_prediccion],
        100*np.max(arr_predicciones),
        nombre_clases[etiqueta_real],
        color=color
    ))
def graficar_valor_arreglo(i, arr_predicciones, etiqueta_real):
    arr_predicciones, etiqueta_real = arr_predicciones[i], etiqueta_real[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    grafica=plt.bar(range(10), arr_predicciones, color="#777777")
    plt.ylim([0,1])
    etiqueta_prediccion = np.argmax(arr_predicciones)

    grafica[etiqueta_prediccion].set_color('red')
    grafica[etiqueta_real].set_color('blue')
filas = 5
columnas = 5
num_imagenes = filas*columnas
plt.figure(figsize=(2*2*columnas, 2*filas))
for i in range(num_imagenes):
    plt.subplot(filas, 2*columnas, 2*i+1)
    graficar_imagen(i, predicciones, etiquetas_prueba, imagenes_prueba)
    plt.subplot(filas, 2*columnas, 2*i+2)
    graficar_valor_arreglo(i, predicciones, etiquetas_prueba)
plt.show()    
# ......... PRUEBA ........
imagen =imagenes_prueba[10]
imagen = np.array([imagen])
prediccion = modelo.predict(imagen)
print("Prediccion: ", nombre_clases[np.argmax(prediccion[0])])
