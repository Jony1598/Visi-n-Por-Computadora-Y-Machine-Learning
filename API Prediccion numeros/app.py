from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import base64
import io
from PIL import Image, ImageOps

app = Flask(__name__)

# Cargar dataset MNIST
datos, metadatos = tfds.load('mnist', as_supervised=True, with_info=True)
datos_entrenamiento, datos_pruebas = datos['train'], datos['test']

# Normalizar y preparar los datos para entrenamiento
def normalizar(imagenes, etiquetas):
    # Convertir enteros aflotantes
    imagenes = tf.cast(imagenes, tf.float32)
    # Dividir enrte 255
    imagenes /= 255
    return imagenes, etiquetas

datos_entrenamiento = datos_entrenamiento.map(normalizar)
datos_pruebas = datos_pruebas.map(normalizar)

# Agregar los datos a caché
datos_entrenamiento = datos_entrenamiento.cache()
datos_pruebas = datos_pruebas.cache()

# Crear modelo
modelo = tf.keras.Sequential([
    # Definir la primera capa de entrada Flatten
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    # Definir la primera capa oculta
    tf.keras.layers.Dense(128, activation='relu'),
    # tf.keras.layers.Dense(100, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.00001)),
    tf.keras.layers.Dense(10, activation='softmax')
])

modelo.compile(optimizer='adam',
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])

# Entrenar (1 época por rapidez, puedes ajustar)
# modelo.fit(datos_entrenamiento, epochs=1, validation_data=datos_pruebas)

# Necesitamos definir los hiperparametros
epochs = 7
learning_rate = 0.0008 #0.0019 #0.0005-34 0.0004-60 0.00039-74

# Definir el tamaño del lote (batch size)
batch_size = 32


# Compilar el modelo
modelo.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate),
    loss= tf.keras.losses.SparseCategoricalCrossentropy(), # Entropia cruzada
    metrics=['accuracy']
)

num_img_enrtrenamiento = metadatos.splits["train"].num_examples
num_img_pruebas = metadatos.splits["test"].num_examples

datos_entrenamiento = datos_entrenamiento.repeat().shuffle(num_img_enrtrenamiento).batch(batch_size)
datos_pruebas = datos_pruebas.batch(batch_size)

import math

# Emtrenar el modelo
steps_per_epoch = math.ceil(num_img_enrtrenamiento/batch_size)
historial = modelo.fit(datos_entrenamiento, epochs=epochs, steps_per_epoch=steps_per_epoch)


# Ruta raíz
@app.route("/")
def index():
    return render_template("index.html")

# Ruta para predecir imagen enviada desde frontend
@app.route("/predict", methods=["POST"])
def predict():
    # Recibir imagen base64 desde el front-end
    data = request.get_json()
    imagen_b64 = data["imagen"]
    imagen_bytes = base64.b64decode(imagen_b64.split(",")[1])
    imagen = Image.open(io.BytesIO(imagen_bytes)).convert("L")  # Convertir a escala de grises

    # Invertir colores (canvas: fondo blanco, dígito negro → MNIST: fondo negro, dígito blanco)
    imagen = ImageOps.invert(imagen)

    # Redimensionar a 28x28
    imagen = imagen.resize((28, 28))

    # Normalizar valores a [0, 1]
    imagen_np = np.array(imagen) / 255.0

    # Expandir dimensiones para que sea (1, 28, 28, 1)
    imagen_np = np.expand_dims(imagen_np, axis=(0, -1))

    # Predecir
    prediccion = modelo.predict(imagen_np)
    resultado = int(np.argmax(prediccion))

    return jsonify({"prediction": resultado})



if __name__ == "__main__":
    app.run(debug=True)
