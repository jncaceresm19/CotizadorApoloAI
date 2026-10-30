import tensorflow as tf 
import os

IMG_SIZE = 224
BATCH = 32
EPOCHS = 15

def entrenar_modelo(ruta_dataset, nombre_salida):
    print(f"\n--- Entrenando {nombre_salida} desde {ruta_dataset} ---")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        ruta_dataset,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        ruta_dataset,
        validation_split=0.2,
        subset="validation", 
        seed=123,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH
    )

    class_names = train_ds.class_names
    print(f"Clases encontradas: {class_names}")

    # modelo base 
    base = tf.keras.applications.MobileNetV3Small(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )

    base.trainable = False

    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255),
        base,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(len(class_names), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

    # guardar como .tflite (4mb aprox)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(f"{nombre_salida}.tflite", "wb") as f:
        f.write(tflite_model)

    # guardar clases para usar en main.py
    with open(f"{nombre_salida}_clases.txt", "w") as f:
        f.write(",".join(class_names))

    print(f"Guardado {nombre_salida}.tflite ({len(tflite_model)/1024:.1f} KB)")


if __name__ == "__main__":
    # entrena material: dataset/material/pla+, dataset/material/pla, dataset/material/abs+
    if os.path.exists("dataset/material"):
        entrenar_modelo("dataset/material", "modelo_material")

    # entrena complejidad
    if os.path.exists("dataset/complejidad"):
        entrenar_modelo("dataset/complejidad", "modelo_complejidad")

    print("\nListo! Sube los .tflite y los .txt a GitHub junto con main.py")