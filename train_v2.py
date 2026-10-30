"""Entrenamiento v2 de los clasificadores del cotizador.

Diferencias contra train.py:
  * validacion cruzada estratificada (k-fold) para medir de verdad, no un split unico
  * augmentation en el pipeline (no dentro del modelo, para no exportarla al .tflite)
  * dos etapas: cabeza con backbone congelado y luego fine-tuning de las ultimas capas
  * class_weight para el desbalance (abs+ 14 / pla 16 / pla+ 18)
  * early stopping + mejor checkpoint por val_accuracy
  * exporta .tflite manteniendo el contrato actual: entrada float32 0-255,
    shape [1,224,224,3], con el Rescaling(1/255) dentro del grafo

Uso:
    python train_v2.py                          # CV + modelo final + export
    python train_v2.py --solo-cv                # solo la medicion
    python train_v2.py --folds 5 --epochs-cabeza 15 --epochs-finetune 20
"""

import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf

IMG_SIZE = 224
SEED = 123
TAREAS = {
    "material": "dataset/material",
    "complejidad": "dataset/complejidad",
}
EXT_IMAGEN = {".jpg", ".jpeg", ".png", ".webp"}


def leer_dataset(carpeta):
    """Devuelve (imagenes float32 0-255, etiquetas int, nombres de clase)."""
    clases = sorted(d.name for d in Path(carpeta).iterdir() if d.is_dir())
    imagenes, etiquetas = [], []
    for idx, clase in enumerate(clases):
        for ruta in sorted((Path(carpeta) / clase).iterdir()):
            if ruta.suffix.lower() not in EXT_IMAGEN:
                continue
            img = tf.io.read_file(str(ruta))
            img = tf.io.decode_image(img, channels=3, expand_animations=False)
            img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE), method="bicubic")
            imagenes.append(tf.cast(img, tf.float32).numpy())
            etiquetas.append(idx)
    return np.stack(imagenes), np.array(etiquetas, dtype=np.int32), clases


def construir_pipeline(x, y, batch, entrenando):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if entrenando:
        ds = ds.shuffle(len(x), seed=SEED, reshuffle_each_iteration=True)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


@tf.keras.utils.register_keras_serializable(package="cotizador")
class NormalizarPorImagen(tf.keras.layers.Layer):
    """Estandariza cada imagen por su propia media y desvio.

    Medido sobre este dataset: quitar la exposicion global sube material de 79% a 88%
    y no afecta complejidad. El recorte aleatorio del augmentation, en cambio, la
    destruye (78%), por eso no se usa augmentation.
    """

    def call(self, entradas):
        x = tf.cast(entradas, tf.float32)
        media = tf.reduce_mean(x, axis=[1, 2, 3], keepdims=True)
        desvio = tf.math.reduce_std(x, axis=[1, 2, 3], keepdims=True)
        normalizado = (x - media) / (desvio + 1e-6) * 40.0 + 128.0
        return tf.clip_by_value(normalizado, 0.0, 255.0)


def crear_modelo(num_clases, base_entrenable=False, capas_descongeladas=0):
    base = tf.keras.applications.MobileNetV3Small(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
        include_preprocessing=False,
    )
    base.trainable = base_entrenable
    if base_entrenable and capas_descongeladas:
        for capa in base.layers[:-capas_descongeladas]:
            capa.trainable = False
    for capa in base.layers:
        if isinstance(capa, tf.keras.layers.BatchNormalization):
            capa.trainable = False

    entradas = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="imagen")
    x = NormalizarPorImagen(name="normalizar")(entradas)
    # MobileNetV3 espera pixeles 0-255 reescalados a [-1, 1] (scale=1/127.5, offset=-1).
    # El .tflite sigue recibiendo 0-255 en float32: toda la normalizacion vive en el grafo.
    x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1.0, name="rescale")(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    salidas = tf.keras.layers.Dense(num_clases, activation="softmax", name="clases")(x)
    return tf.keras.Model(entradas, salidas)


def pesos_por_clase(y, num_clases):
    conteo = np.bincount(y, minlength=num_clases)
    total = conteo.sum()
    return {i: float(total / (num_clases * max(c, 1))) for i, c in enumerate(conteo)}


def _etapa(modelo, lr, epochs, x_train, y_train, x_val, y_val, batch, pesos, ckpt, verbose):
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    callbacks = []
    if x_val is not None:
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="val_accuracy"),
            tf.keras.callbacks.ModelCheckpoint(str(ckpt), save_best_only=True, monitor="val_accuracy"),
        ]
    modelo.fit(
        construir_pipeline(x_train, y_train, batch, True),
        validation_data=construir_pipeline(x_val, y_val, batch, False) if x_val is not None else None,
        epochs=epochs,
        class_weight=pesos,
        verbose=verbose,
        callbacks=callbacks,
    )


def entrenar_modelo(x_train, y_train, x_val=None, y_val=None, num_clases=2,
                    epochs_cabeza=15, epochs_finetune=20, batch=16, verbose=0):
    """Dos etapas. Con x_val usa early stopping; sin x_val entrena epocas fijas."""
    ckpt = Path(f"_ckpt_{random.randint(0, 10 ** 9)}.keras")
    pesos = pesos_por_clase(y_train, num_clases)
    con_val = x_val is not None

    modelo = crear_modelo(num_clases)
    _etapa(modelo, 1e-3, epochs_cabeza, x_train, y_train, x_val, y_val, batch, pesos, ckpt, verbose)

    pesos_etapa2 = modelo.get_weights()
    modelo = crear_modelo(num_clases, base_entrenable=True, capas_descongeladas=40)
    modelo.set_weights(pesos_etapa2)
    _etapa(modelo, 1e-4, epochs_finetune, x_train, y_train, x_val, y_val, batch, pesos, ckpt, verbose)

    if con_val and ckpt.exists():
        final = tf.keras.models.load_model(str(ckpt))
        ckpt.unlink(missing_ok=True)
        return final
    ckpt.unlink(missing_ok=True)
    return modelo


def estratificar(y, k, rng):
    """Folds con la misma proporcion de clases que el total."""
    indices = np.arange(len(y))
    folds = [[] for _ in range(k)]
    for clase in np.unique(y):
        idx = indices[y == clase].copy()
        rng.shuffle(idx)
        for pos, i in enumerate(idx):
            folds[pos % k].append(i)
    return [np.array(sorted(f)) for f in folds]


def evaluar(modelo, x, y, clases):
    pred = np.argmax(modelo.predict(x, verbose=0), axis=1)
    matriz = np.zeros((len(clases), len(clases)), dtype=int)
    for real, p in zip(y, pred):
        matriz[real, p] += 1
    return float((pred == y).mean()), matriz


def validacion_cruzada(x, y, clases, folds, epochs_cabeza, epochs_finetune, batch):
    rng = random.Random(SEED)
    particiones = estratificar(y, folds, rng)
    exactitudes, matrices = [], []
    for i, val_idx in enumerate(particiones, start=1):
        train_idx = np.setdiff1d(np.arange(len(y)), val_idx)
        modelo = entrenar_modelo(
            x[train_idx], y[train_idx], x[val_idx], y[val_idx], len(clases),
            epochs_cabeza, epochs_finetune, batch,
        )
        acc, matriz = evaluar(modelo, x[val_idx], y[val_idx], clases)
        exactitudes.append(acc)
        matrices.append(matriz)
        print(f"   fold {i}: {acc * 100:5.1f}%  ({len(val_idx)} imgs)")
    return exactitudes, matrices


def exportar_tflite(modelo, destino):
    convertidor = tf.lite.TFLiteConverter.from_keras_model(modelo)
    convertidor.optimizations = [tf.lite.Optimize.DEFAULT]
    contenido = convertidor.convert()
    Path(destino).write_bytes(contenido)
    return len(contenido)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs-cabeza", type=int, default=15)
    ap.add_argument("--epochs-finetune", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--solo-cv", action="store_true")
    args = ap.parse_args()

    tf.get_logger().setLevel("ERROR")
    resumen = {}

    for tarea, carpeta in TAREAS.items():
        x, y, clases = leer_dataset(carpeta)
        print(f"\n=== {tarea}: {len(x)} imagenes, clases {clases} ===")

        exactitudes, matrices = validacion_cruzada(
            x, y, clases, args.folds, args.epochs_cabeza, args.epochs_finetune, args.batch
        )
        media, desvio = float(np.mean(exactitudes)), float(np.std(exactitudes))
        azar = 1.0 / len(clases)
        print(f"   CV {args.folds}-fold: {media * 100:.1f}% ± {desvio * 100:.1f}  (azar {azar * 100:.0f}%)")

        matriz_total = np.sum(matrices, axis=0)
        print("   matriz (filas=real, columnas=predicho):")
        print("      " + " ".join(f"{c:>8s}" for c in clases))
        for nombre, fila in zip(clases, matriz_total):
            print(f"      {nombre:>8s} " + " ".join(f"{v:8d}" for v in fila))

        resumen[tarea] = {
            "imagenes": int(len(x)),
            "clases": clases,
            "azar": round(azar, 3),
            "cv_media": round(media, 3),
            "cv_desvio": round(desvio, 3),
            "folds": [round(v, 3) for v in exactitudes],
            "matriz": matriz_total.tolist(),
        }

        if args.solo_cv:
            continue

        print("   entrenando modelo final con el 100% del dataset...")
        modelo = entrenar_modelo(x, y, num_clases=len(clases),
                                 epochs_cabeza=args.epochs_cabeza,
                                 epochs_finetune=args.epochs_finetune,
                                 batch=args.batch)
        destino = Path(f"modelo_{tarea}.tflite")
        backup = Path("modelos_v1")
        backup.mkdir(exist_ok=True)
        if destino.exists():
            shutil.copy2(destino, backup / destino.name)
            Path(f"modelo_{tarea}_clases.txt").exists() and shutil.copy2(
                Path(f"modelo_{tarea}_clases.txt"), backup / f"modelo_{tarea}_clases.txt"
            )
        tam = exportar_tflite(modelo, destino)
        Path(f"modelo_{tarea}_clases.txt").write_text(",".join(clases), encoding="utf-8")
        print(f"   exportado {destino} ({tam / 1024:.0f} KB), respaldo previo en modelos_v1/")

    Path("reporte_entrenamiento.json").write_text(json.dumps(resumen, indent=2), encoding="utf-8")
    print("\nreporte_entrenamiento.json escrito.")


if __name__ == "__main__":
    main()
