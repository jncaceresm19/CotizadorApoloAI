import sys, io, glob
import numpy as np
from PIL import Image

import tensorflow as tf

MODELOS = {
    "material": ("modelo_material.tflite", "modelo_material_clases.txt"),
    "complejidad": ("modelo_complejidad.tflite", "modelo_complejidad_clases.txt"),
}


def cargar(path_modelo, path_clases):
    interp = tf.lite.Interpreter(model_path=path_modelo)
    interp.allocate_tensors()
    with open(path_clases, "r", encoding="utf-8") as f:
        clases = f.read().split(",")
    return interp, clases


def predecir(interp, clases, ruta_img):
    img = Image.open(ruta_img).convert("RGB").resize((224, 224))
    arr = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
    inp = interp.get_input_details()
    out = interp.get_output_details()
    interp.set_tensor(inp[0]["index"], arr)
    interp.invoke()
    pred = interp.get_tensor(out[0]["index"])[0]
    idx = int(np.argmax(pred))
    return clases[idx], round(float(pred[idx]), 3)


cargados = {k: cargar(*v) for k, v in MODELOS.items()}

muestras = []
for clase in ("complejo", "simple"):
    for ruta in glob.glob(f"dataset/complejidad/{clase}/*.jpg")[:2]:
        muestras.append((f"complejidad/{clase}", ruta))
for clase in ("abs+", "pla", "pla+"):
    for ruta in glob.glob(f"dataset/material/{clase}/*.jpg")[:2]:
        muestras.append((f"material/{clase}", ruta))

print("python", sys.version.split()[0], "| tf", tf.__version__)
print()
for esperado, ruta in muestras:
    mat, cm = predecir(*cargados["material"], ruta)
    comp, cc = predecir(*cargados["complejidad"], ruta)
    print(f"{esperado:22s} -> material={mat:5s} ({cm})  complejidad={comp:8s} ({cc})")
