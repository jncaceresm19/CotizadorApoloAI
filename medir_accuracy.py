import glob
import numpy as np
from PIL import Image
import tensorflow as tf

CASOS = {
    "complejidad": ("modelo_complejidad.tflite", "modelo_complejidad_clases.txt"),
    "material": ("modelo_material.tflite", "modelo_material_clases.txt"),
}


def cargar(path_modelo, path_clases):
    interp = tf.lite.Interpreter(model_path=path_modelo)
    interp.allocate_tensors()
    with open(path_clases, "r", encoding="utf-8") as f:
        clases = f.read().split(",")
    return interp, clases


def predecir(interp, clases, ruta):
    img = Image.open(ruta).convert("RGB").resize((224, 224))
    arr = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
    interp.set_tensor(interp.get_input_details()[0]["index"], arr)
    interp.invoke()
    pred = interp.get_tensor(interp.get_output_details()[0]["index"])[0]
    idx = int(np.argmax(pred))
    return clases[idx], float(pred[idx])


for nombre, (modelo, clases_txt) in CASOS.items():
    interp, clases = cargar(modelo, clases_txt)
    total = aciertos = 0
    confianzas = []
    por_clase = {}
    for clase_real in clases:
        rutas = glob.glob(f"dataset/{nombre}/{clase_real}/*")
        aciertos_clase = 0
        for ruta in rutas:
            pred, conf = predecir(interp, clases, ruta)
            total += 1
            confianzas.append(conf)
            if pred == clase_real:
                aciertos += 1
                aciertos_clase += 1
        por_clase[clase_real] = (aciertos_clase, len(rutas))
    print(f"{nombre}: {aciertos}/{total} = {100 * aciertos / max(total, 1):.1f}%  "
          f"(azar {100 / len(clases):.0f}%)  confianza media {np.mean(confianzas):.2f}")
    for clase_real, (ok, n) in por_clase.items():
        print(f"   {clase_real:8s} {ok}/{n}")
