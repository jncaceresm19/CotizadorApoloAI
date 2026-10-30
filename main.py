from fastapi import FastAPI, UploadFile, Form
import numpy as np
from PIL import Image, ImageOps
import io

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

app = FastAPI(title="ApoloVibes Cotizador 3D")

# --- CARGA MODELOS ---
def cargar_modelo(path_modelo, path_clases):
    interpreter = tflite.Interpreter(model_path=path_modelo)
    interpreter.allocate_tensors()
    with open(path_clases, "r") as f:
        clases = [c.strip() for c in f.read().split(",") if c.strip()]
    return interpreter, clases

interpreter_mat, clases_mat = cargar_modelo("modelo_material.tflite", "modelo_material_clases.txt")
interpreter_comp, clases_comp = cargar_modelo("modelo_complejidad.tflite", "modelo_complejidad_clases.txt")

# --- CONFIGURACION REAL APOLOVIBES ---
DENSIDAD_G_CM3 = {"pla": 1.24, "pla+": 1.24, "abs": 1.04, "abs+": 1.04}
PRECIO_KG_CLP = {"pla": 20000, "pla+": 22000, "abs+": 28000, "abs": 26000}
FACTOR_OCUPACION = {"simple": 0.28, "medio": 0.45, "complejo": 0.68, "compleja": 0.68}
CONFIG_USO = {
    "Decorativo": {"relleno": 0.15, "pared_mm": 1.2, "vel_g_h": 12.0, "merma_extra": 1.0},
    "Funcional":  {"relleno": 0.40, "pared_mm": 2.0, "vel_g_h": 10.0, "merma_extra": 1.05},
    "Prototipo":  {"relleno": 0.15, "pared_mm": 1.2, "vel_g_h": 13.0, "merma_extra": 1.0},
    "Exterior":   {"relleno": 0.60, "pared_mm": 2.4, "vel_g_h": 9.0,  "merma_extra": 1.1},
    "Otro":       {"relleno": 0.30, "pared_mm": 1.6, "vel_g_h": 11.0, "merma_extra": 1.05},
}
COSTO_HORA_MAQUINA = 1800  
COSTO_BASE = 3500 
TIEMPO_SETUP_H = 0.33

def predecir(interpreter, clases, imagen_bytes):
    
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(imagen_bytes))).convert("RGB")
    img = img.resize((224, 224), Image.BICUBIC)
    # FIX CRITICO: normalizar a 0-1 si tu modelo es MobileNet/EfficientNet
    img_array = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details[0]['index'])[0]
    idx = int(np.argmax(pred))
    return clases[idx].lower(), float(pred[idx])

@app.get("/")
def home():
    return {"status": "Cotizador 3D ApoloVibes activo - v2 preciso"}

@app.post("/cotizar")
async def cotizar(
    file: UploadFile,
    alto: float = Form(..., description="Alto en cm/mm"),
    ancho: float = Form(..., description="Ancho en cm/mm"),
    largo: float = Form(..., description="Largo en cm/mm"),
    unidad: str = Form("cm"),
    cantidad: int = Form(1),
    uso: str = Form("Decorativo"),
    material_usuario: str = Form(None, description="PLA, PLA+, ABS+, No estoy seguro"),
    color: str = Form(None)
):
    img_bytes = await file.read()

    # 1. Normalizar unidades a cm
    mult = 0.1 if unidad.lower() == "mm" else 1.0
    h, w, l = alto * mult, ancho * mult, largo * mult

    # 2. Material
    mat_input = (material_usuario or "").lower().strip()
    if mat_input in ["", "no estoy seguro", "no lo se", "no estoy seguro"]:
        material, conf_mat = predecir(interpreter_mat, clases_mat, img_bytes)
    else:
        material = mat_input
        conf_mat = 1.0
    
    if material not in DENSIDAD_G_CM3:
        material = "pla+"

    # 3. Complejidad
    complejidad, conf_comp = predecir(interpreter_comp, clases_comp, img_bytes)
    complejidad = complejidad.lower()
    if complejidad not in FACTOR_OCUPACION:
        complejidad = "medio"

    # 4. Calculo preciso
    cfg = CONFIG_USO.get(uso, CONFIG_USO["Decorativo"])
    densidad = DENSIDAD_G_CM3[material]
    factor_ocup = FACTOR_OCUPACION[complejidad]
    costo_g = (PRECIO_KG_CLP.get(material, 22000) / 1000.0) * 1.05 # 5% IVA perdido aprox

    vol_caja_cm3 = h * w * l
    vol_real_cm3 = vol_caja_cm3 * factor_ocup

    # Modelo hueco real: 18% es pared, 82% es interior con relleno
    vol_pared = vol_real_cm3 * 0.18
    vol_relleno = vol_real_cm3 * 0.82 * cfg["relleno"]
    vol_material_cm3 = vol_pared + vol_relleno

    # Merma por soportes segun complejidad
    merma_soporte = {"simple": 1.10, "medio": 1.25, "complejo": 1.35, "compleja": 1.35}[complejidad]
    merma_soporte *= cfg["merma_extra"]

    gramos_1_pieza = vol_material_cm3 * densidad * merma_soporte
    # tope minimo 8g (un llavero no puede pesar 2g)
    gramos_1_pieza = max(gramos_1_pieza, 8.0)
    
    gramos_total = gramos_1_pieza * cantidad
    
    # Tiempo: gramos / velocidad + setup por pieza
    tiempo_1 = (gramos_1_pieza / cfg["vel_g_h"]) + TIEMPO_SETUP_H
    # si es alta calidad (0.12mm) tarda x1.8
    if color and "alta" in color.lower():
        tiempo_1 *= 1.8
        
    tiempo_total = tiempo_1 * cantidad

    precio_unit = int(gramos_1_pieza * costo_g + tiempo_1 * COSTO_HORA_MAQUINA + COSTO_BASE)
    precio_total = int(gramos_total * costo_g + tiempo_total * COSTO_HORA_MAQUINA + COSTO_BASE * cantidad)
    
    # Precio minimo $7000
    precio_unit = max(precio_unit, 7000)
    precio_total = max(precio_total, 7000 * cantidad)

    return {
        "material_detectado": material,
        "confianza_material": round(conf_mat, 2),
        "complejidad": complejidad,
        "confianza_complejidad": round(conf_comp, 2),
        "uso_detectado": uso,
        "volumen_caja_cm3": round(vol_caja_cm3, 1),
        "gramos_1_pieza": round(gramos_1_pieza, 1),
        "gramos_estimados": round(gramos_total, 1),
        "tiempo_estimado_horas": round(tiempo_total, 2),
        "tiempo_unitario_horas": round(tiempo_1, 2),
        "precio_unitario_clp": precio_unit,
        "precio_clp": precio_total,
        "detalle": {
            "costo_filamento": int(gramos_total * costo_g),
            "costo_tiempo_maquina": int(tiempo_total * COSTO_HORA_MAQUINA),
            "costo_base": COSTO_BASE * cantidad
        }
    }