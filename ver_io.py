import tensorflow as tf

for nombre in ("modelo_material.tflite", "modelo_complejidad.tflite"):
    interp = tf.lite.Interpreter(model_path=nombre)
    interp.allocate_tensors()
    print(nombre)
    for d in interp.get_input_details():
        print("  input ", d["name"], d["shape"], d["dtype"].__name__)
    for d in interp.get_output_details():
        print("  output", d["name"], d["shape"], d["dtype"].__name__)
