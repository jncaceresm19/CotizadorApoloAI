import os, rawpy, imageio

exts = ('.arw','.ARW')

for root, dirs, files in os.walk("dataset"):
    for f in files:
        if f.endswith(exts):
            path = os.path.join(root, f)
            try:
                with rawpy.imread(path) as raw:
                    rgb = raw.postprocess()
                nuevo = os.path.splitext(path)[0] + ".jpg"
                imageio.imsave(nuevo, rgb)
                print(f"OK: {f}")
                os.remove(path)
            except Exception as e:
                print(f"Error {f}: {e}")

print("Listo!")