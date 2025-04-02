import kagglehub
import shutil
import os


# Download latest version
path = kagglehub.dataset_download("andrewmvd/isic-2019")

print("Path to dataset files:", path)

# Ruta destino a tu carpeta 'dataset'
dest_path = "../dataset"  # ajusta según tu estructura

# Crear carpeta si no existe
os.makedirs(dest_path, exist_ok=True)

# Mover archivos y carpetas
for item in os.listdir(path):
    src = os.path.join(path, item)
    dst = os.path.join(dest_path, item)
    shutil.move(src, dst)

print("Todo el contenido fue movido a:", dest_path)
