import os
from collections import defaultdict

def parse_breakhis_filename(filename):
    """
    Extrae información del nombre de archivo BreakHis.
    Devuelve un diccionario con los campos relevantes.
    """
    name = os.path.splitext(filename)[0]

    try:
        parts = name.split('-')
        prefix = parts[0].split('_')

        info = {
            "dataset": prefix[0],              # SOB
            "class_code": prefix[1],           # B / M
            "subclass": prefix[2],             # A, F, PT, etc.
            "year": parts[1],                  # 14
            "patient_id": parts[2],            # 22549AB
            "magnification": parts[3],          # 200
            "sequence": parts[4]               # 001
        }
        return info

    except Exception as e:
        raise ValueError(f"Error parseando nombre de archivo: {filename}") from e


def read_binary_breakhis_data(base_path, verbose=False):
    """
    Lee el dataset BreakHis en clasificación binaria (benign / malignant).

    Returns:
        data: dict jerárquico [zoom][class] -> list(image_paths)
        all_images: list de rutas completas
        all_labels: list de labels (0 benign, 1 malignant)
        label_map: dict {'benign':0, 'malignant':1}
        slides: list de patient_id por imagen (alineado con all_images)
    """

    label_map = {
        "benign": 0,
        "malignant": 1
    }

    data = {}
    all_images = []
    all_labels = []
    slides = []

    if not os.path.isdir(base_path):
        raise ValueError(f"Ruta no válida: {base_path}")

    zoom_levels = sorted(os.listdir(base_path))

    if verbose:
        print(f"\n📂 Base path: {base_path}")
        print(f"🔍 Zoom levels encontrados: {zoom_levels}\n")

    for zoom in zoom_levels:
        zoom_path = os.path.join(base_path, zoom)
        if not os.path.isdir(zoom_path):
            continue

        data[zoom] = {}

        for class_name in ["benign", "malignant"]:
            class_path = os.path.join(zoom_path, class_name)

            if not os.path.isdir(class_path):
                if verbose:
                    print(f"⚠️  Carpeta no encontrada: {class_path}")
                continue

            image_files = [
                f for f in os.listdir(class_path)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]

            image_paths = []
            for img in image_files:
                full_path = os.path.join(class_path, img)
                image_paths.append(full_path)

                # Parseamos slide / paciente
                info = parse_breakhis_filename(img)
                patient_id = info["patient_id"]

                all_images.append(full_path)
                all_labels.append(label_map[class_name])
                slides.append(patient_id)

            data[zoom][class_name] = image_paths

            if verbose:
                print(f"🔬 Zoom {zoom} | Clase {class_name}")
                print(f"   ├── Imágenes: {len(image_paths)}")
                print(f"   ├── Label: {label_map[class_name]}")
                unique_patients = len(set(slides))
                print(f"   └── Pacientes únicos acumulados: {unique_patients}\n")

    if verbose:
        print("📊 RESUMEN FINAL")
        print(f"   Total imágenes: {len(all_images)}")
        print(f"   Total benignas: {all_labels.count(0)}")
        print(f"   Total malignas: {all_labels.count(1)}")
        print(f"   Total pacientes únicos: {len(set(slides))}")
        print(f"   Label map: {label_map}\n")

    return data, all_images, all_labels, label_map, slides

base_path = r"D:\TFG_BREAKHIST\BreakHist\data\BreakHis - Breast Cancer Histopathological Database\dataset_cancer_v1\dataset_cancer_v1\classificacao_binaria"

data, all_images, all_labels, label_map, slides = read_binary_breakhis_data(
    base_path=base_path,
    verbose=True
)