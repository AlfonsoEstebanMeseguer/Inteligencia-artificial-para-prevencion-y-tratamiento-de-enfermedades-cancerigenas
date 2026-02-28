import os
import argparse
from pathlib import Path


DEFAULT_BASE_PATH = os.environ.get("BREAKHIS_BASE_MULTICLASS",os.path.join(os.getcwd(),"BreakHist_Multiclass","BreakHist","data"
                                                                           ,"BreakHis - Breast Cancer Histopathological Database","dataset_cancer_v1"
                                                                           ,"dataset_cancer_v1","classificacao_multiclasse"))


def parse_breakhis_filename(filename: str) -> dict:
    name = os.path.splitext(filename)[0]
    try:
        parts = name.split("-")  # ['SOB_B_A', '14', '22549AB', '200', '001']
        prefix = parts[0].split("_")  # ['SOB', 'B', 'A']
        return {
            "dataset": prefix[0],
            "class_code": prefix[1],  # B|M
            "subclass": prefix[2],  # A,F,PT,etc.
            "year": parts[1],
            "patient_id": parts[2],
            "magnification": parts[3],
            "sequence": parts[4],
        }
    except Exception as e:
        raise ValueError(f"Error parseando nombre de archivo: {filename}") from e


def read_multiclass_breakhis_data(base_path: str, verbose: bool = True):
    data = {}
    all_images = []
    all_labels = []
    slides = []

    if not os.path.isdir(base_path):
        raise ValueError(f"Ruta no válida: {base_path}")

    zoom_levels = sorted(os.listdir(base_path))
    if not zoom_levels:
        raise ValueError("No se encontraron carpetas de zoom en la ruta base.")

    # Definimos el mapa de labels usando el primer zoom
    first_zoom = zoom_levels[0]
    first_zoom_path = os.path.join(base_path, first_zoom)
    if not os.path.isdir(first_zoom_path):
        raise ValueError(f"{first_zoom_path} no es un directorio")

    first_classes = []
    for entry in os.listdir(first_zoom_path):
        entry_path = os.path.join(first_zoom_path, entry)
        if os.path.isdir(entry_path):
            first_classes.append(entry)
    first_classes = sorted(first_classes)
    if not first_classes:
        raise ValueError(f"No se encontraron clases dentro de {first_zoom_path}")

    label_map = {}
    class_index = 0
    for class_name in first_classes:
        label_map[class_name] = class_index
        class_index += 1

    for zoom in zoom_levels:
        zoom_path = os.path.join(base_path, zoom)
        if not os.path.isdir(zoom_path):
            raise ValueError(f"{zoom_path} no es un directorio")

        data[zoom] = {}
        class_dirs = []
        for entry in os.listdir(zoom_path):
            entry_path = os.path.join(zoom_path, entry)
            if os.path.isdir(entry_path):
                class_dirs.append(entry)
        class_dirs = sorted(class_dirs)

        if class_dirs != first_classes:
            raise ValueError(f"Clases no coinciden en zoom {zoom}: {class_dirs} vs {first_classes}")

        for class_name in class_dirs:
            class_path = os.path.join(zoom_path, class_name)
            image_files = []
            for f in os.listdir(class_path):
                if f.lower().endswith(".png"):
                    image_files.append(f)

            image_paths = []
            for img in image_files:
                full_path = os.path.join(class_path, img)
                image_paths.append(full_path)
                info = parse_breakhis_filename(img)
                slides.append(info["patient_id"])
                all_images.append(full_path)
                all_labels.append(label_map[class_name])

            data[zoom][class_name] = image_paths
            if verbose:
                print(f"Zoom {zoom}")
                print(f"Clase {class_name}")
                print(f"Imágenes: {len(image_paths)}")
                print(f"Label: {label_map[class_name]}")
                print(f"Pacientes únicos acumulados: {len(set(slides))}\n")

    if verbose:
        print("RESUMEN FINAL")
        print(f"Total imágenes: {len(all_images)}")
        for cls, idx in label_map.items():
            print(f"Total {cls}: {all_labels.count(idx)}")
        print(f"Total pacientes únicos: {len(set(slides))}")
        print(f"Label map: {label_map}\n")

    return data, all_images, all_labels, label_map, slides


def parse_arguments():
    parser = argparse.ArgumentParser("Lectura del dataset BreakHis en clasificación multiclase.")
    parser.add_argument("--base-path",default=DEFAULT_BASE_PATH,help="Ruta raíz del dataset BreakHis multiclase (puede ser sobrescrita por BREAKHIS_BASE_MULTICLASS).")
    parser.add_argument("--no-verbose", action="store_true", help="No imprimir resumen detallado.")
    return parser.parse_args()


def main():
    args = parse_arguments()
    return read_multiclass_breakhis_data(args.base_path, verbose=(not args.no_verbose))


if __name__ == "__main__":
    main()

