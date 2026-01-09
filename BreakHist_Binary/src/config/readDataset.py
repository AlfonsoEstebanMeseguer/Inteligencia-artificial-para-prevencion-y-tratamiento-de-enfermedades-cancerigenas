import os
import argparse  

DEFAULT_BASE_PATH = os.environ.get("BREAKHIS_BASE",os.path.join(os.getcwd(),"BreakHist_Binary","BreakHist","data"
                                ,"BreakHis - Breast Cancer Histopathological Database"
                                ,"dataset_cancer_v1","dataset_cancer_v1","classificacao_binaria"))

"""
Función que devuelve un diccionario con las columnas parseadas del nombre de alguna imágen de BreakHist
"""
def parse_breakhis_filename(filename):
    name=os.path.splitext(filename)[0]
    # Eliminamos la extensión .png
    try:
        #   Ejemplo:                        SOB_B_A-14-22549AB-200-001.png
        parts=name.split('-')               # ['SOB_B_A','14','22549AB','200','001']
        prefix=parts[0].split('_')          # ['SOB','B','A']
        info = {
            "dataset":prefix[0],              # SOB
            "class_code":prefix[1],           # B|M
            "subclass":prefix[2],             # A,F,PT,etc.
            "year":parts[1],                  # 14
            "patient_id":parts[2],            # 22549AB
            "magnification":parts[3],         # 200
            "sequence":parts[4]               # 001
        }
        return info
    
    except Exception as e:
        raise ValueError(f"Error parseando nombre de archivo:{filename}") from e
    
"""
Función para leer el dataset BreakHis en clasificación binaria.
Lo que hacemos es:
    1. Definir un diccionario que mapea el nombre de clase al label numérico.
    2. Inicializar estructuras 
    3. Recorre carpeta zoom, luego la carpeta de clase benigna o maligna y por último las imágenes de cada una. 
    4. Se guardan las estructuras de datos de la siguiente forma:
        data={
            "40X":{
                "benign":[ruta_img1,ruta_img2,...],
                "malignant":[ruta_img1,ruta_img2,...]
            },
            "100X":{
                "benign":[ruta_img1,ruta_img2,...],
                "malignant":[ruta_img1,ruta_img2,...]
            },
            ...
        }
        slides=[patient_id1,patient_id2,...] # lista de pacientes (para futuro split)
        all_images=[ruta_img1,ruta_img2,...] # lista de todas las imágenes (para futuro train)
        all_labels=[label1,label2,...]       # lista de todos los labels (para futuro train)
"""
def read_binary_breakhis_data(base_path,verbose=True):
    data={}
    all_images=[]
    all_labels=[]
    slides=[] # guardar pacientes
    if not os.path.isdir(base_path): # Comprobamos que ruta base es válida
        raise ValueError(f"Ruta no válida:{base_path}")

    # obtenemos carpetas zoom
    zoom_levels=sorted(os.listdir(base_path))
    if not zoom_levels:
        raise ValueError("No se encontraron carpetas de zoom en la ruta base.")

    # definimos el mapa de labels usando el primer zoom
    first_zoom=zoom_levels[0]
    first_zoom_path=os.path.join(base_path,first_zoom)
    if not os.path.isdir(first_zoom_path): 
        raise ValueError(f"{first_zoom_path} no es un directorio")
    first_classes=[]
    for d in os.listdir(first_zoom_path):
        ruta=os.path.join(first_zoom_path,d)
        if os.path.isdir(ruta):
            first_classes.append(d)
    first_classes=sorted(first_classes)
    if not first_classes:
        raise ValueError(f"No se encontraron clases dentro de {first_zoom_path}")
    label_map={}
    for i,cls in enumerate(first_classes):
        label_map[cls]=i

    for zoom in zoom_levels: # Comprobamos que cada carpeta zoom es un directorio válido
        zoom_path=os.path.join(base_path,zoom)
        if not os.path.isdir(zoom_path): 
            raise ValueError(f"{zoom_path} no es un directorio")

        data[zoom]={}
        # obtenemos carpetas de clase dentro de cada zoom (forma simple)
        class_dirs=[]
        for d in os.listdir(zoom_path):
            ruta=os.path.join(zoom_path,d)
            if os.path.isdir(ruta):
                class_dirs.append(d)
        class_dirs=sorted(class_dirs)

        # comprobamos coherencia de clases con el primer zoom
        if class_dirs!=first_classes:
            raise ValueError(f"Clases no coinciden en zoom {zoom}: {class_dirs} vs {first_classes}")

        for class_name in class_dirs:
            class_path=os.path.join(zoom_path,class_name) # Ruta a la carpeta de la clase
            # Filtramos extensiones .png para este dataset
            image_files=[]
            for f in os.listdir(class_path):
                if f.lower().endswith('.png'):
                    image_files.append(f)

            image_paths = []
            for img in image_files:
                full_path = os.path.join(class_path, img) # Ruta completa a la imagen
                image_paths.append(full_path)
                # Obtenemos paciente (interesante para futuro split por paciente -> fuga de datos)
                info = parse_breakhis_filename(img)
                patient_id = info["patient_id"]
                all_images.append(full_path)
                all_labels.append(label_map[class_name])
                slides.append(patient_id)

            data[zoom][class_name] = image_paths 
            if verbose:
                print(f"Zoom{zoom}")
                print(f"Clase{class_name}")
                print(f"Imágenes:{len(image_paths)}")
                print(f"Label:{label_map[class_name]}")
                print(f"Pacientes únicos acumulados:{len(set(slides))}\n")
            
    if verbose:
        print("RESUMEN FINAL")
        print(f"Total imágenes:{len(all_images)}")
        for cls,idx in label_map.items():
            print(f"Total {cls}:{all_labels.count(idx)}")
        print(f"Total pacientes únicos:{len(set(slides))}")
        print(f"Label map:{label_map}\n")
    return data,all_images,all_labels,label_map,slides

# Función para parsear argumentos de línea de comandos 
# Ejemplo de uso: python readDataset.py --base-path "D:\\ruta\\al\\dataset" --no-verbose.
def parse_arguments():
    parser=argparse.ArgumentParser("Lectura del dataset BreakHis en clasificación binaria.")
    parser.add_argument("--base-path", default=DEFAULT_BASE_PATH, help="Ruta raíz del dataset BreakHis (por defecto la ruta original del script).")
    return parser.parse_args()

def main():
    args=parse_arguments()
    return read_binary_breakhis_data(args.base_path,True)

if __name__=="__main__":
    main()

