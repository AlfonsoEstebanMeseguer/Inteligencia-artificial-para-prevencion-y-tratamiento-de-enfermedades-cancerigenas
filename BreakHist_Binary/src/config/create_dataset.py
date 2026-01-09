import os
import sys
import json
import argparse  # Parametrización CLI sin alterar la lógica original
from datetime import datetime
import glob
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path

PROJECT_ROOT=Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0,str(PROJECT_ROOT))

from src.config.augmentations import (NONE, LOW, MEDIUM, ADVANCED, EXPERT, apply_augmentations, params)
from src.config.normalization import (IMAGENET, RESNET, EFFICIENTNET, CUSTOM, STANDARD, normalize_image)

SIZE_DEFAULT = (224,224)
BUFFER_DEFAULT = 1000
CACHE_DEFAULT = True
SHUFFLE_DEFAULT = True
PREFETCH_DEFAULT = True

"""
Esta es una función muy sencilla, hace comprobaciones de la configuración solicitada y devuelve un diccionario con la configuración del dataset, listo
para su construcción. 
"""
def DatasetConfig(img_size,batch_size,buffer_size,augmentation_level
                  ,normalization_mode,seed,use_class_weights,cache
                  ,shuffle_train,prefetch):
    # Comprobación de valores encontrados (no hace falta hacer minúsculas, se ahce antes de llamar a la función)
    if augmentation_level not in [NONE,LOW,MEDIUM,ADVANCED,EXPERT]:
        raise ValueError(f"Nivel de augmentación'{augmentation_level}' no válido. Valores permitidos: {NONE}, {LOW}, {MEDIUM}, {ADVANCED}, {EXPERT}")

    if normalization_mode not in [IMAGENET,RESNET,EFFICIENTNET,CUSTOM,STANDARD]:
        raise ValueError(f"Modo de normalización'{normalization_mode}' no válido. Valores permitidos: {IMAGENET}, {RESNET}, {EFFICIENTNET}, {CUSTOM}, {STANDARD}")

    augmentation_params=params.get(augmentation_level) # obtenemos parámetros de augmentación
    # Devolvemos configuración como diccionario para la construcción del dataset
    return {"img_size":tuple(img_size),"batch_size":batch_size,"buffer_size":buffer_size,"augmentation_level":augmentation_level
            ,"normalization_mode":normalization_mode,"seed":seed,"use_class_weights":use_class_weights,"cache":cache,"shuffle_train":shuffle_train
            ,"prefetch":prefetch,"augmentation_params":augmentation_params}

"""
Función que carga archivo json con los valores spliteados de cada conjunto y devuelve las listas de imágenes y labels correspondientes por conjunto
Esta función es necesaria para evitar por ejemplo augmentations en otros conjuntos que no sean train, o normalizar para todos, es decir, 
separar el tratamiento de los distintos conjuntos
"""
def load_split(split_dir,split_type):
    json_path=os.path.join(split_dir,f"{split_type}.json") 
    # Buscamos json de "train","val","test" en directorio split generado tras script split_dataset.py
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"No se encontró el archivo: {json_path}")
    
    with open(json_path,"r",encoding="utf-8") as f:
        data=json.load(f) # Cargamos el json correspondiente

    images=data.get("images",[])
    labels=data.get("labels",[])
    if len(images)!=len(labels): # Comprobación de integridad
        raise ValueError(f"Inconsistencia en {json_path}:{len(images)}imágenes vs{len(labels)}etiquetas")
    
    return images,labels

"""
Funciónq que calcula los pesos de clase dados los labels de entrenamiento. Para ello usamos la función compute_class_weight de scikit-learn (preparada ya para
tarbajar con este tipo de trabajos de clasificación). La fórmula empleada para devolver los pesos es la siguiente:

wc = N / (K * nc) donde:
N = num muestras, K =num clases, nc = num muestras de clase 'c'. 

Ejemplo visual con nuestro dataset:
labels=[0]*800 +[1]*200
N = 1000, K = 2, clases benignas = 800, clases malignas = 200
w(benignos) = 1000 / (2*800) = 0.625
w(malignos) = 1000 / 2*200) = 2.5
"""
def compute_class_weights(train_labels):
    labels = np.array(train_labels,np.int32) # para operaciones matemáticas forzamos
    classes = np.unique(labels) # evitamos duplicados (0,1 y ya)
    weights=compute_class_weight(class_weight="balanced",classes=classes,y=labels)
    # Es necesario guardar el índice, porque aunque calculemos el peso no sabemos de que entidad,
    # Ejemplo:(0,0.625), (1,2.5)
    weights=dict(zip(classes,weights))
    counts=np.unique(labels,return_counts=True)[1]
    print("Pesos de clase:")
    for cls,w in weights.items():
        cls_count=counts[list(classes).index(cls)]
        print(f"clase {cls}:peso={w:.3f},muestras={cls_count}")
    return weights

"""
Función que decodifica una imagen desde una ruta dada . img_path no es un string normal, es tipo:
tf.Tensor(b"/ruta/a/imagen.png", shape=(), dtype=string), con este objeto no se puede usar open() ni similares.
Por eso vamos a leer los bytes, que nos devuelve el mismo tensor en bytes y gracias a tf.image.decode_image
obtenemos formato (.png) y con ello la imagen en un tensor. Despues fijamos los mismos tamaños y forzamos a 3 canales 
(para RGB, que se necesita en tf).
"""
def decode_image(img_path):
    img_bytes=tf.io.read_file(img_path)
    img=tf.image.decode_image(img_bytes,3)
    img.set_shape([None, None, 3])
    return img
"""
Esta función recibe una iamgen decodificada. Aplica preprocesado a train/val/test, aplica augmentación solo al entrenamiento y devuelve
la imagen y el label listo para entrenar. Primero coge la imagen decodificada y la castea a float32, más que nada porque tensorflow y las cnns
asumen float32 en las imágenes, así como funciones como resize (que son necesarias para transfer-learning, por ejemplo). 
Obligatorio redimnesionar, pese a que todas las imágenes de BreakHis son 700x460, es buena práctica redimensionar a un tamaño fijo (config.img_size),
pues las cnns no trabajan con tamaños variables. Además, hacerlo pronto reduce memoria y acelera pipeline. 
"""
def preprocess_image(img,training,config):
    img=tf.cast(img,tf.float32)
    img=tf.image.resize(img,config["img_size"])
    if training:
        """
        Esto es importante, solo buscamos variabilidad en el entrenamiento, por ello solo vamos a augmentar train. Lo primero que hacemos aquí 
        es pasar a rango [0,1]. ¿Por que? porque la mayoría de augmentaciones están definidas para imágenes en ese rango (0-1), no 0-255.
        Si el rango fuera 0-225 (imágenes tçipicas), los cambios serían enormes. 
        """
        img01=tf.clip_by_value(img/255.0,0.0,1.0)
        img01=apply_augmentations(img01,config)
        """
        Después de augmentar, volvemos a rango 0-255 para normalizar según el modo elegido. Esto es necesario porque a la hora de normalizar
        según imagenet, resnet, etc, los valores esperados son en rango 0-255, no 0-1. Podríamos pensar en normalizar aquí directamente, pero
        por ejemplo resnet o efficientet tienen sus propias funciones de preprocesado que esperan 0-255.
        """
        img=tf.clip_by_value(img01*255.0,0.0,255.0)
    img=normalize_image(img,config) # Ahora sí normalizamos
    return img


"""
Esta función realiza una serie de preparaciones para el correcto entrenamiento. Pues será la encargadad de designar a cada dataset un tipo de construcción distinta
basada en sus splits, labels y como no, si usará o no augmentaciones. Para ello tendrá primeramente que obtener el split de cada conjunto de datos. Adicionalmente, también
obtiene los pesos para el conjunto de entrenamiento. 
"""
def create_datasets(split_dir,config):
    datasets={}
    class_weights=None
    try:
        train_images,train_labels=load_split(split_dir,"train")
        val_images,val_labels=load_split(split_dir,"val")
        test_images,test_labels=load_split(split_dir,"test")
    except FileNotFoundError as e:
        raise RuntimeError(f"Error cargando los splits:{e}")
    # Comprobaciones básicas
    if not train_images:
        raise ValueError("El split de entrenamiento está vacío.")
    if not val_images:
        print("el split de validación está vacío.")
    if not test_images:
        print("el split de test está vacío.")

    print(f"Imágenes cargadas:"f"Train:{len(train_images)},"f"Val:{len(val_images)},"f"Test:{len(test_images)}")
    # Pesos de clase
    if config.get("use_class_weights")==True:
        class_weights=compute_class_weights(train_labels)
    else:
        print("Pesos de clase desactivados.")
    # Creación de datasets
    datasets["train"]=create_dataset(train_images,train_labels,True,config)
    datasets["val"]=create_dataset(val_images,val_labels,False,config)
    datasets["test"]=create_dataset(test_images,test_labels,False,config)
    return datasets,class_weights

"""
Esta función se encarga de construir adecuadamente cada conjunto de datos. Lo primero que hace es crear los datasets de las rutas y los valores de las etiquetas
en su versión tensorflow. Posteriormente une ambos datasets, ejemplo: 
("img1.png",0)
("img2.png",1)
Esto es todo lo que necesita tensorflow para poder entrenar. Ahora a continuación se decodifican las imágenes, pasando de el tensor de imagen a imagen real (paso necesario)
si buscamos hacer augmentaciones o normalizaciones (no se puede con tensores de tensorflow). Se hace antes que el preprocesado porque la imagen se carga justoa  tiempo y
evita cargar todo el dataset en memoria. A continuación preprocesamos y augmentamos si la flag training está a valor 'true'. Tras esto, barajamos (si procede) el dataset
de entrenamiento, muy importante si buscamos romper patrones de ocurrencia entre datos que puedan causar sesgos. En validación/test no se debe alterar el orden. Para terminar,
Cacheamos el dataset en memoria del tirón para que las épocas sean muy rápidas (NOTA DE ADVERTENCIA: SI EL ORDENADOR DISPONE DE POCA MEMORIA, NO ES RECOMENDABLE ACTIVAR ESTA FLAG: 
YA QUE PUEDE BLOQUEAR TODA LA MEMORIA DEL SISTEMA Y CONGELARLO, SOLO SI DISPONE DE ALTA MEMORIA). Finalmente hacemos batch en el dataset para agrupar datos para el entrenamiento/validacion:
ej:
(32, 224, 224, 3), (32,)
y por último prefetch, y repeat que simplemente prepara el siguiente batch cuando el modelo está entrenando uno ahora y ko vuelve infinito para que se repitan imágenes.
"""
def create_dataset(image_paths,labels,training,config):
    paths_ds=tf.data.Dataset.from_tensor_slices(image_paths)
    labels_ds=tf.data.Dataset.from_tensor_slices(labels)
    dataset=tf.data.Dataset.zip((paths_ds, labels_ds))
    dataset=dataset.map(lambda path,label:(decode_image(path),label),tf.data.AUTOTUNE)
    # AUTOTUNE: Tensorflow decide automáticamente cuantos hilos usar, para máximo rendimiento.
    dataset=dataset.map(lambda img,label:(preprocess_image(img,training,config),label),tf.data.AUTOTUNE)
    if training and config["shuffle_train"]:
        # buffer size = tamaño de la mezcla, min ( evitar suffles garndes)
        # seed = semilla de reproducibilidad, si seed = 42 entonces siempre haremos la misma mezcla
        dataset=dataset.shuffle(buffer_size=min(len(image_paths),config["buffer_size"]),seed=config["seed"],reshuffle_each_iteration=True)
    if config["cache"]:
        dataset=dataset.cache()
    dataset=dataset.batch(config["batch_size"])
    if training:
        dataset=dataset.repeat()
    if config["prefetch"]:
        dataset=dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

"""
Función para visualizar augmentaciones aplicadas a una imagen dada.
Muestra la imagen original y varias versiones augmentadas según la configuración dada.
Guarda la visualización en un archivo PNG con timestamp.
"""
def visualize_augmentations(image_path,config,num_samples):
    if not os.path.exists(image_path): # Control de errores por ruta no encontrada
        print(f"Imagen no encontrada:{image_path}")
        return
    original_img=decode_image(image_path) # Obtenemos la imagen del dataset tensorflow
    # Normalizamos según modo (no es neceasario para visualización pero es más preciso a lo que ve el modelo)
    original_img_norm=normalize_image(original_img,config["normalization_mode"]) 
    fig, axes=plt.subplots(1,num_samples+1,(15,3))
    orig_np=np.clip(original_img_norm.numpy(),0,1)
    axes[0].imshow(orig_np)
    axes[0].set_title("Original")
    axes[0].axis("off")
    # La decisión de diseño que he escogido es muestrear una sola imagen por el simple hecho de que 
    # si muestreamos varias, quizás no notemos bien las diferencias de augmentación. Pero si muestreamos
    # varias augmentaciones de la misma imagen, veremos los distintos augmentos claramente
    for i in range(num_samples): 
        img_aug=apply_augmentations(original_img_norm,config)
        img_aug_np=np.clip(img_aug.numpy(),0,1)
        axes[i+1].imshow(img_aug_np)
        axes[i+1].set_title(f"Aug{i+1}")
        axes[i+1].axis("off")

    plt.suptitle(f"Nivel de Augmentación:{config['augmentation_level']}",14)
    plt.tight_layout()
    return fig

"""
Esta función es la operadora de todo el script, recibe como parámetros un conjunto de valores de configuración, hace/delega a otras funciones para comprobar errores o 
recomendar parámetros (ej: Expert), imprime parámetros por pantalla y construye todos los conjuntos de datos listos para el entrenamiento.
"""
def create_breakhis_pipeline(split_dir,img_size,batch_size,buffer_size,augmentation_level
                             ,normalization_mode,use_class_weights,seed,cache,shuffle_train,prefetch):
    # Control por si escribimos Medio o MEdio o algo asi
    level_lower=augmentation_level.lower()
    norm_lower=normalization_mode.lower()
    config=DatasetConfig(img_size,batch_size,buffer_size,level_lower,norm_lower,seed,use_class_weights
                         ,cache,shuffle_train,prefetch)
    print("CONFIGURACIÓN DEL PIPELINE DE DATOS")
    print(f"Tamaño imagen:{img_size}")
    print(f"Batch size:{batch_size}")
    print(f"Augmentación:{level_lower.upper()}")
    print(f"Normalización:{norm_lower.upper()}")
    if use_class_weights:
        print(f"Pesos de clase:{'Si'}")
    else:
        print(f"Pesos de clase:{'No'}")
    print(f"Semilla:{seed}")
    # Conjuntos de datos listos para el entrenamiento
    datasets,class_weights =create_datasets(split_dir,config)
    print("RESUMEN FINAL DE DATASETS")
    for split_name,dataset in datasets.items():
        try:
            count=0
            for x in dataset:
                count=count + 1
            print(f"{split_name.upper()}:{count} batches")
        except Exception:
            print(f"{split_name.upper()}:Dataset creado")

    return datasets,class_weights

def parse_arguments():
    parser = argparse.ArgumentParser("Configura y lanza el pipeline de datos BreakHis.")
    parser.add_argument("--split-dir", default="splits", help="Directorio con train.json, val.json y test.json.")
    parser.add_argument("--img-size", type=int, nargs=2, default=[256, 256], metavar=("HEIGHT", "WIDTH"), help="Tamaño de imagen (alto ancho). Por defecto 256 256.")
    parser.add_argument("--batch-size", type=int, default=16, help="Tamaño de batch. Por defecto 16.")
    parser.add_argument("--buffer-size", type=int, default=BUFFER_DEFAULT, help=f"Tamaño de buffer para shuffle (default {BUFFER_DEFAULT}).")
    parser.add_argument("--augmentation-level", default="expert", help="Nivel de augmentación (default: expert).")
    parser.add_argument("--normalization-mode", default="imagenet", help="Modo de normalización (default: imagenet).")
    parser.add_argument("--use-class-weights", dest="use_class_weights", action="store_true", default=True, help="Usa pesos de clase (default True).")
    parser.add_argument("--no-class-weights", dest="use_class_weights", action="store_false", help="Desactiva los pesos de clase si se desea.")
    parser.add_argument("--cache", dest="cache", action="store_true", default=CACHE_DEFAULT, help="Activa cache del dataset (default True).")
    parser.add_argument("--no-cache", dest="cache", action="store_false", help="Desactiva cache del dataset.")
    parser.add_argument("--shuffle-train", dest="shuffle_train", action="store_true", default=SHUFFLE_DEFAULT, help="Baraja el dataset de train (default True).")
    parser.add_argument("--no-shuffle-train", dest="shuffle_train", action="store_false", help="No barajar el dataset de train.")
    parser.add_argument("--prefetch", dest="prefetch", action="store_true", default=PREFETCH_DEFAULT, help="Activa prefetch (default True).")
    parser.add_argument("--no-prefetch", dest="prefetch", action="store_false", help="Desactiva prefetch.")
    parser.add_argument("--seed", type=int, default=42, help="Semilla para reproducibilidad (default 42).")
    return parser.parse_args()


if __name__ == "__main__":
    # Obtenemos parámetros
    args=parse_arguments()
    SPLIT_DIR=args.split_dir  # Directorio con train.json, val.json, test.json
    IMG_SIZE=tuple(args.img_size)
    BATCH_SIZE=args.batch_size
    BUFFER_SIZE=args.buffer_size
    AUG_LEVEL=args.augmentation_level
    NORM_MODE=args.normalization_mode
    USE_CLASS_WEIGHTS=args.use_class_weights
    SEED=args.seed
    CACHE=args.cache
    SHUFFLE=args.shuffle_train
    PREFETCH=args.prefetch
    try: 
        datasets,class_weights=create_breakhis_pipeline(SPLIT_DIR,IMG_SIZE,BATCH_SIZE,BUFFER_SIZE,AUG_LEVEL
                                                        ,NORM_MODE,USE_CLASS_WEIGHTS,SEED,CACHE,SHUFFLE,PREFETCH)
        train_ds=datasets.get("train")
        if train_ds: 
            print("\nDebug para primer batch de entrenamiento:")
            for images,labels in train_ds.take(1):
                print(f"Forma imágenes: {images.shape}")
                print(f"Rango imágenes: [{tf.reduce_min(images):.3f},{tf.reduce_max(images):.3f}]")
                print(f"Labels (primeros 5): {labels.numpy()[:5]}")
                print(f"Distribución labels: {np.unique(labels.numpy(),True)}")

        if class_weights: # Si aplicamos class_weights
            print("\nPesos de clase para training:")
            for class_id,weight in class_weights.items():
                if class_id == 0:
                    print(f"Benigno:peso={weight:.3f}")
                else:
                    print(f"Maligno:peso={weight:.3f}")

        # Visualización de augmentaciones (solo si existe carpeta split y alguna imagen .png)
        example_images=glob.glob(os.path.join(SPLIT_DIR,"*.png"))
        if example_images:
            viz_config=DatasetConfig(IMG_SIZE,BATCH_SIZE,BUFFER_SIZE,AUG_LEVEL,NORM_MODE,SEED
                                     ,USE_CLASS_WEIGHTS,CACHE,SHUFFLE,PREFETCH)
            visualize_augmentations(example_images[0],viz_config,5)

    except Exception as e:
        print(f"Error creando pipeline: {e}")
        print(f"Tipo de error: {type(e).__name__}")
