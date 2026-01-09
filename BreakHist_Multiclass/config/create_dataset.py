import os
import sys
import argparse
import glob
from pathlib import Path
from typing import List, Tuple
import numpy as np
import tensorflow as tf
import json

# Reutilizamos utilidades binaria para no duplicar código
BINARY_CONFIG_DIR=Path(__file__).resolve().parents[2]/"BreakHist_Binary"/"src"/"config"
BINARY_SRC_DIR=Path(__file__).resolve().parents[2]/"BreakHist_Binary"
for p in [BINARY_CONFIG_DIR,BINARY_SRC_DIR]:
    sp=str(p)
    if sp not in sys.path:
        sys.path.insert(0,sp)
try:
    from create_dataset import load_split, decode_image,preprocess_image, create_datasets, create_dataset, compute_class_weights
    from src.config.augmentations import LOW, apply_augmentations, params
    from src.config.normalization import STANDARD, EFFICIENTNET, normalize_image
except ImportError as e:
    raise ImportError(f"No se pudo importar utilidades de dataset desde {BINARY_CONFIG_DIR}") from e

# Normalizaciones permitidas en multiclase: estándar y efficientnet
NORM_EFFNET = EFFICIENTNET
NORM_BASIC = STANDARD
IMGAGE_SIZE = [224,224]
BATCH_SIZE = 32
BUFFER_DEFAULT = 1000
CACHE_DEFAULT = True
SHUFFLE_DEFAULT = True
PREFETCH_DEFAULT = True
SEED_DEFAULT = 42
"""
Función de checkeo y configuración simplificado para el pipeline de datos multiclase.
En este caso no tenemos augmentaciones de distintos tipos porque decidí que para el dataset multiclase no era 
necesario complicarlo tanto ya que sabemos que no se puede hacer por clasificación multiclase y que ofrezca altos resultados
"""
def DatasetConfig(img_size,batch_size,use_class_weights,normalization_mode,
                  buffer_size=BUFFER_DEFAULT,cache=CACHE_DEFAULT,shuffle_train=SHUFFLE_DEFAULT,prefetch=PREFETCH_DEFAULT,seed=SEED_DEFAULT):
    norm=normalization_mode.lower()
    if norm not in (NORM_EFFNET.lower(),NORM_BASIC.lower()):
        raise ValueError(f"Modo de normalización '{normalization_mode}' no válido (usar {NORM_BASIC} o {NORM_EFFNET}) para multiclase")
    
    return {"img_size":tuple(img_size),"batch_size":batch_size,"use_class_weights":use_class_weights,
            "normalization_mode":norm,"augmentation_level":LOW,"augmentation_params":params.get(LOW),
            "buffer_size":buffer_size,"cache":cache,"shuffle_train":shuffle_train,"prefetch":prefetch,
            "seed":seed}

def parse_arguments():
    parser=argparse.ArgumentParser(description="Pipeline de datos multiclase simplificado.")
    parser.add_argument("--split-dir",default="splits",help="Directorio con train.json, val.json, test.json.")
    parser.add_argument("--img-size",type=int,nargs=2,default=IMGAGE_SIZE,metavar=("H","W"),help="Tamaño de imagen.")
    parser.add_argument("--batch-size",type=int,default=BATCH_SIZE,help="Batch size.")
    parser.add_argument("--use-class-weights",dest="use_class_weights",action="store_true",default=True,help="Usar pesos de clase.")
    parser.add_argument("--no-class-weights",dest="use_class_weights",action="store_false",help="No usar pesos de clase.")
    parser.add_argument("--normalization-mode",default=NORM_BASIC,help=f"Normalización: {NORM_BASIC} o {NORM_EFFNET}.")
    return parser.parse_args()

if __name__=="__main__":
    args=parse_arguments()
    cfg=DatasetConfig(img_size=tuple(args.img_size),batch_size=args.batch_size
                      ,use_class_weights=args.use_class_weights,normalization_mode=args.normalization_mode)
    ds,class_weights,images=create_datasets(args.split_dir,cfg)
    for split in ["train","val","test"]:
        print(f"{split}:{len(images[split])} imágenes")
    if class_weights:
        print("Pesos de clase:",class_weights)
