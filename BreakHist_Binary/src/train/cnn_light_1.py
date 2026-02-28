import os
import math
import json
import sys
from pathlib import Path
import argparse  # Parametrización por CLI sin alterar la lógica
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, regularizers
from sklearn.metrics import (confusion_matrix,classification_report,roc_auc_score,average_precision_score)
from src.config.create_dataset import DatasetConfig, BUFFER_DEFAULT, SHUFFLE_DEFAULT, PREFETCH_DEFAULT, CACHE_DEFAULT
from src.config.readDataset import read_binary_breakhis_data
from src.config.split_dataset import split_by_patient
from src.utils.utils import ensure_splits,get_datasets_basic,run_eval_and_artifacts,resolve_split_dir,plot_training_history
from src.models.models_definitions import build_cnn_light

PROJECT_ROOT=Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0,str(PROJECT_ROOT))

# Permite ejecutar este script directamente (`python src/train/...py`) sin romper imports.
if __name__=="__main__" and __package__ is None:
    project_root=Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0,str(project_root))
else:
    project_root=Path(__file__).resolve().parents[2]

# Valores por defecto 
EPOCHS_DEFAULT=10
LR_DEFAULT=1e-3
IMG_SIZE_DEFAULT=[224,224]
BATCH_SIZE_DEFAULT=32
AUG_LEVEL_DEFAULT="none"
NORM_MODE_DEFAULT="imagenet"
USE_CLASS_WEIGHTS_DEFAULT=False
L2_REG_DEFAULT=5e-3
DROPOUT_DEFAULT=0.8
TRAIN_SIZE_DEFAULT=0.8
VAL_SIZE_DEFAULT=0.1
TEST_SIZE_DEFAULT=0.1
SEED=42
PATIENTE_STOPPING=2
PATIENTE_RL=2
MIN_LR=1e-6
FACTOR_RL=0.3

# Parser
def parse_arguments():
    parser = argparse.ArgumentParser(description="Entrena la CNN ligera (desde cero) para BreakHis.")
    project_root=Path(__file__).resolve().parents[2]
    default_base=project_root/"BreakHist"/"data"/"BreakHis - Breast Cancer Histopathological Database"/"dataset_cancer_v1"/"dataset_cancer_v1"/"classificacao_binaria"
    default_splits=project_root/"splits"
    parser.add_argument("--base-path",default=str(default_base),help="Ruta raíz del dataset BreakHis (binario).")
    parser.add_argument("--splits-dir",default=str(default_splits),help="Directorio donde se guardan/leen los JSON de splits.")
    parser.add_argument("--epochs",type=int,default=EPOCHS_DEFAULT,help=f"Número de épocas de entrenamiento (default {EPOCHS_DEFAULT}).")
    parser.add_argument("--lr",type=float,default=LR_DEFAULT,help=f"Learning rate para Adam (default {LR_DEFAULT}).")
    parser.add_argument("--img-size",type=int,nargs=2,default=IMG_SIZE_DEFAULT,metavar=("H","W"),help="Tamaño de imagen (alto ancho).")
    parser.add_argument("--batch-size",type=int,default=BATCH_SIZE_DEFAULT,help=f"Batch size (default {BATCH_SIZE_DEFAULT}).")
    parser.add_argument("--augmentation-level",default=AUG_LEVEL_DEFAULT,help=f"Nivel de augmentación (default {AUG_LEVEL_DEFAULT}).")
    parser.add_argument("--normalization-mode",default=NORM_MODE_DEFAULT,help=f"Modo de normalización (default {NORM_MODE_DEFAULT}).")
    parser.add_argument("--use-class-weights",dest="use_class_weights",action="store_true",default=USE_CLASS_WEIGHTS_DEFAULT,help=f"Usa pesos de clase (default {USE_CLASS_WEIGHTS_DEFAULT}).")
    parser.add_argument("--no-class-weights",dest="use_class_weights",action="store_false",help="Desactiva los pesos de clase si se desea.")
    parser.add_argument("--l2-reg",type=float,default=L2_REG_DEFAULT,help=f"Regularización L2 (default {L2_REG_DEFAULT}).")
    parser.add_argument("--dropout",type=float,default=DROPOUT_DEFAULT,help=f"Dropout en cabeza densa (default {DROPOUT_DEFAULT}).")
    parser.add_argument("--buffer-size",type=int,default=BUFFER_DEFAULT,help=f"Tamaño de buffer para shuffle (default {BUFFER_DEFAULT}).")
    parser.add_argument("--cache",dest="cache",action="store_true",default=CACHE_DEFAULT,help="Activa cache del dataset (default True).")
    parser.add_argument("--no-cache",dest="cache",action="store_false",help="Desactiva cache del dataset.")
    parser.add_argument("--shuffle-train",dest="shuffle_train",action="store_true",default=SHUFFLE_DEFAULT,help="Baraja el dataset de train (default True).")
    parser.add_argument("--no-shuffle-train",dest="shuffle_train",action="store_false",help="No barajar el dataset de train.")
    parser.add_argument("--prefetch",dest="prefetch",action="store_true",default=PREFETCH_DEFAULT,help="Activa prefetch (default True).")
    parser.add_argument("--no-prefetch",dest="prefetch",action="store_false",help="Desactiva prefetch.")
    parser.add_argument("--train-size",type=float,default=TRAIN_SIZE_DEFAULT,help=f"Proporción de entrenamiento (default {TRAIN_SIZE_DEFAULT}).")
    parser.add_argument("--val-size",type=float,default=VAL_SIZE_DEFAULT,help=f"Proporción de validación (default {VAL_SIZE_DEFAULT}).")
    parser.add_argument("--test-size",type=float,default=TEST_SIZE_DEFAULT,help=f"Proporción de test (default {TEST_SIZE_DEFAULT}).")
    parser.add_argument("--split-mode",default="patient",choices=["patient","image"],help="patient (sin fuga) o image (estratificado por clase, con fuga).")
    return parser.parse_args()

def main():
    # Leer argumentos sin modificar los valores por defecto originales
    args=parse_arguments()
    model_dir=project_root/"models"/Path(__file__).stem
    model_dir.mkdir(parents=True,exist_ok=True)
    # Chequeo de configuraciones y diccionario para construcción de dataset de entrenamiento
    config=DatasetConfig(tuple(args.img_size),args.batch_size,args.buffer_size,args.augmentation_level.lower()
                         ,args.normalization_mode.lower(),SEED,args.use_class_weights
                         ,args.cache,args.shuffle_train,args.prefetch)

    config["gradcam_fixed_path"]=str(project_root/"splits"/"gradcam_fixed.json")
    # Chequeo de splits
    split_dir=resolve_split_dir(args.splits_dir,args.split_mode)
    ensure_splits(args.base_path,split_dir,args.train_size,args.val_size,args.test_size,args.split_mode,dataset_type="binary",random_state=SEED,ensure_all_classes=False)
    # Obtenemos el dataset de TODOS los conjuntos de datos listos para ENTRENAR/VALIDAR
    ds_bundle=get_datasets_basic(config,split_dir,False,"binary")
    # Modelo 1
    model=build_cnn_light((*config["img_size"],3),args.l2_reg,args.dropout)
    """
    Adam es un optimizador que combina momentun (historial) y adaptación del learning rate(cuanto aprende el modelo por cada paso
    (grande -> saltos largos, pequeño -> aprende lento, pero seguro)) por parámetro. Es un optimizador que funciona muy bien para imágenes, 
    estandarizado para CNNs médicas. Converte más rápido y tiene menos tuning que otros compiladores como SGD. Binary Crossentropy para 
    salida binaria (sigmoide) (aproxima la predicción a 0 o a 1). Muy útil e imprescindible para este caso porque además castiga más cuanto la predicción es menos
    segura (valores intermedios entre 0-1). Si el modelo está muy seguro y se equivoca -> castigo grande, si duda y se equivoca -> castigo pequeño.

    Métricas: quitando las comentadas en evaluate_model, destacan SensitivityAtSpecificity(0.9) que mide la sensibilidad máxima cuando la especificidad es al menos 90%,
    SpecificityAtSensitivity(0.9) -> mide la especificidad máxima cuando la sensibilidad es al menos 90%.
    Resuelven las siguientes dudas:
    Si obligamos a acertar al menos el 90% de benignos,cuántos malignos detectamos?
    Si detectamos al menos el 90% de cánceres,cuántos sanos clasificamos bien?
    """
    metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"),tf.keras.metrics.Precision(name="precision")
             ,tf.keras.metrics.Recall(name="recall"),tf.keras.metrics.AUC(name="auc")
             ,tf.keras.metrics.AUC(name="prc",curve="PR"),tf.keras.metrics.SensitivityAtSpecificity(0.9,name="sens_at_spec90")
             ,tf.keras.metrics.SpecificityAtSensitivity(0.9,name="spec_at_sens90")]
    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr),loss="binary_crossentropy",metrics=metrics)
    model.summary()
    class_weights=ds_bundle["class_weights"]
    history=model.fit(ds_bundle["train_ds"],validation_data=ds_bundle["val_ds"],epochs=args.epochs
                      ,steps_per_epoch=ds_bundle["steps_per_epoch"],validation_steps=ds_bundle["val_steps"]
                      ,class_weight=class_weights,verbose=1)
    
    plot_training_history(history,None,None,True)
    run_eval_and_artifacts(model, ds_bundle, gradcam_dir=True,last_conv_layer_name="auto", save_path=str(model_dir/"cnn_light_breakhis.h5"),prefix="TEST",force_target_class=None)

if __name__=="__main__":
    main()
