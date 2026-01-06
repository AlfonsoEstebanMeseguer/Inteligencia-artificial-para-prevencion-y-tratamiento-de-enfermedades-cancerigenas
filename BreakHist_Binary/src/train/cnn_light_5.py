import os
import math
import json
import sys
from pathlib import Path
import argparse  # Parametrización por CLI
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, regularizers
from sklearn.metrics import (confusion_matrix,classification_report,roc_auc_score,average_precision_score)
from src.config.create_dataset import DatasetConfig, BUFFER_DEFAULT, SHUFFLE_DEFAULT, PREFETCH_DEFAULT, CACHE_DEFAULT
from src.config.readDataset import read_binary_breakhis_data
from src.config.split_dataset import split_by_patient
from src.models.models_definitions import build_cnn5_densenet_se
from src.utils.utils import ensure_splits,get_datasets_basic,run_eval_and_artifacts,focal_loss,resolve_split_dir

# Defaults centralizados
EPOCHS_DEFAULT=120
BATCH_SIZE_DEFAULT=32
IMG_SIZE_DEFAULT=[224,224]
AUG_LEVEL_DEFAULT="advanced"
NORM_MODE_DEFAULT="imagenet"
USE_CLASS_WEIGHTS_DEFAULT=False
TRAIN_SIZE_DEFAULT=0.8
VAL_SIZE_DEFAULT=0.1
TEST_SIZE_DEFAULT=0.1
INITIAL_LR_DEFAULT=3e-4
THRESHOLD_DEFAULT=0.5
ES_PATIENCE=15
SEED=42
# Permite ejecutar este script directamente (`python src/train/...py`) sin romper imports.
if __name__=="__main__" and __package__ is None:
    project_root=Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0,str(project_root))

def parse_arguments():
    """
    Hiperparámetros expuestos por CLI manteniendo los defaults actuales.
    """
    default_base = Path(__file__).resolve().parents[2] / "BreakHist" / "data" / "BreakHis - Breast Cancer Histopathological Database" / "dataset_cancer_v1" / "dataset_cancer_v1" / "classificacao_binaria"
    default_splits = Path(__file__).resolve().parents[2] / "splits"
    parser=argparse.ArgumentParser(description="Entrena la CNN #5 (DenseNet-SE + Focal loss) para BreakHis.")
    parser.add_argument("--base-path",default=str(default_base),help="Ruta raíz del dataset BreakHis (binario).")
    parser.add_argument("--splits-dir",default=str(default_splits),help="Directorio de JSONs de split (por defecto el de BreakHist_Binary/splits).")
    parser.add_argument("--epochs",type=int,default=EPOCHS_DEFAULT,help=f"Número de épocas (default {EPOCHS_DEFAULT}).")
    parser.add_argument("--batch-size",type=int,default=BATCH_SIZE_DEFAULT,help=f"Batch size (default {BATCH_SIZE_DEFAULT}).")
    parser.add_argument("--img-size",type=int,nargs=2,default=IMG_SIZE_DEFAULT,metavar=("H","W"),help="Tamaño de imagen (alto ancho).")
    parser.add_argument("--augmentation-level",default=AUG_LEVEL_DEFAULT,help=f"Nivel de augmentación (default {AUG_LEVEL_DEFAULT}).")
    parser.add_argument("--normalization-mode",default=NORM_MODE_DEFAULT,help=f"Modo de normalización (default {NORM_MODE_DEFAULT}).")
    parser.add_argument("--use-class-weights",dest="use_class_weights",action="store_true",default=USE_CLASS_WEIGHTS_DEFAULT,help=f"Activa pesos de clase (default {USE_CLASS_WEIGHTS_DEFAULT}).")
    parser.add_argument("--no-class-weights",dest="use_class_weights",action="store_false",help="Desactiva pesos de clase.")
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
    parser.add_argument("--initial-lr",type=float,default=INITIAL_LR_DEFAULT,help=f"Learning rate inicial para CosineDecay (default {INITIAL_LR_DEFAULT}).")
    parser.add_argument("--threshold",type=float,default=THRESHOLD_DEFAULT,help=f"Umbral explícito para binarizar y_prob en test (default {THRESHOLD_DEFAULT}).")
    parser.add_argument("--split-mode",default="patient",choices=["patient","image"],help="patient (sin fuga) o image (estratificado por clase, con fuga).")
    return parser.parse_args()

def main():
    args=parse_arguments()

    # Config: subimos augment a ADVANCED para mejorar generalización
    config=DatasetConfig((args.img_size,args.img_size),args.batch_size,args.buffer_size,args.augmentation_level.lower()
                           ,args.normalization_mode.lower(),SEED,args.use_class_weights
                           ,args.cache,args.shuffle_train,args.prefetch)
    
    model_dir=Path("models")/Path(__file__).stem
    os.makedirs(model_dir,True)
    split_dir=resolve_split_dir(args.splits_dir,args.split_mode)
    ensure_splits(args.base_path,split_dir,args.train_size,args.val_size,args.test_size,args.split_mode)
    ds_bundle = get_datasets_basic(config,split_dir,True)
    pos = ds_bundle["train_labels"].mean() # suma de valores / numero de valores, solo hay unos porque 0 no suman valor
    # entonces es sumas 1s / num 0s + num 1s
    alpha=float(1.0-pos) # invertimos porque antes teníamos proporcion de negativos, el contrario será de positivos
    alpha=min(max(alpha,0.10),0.90) # umbralizamos para obtener alfa (peso que le daremos a cada clase)
    model=build_cnn5_densenet_se((*config["img_size"],3),24,(6,10,14),0.5,1e-4,0.2)
    # Optimizador: Adam con Cosine Decay (TF 2.10 no soporta weight_decay en Adam)
    total_steps=ds_bundle["steps_per_epoch"]*args.epochs # necesario para CosineDecay porque necesita saber cuantos pasos totales ten drá el entrenamiento
    lr_schedule=tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=args.initial_lr,decay_steps=total_steps)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule,clipnorm=1.0)
    model.compile(optimizer=optimizer,loss=lambda y_true,y_pred: focal_loss(y_true,y_pred,alpha=alpha,gamma=2.0,label_smoothing=0.05),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"),tf.keras.metrics.Precision(name="precision")
                 ,tf.keras.metrics.Recall(name="recall")
                 ,tf.keras.metrics.AUC(name="auc"),tf.keras.metrics.AUC(name="prc", curve="PR")
                 ,tf.keras.metrics.SensitivityAtSpecificity(0.9,name="sens_at_spec90")
                 ,tf.keras.metrics.SpecificityAtSensitivity(0.9,name="spec_at_sens90")])

    model.summary()
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_auc",mode="max",patience=ES_PATIENCE,restore_best_weights=True)
                 ,tf.keras.callbacks.ModelCheckpoint(str(model_dir/"cnn5_densenet_se_best.h5"),monitor="val_auc",mode="max",save_best_only=True,verbose=1)]

    history=model.fit(ds_bundle["train_ds"],validation_data=ds_bundle["val_ds"],epochs=args.epochs,steps_per_epoch=ds_bundle["steps_per_epoch"],validation_steps=ds_bundle["val_steps"]
                        ,class_weight=ds_bundle["class_weights"],callbacks=callbacks,verbose=1)

    run_eval_and_artifacts(model,ds_bundle,args.threshold,None,str(model_dir/"cnn5_densenet_se_predictions.npz")
                            ,True,"last_conv",str(model_dir/"cnn5_densenet_se_final.h5"))

if __name__=="__main__":
    main()
