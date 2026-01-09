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
from src.models.models_definitions import build_cnn4_residual
from src.utils.utils import ensure_splits,get_datasets_basic,run_eval_and_artifacts,resolve_split_dir,plot_training_history

# Defaults centralizados
EPOCHS_DEFAULT=70
LR_DEFAULT=1e-3
IMG_SIZE_DEFAULT=[224,224]
BATCH_SIZE_DEFAULT=32
AUG_LEVEL_DEFAULT="medium"
NORM_MODE_DEFAULT="imagenet"
USE_CLASS_WEIGHTS_DEFAULT=True
TRAIN_SIZE_DEFAULT=0.8
VAL_SIZE_DEFAULT=0.1
TEST_SIZE_DEFAULT=0.1
THRESHOLD_DEFAULT=0.5
ES_PATIENCE=12
RL_PATIENCE=6
RL_FACTOR=0.3
RL_MIN_LR=1e-6
SEED=42
# Permite ejecutar este script directamente (`python src/train/...py`) sin romper imports.
PROJECT_ROOT=Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0,str(PROJECT_ROOT))

def parse_arguments():
    """
    Hiperparámetros expuestos por CLI manteniendo los defaults originales.
    """
    parser=argparse.ArgumentParser(description="Entrena la CNN #4 (Residual / ResNet-18 like) para BreakHis.")
    project_root=Path(__file__).resolve().parents[2]
    default_base=project_root/"BreakHist"/"data"/"BreakHis - Breast Cancer Histopathological Database"/"dataset_cancer_v1"/"dataset_cancer_v1"/"classificacao_binaria"
    default_splits=project_root/"splits"
    parser.add_argument("--base-path",default=str(default_base),help="Ruta raíz del dataset BreakHis (binario).")
    parser.add_argument("--splits-dir",default=str(default_splits),help="Directorio de JSONs de split.")
    parser.add_argument("--epochs",type=int,default=EPOCHS_DEFAULT,help=f"Épocas de entrenamiento (default {EPOCHS_DEFAULT}).")
    parser.add_argument("--lr",type=float,default=LR_DEFAULT,help=f"Learning rate para Adam (default {LR_DEFAULT}).")
    parser.add_argument("--img-size",type=int,nargs=2,default=IMG_SIZE_DEFAULT,metavar=("H","W"),help="Tamaño de imagen (alto ancho).")
    parser.add_argument("--batch-size",type=int,default=BATCH_SIZE_DEFAULT,help=f"Batch size (default {BATCH_SIZE_DEFAULT}).")
    parser.add_argument("--augmentation-level",default=AUG_LEVEL_DEFAULT,help=f"Nivel de augmentación (default {AUG_LEVEL_DEFAULT}).")
    parser.add_argument("--normalization-mode",default=NORM_MODE_DEFAULT,help=f"Modo de normalización (default {NORM_MODE_DEFAULT}).")
    parser.add_argument("--use-class-weights",dest="use_class_weights",action="store_true",default=USE_CLASS_WEIGHTS_DEFAULT,help=f"Usar pesos de clase (default {USE_CLASS_WEIGHTS_DEFAULT}).")
    parser.add_argument("--no-class-weights",dest="use_class_weights",action="store_false",help="Desactiva los pesos de clase si se desea.")
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
    parser.add_argument("--threshold",type=float,default=THRESHOLD_DEFAULT,help=f"Umbral explícito para binarizar y_prob (default {THRESHOLD_DEFAULT}).")
    parser.add_argument("--split-mode",default="patient",choices=["patient","image"],help="patient (sin fuga) o image (estratificado por clase, con fuga).")
    return parser.parse_args()

def main():
    args=parse_arguments()
    model_dir=PROJECT_ROOT/"models"/Path(__file__).stem
    model_dir.mkdir(parents=True,exist_ok=True)
    config=DatasetConfig(tuple(args.img_size),args.batch_size,args.buffer_size,args.augmentation_level.lower()
                         ,args.normalization_mode.lower(),SEED,args.use_class_weights
                         ,args.cache,args.shuffle_train,args.prefetch)

    split_dir=resolve_split_dir(args.splits_dir,args.split_mode)
    ensure_splits(args.base_path,split_dir,args.train_size,args.val_size,args.test_size,args.split_mode)
    ds_bundle=get_datasets_basic(config,split_dir)
    model=build_cnn4_residual((*config["img_size"],3),1,1e-4,0.4)
    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr),loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"),tf.keras.metrics.Precision(name="precision"),tf.keras.metrics.Recall(name="recall")
                 ,tf.keras.metrics.AUC(name="auc"),tf.keras.metrics.AUC(name="prc",curve="PR"),tf.keras.metrics.SensitivityAtSpecificity(0.9,name="sens_at_spec90")
                 ,tf.keras.metrics.SpecificityAtSensitivity(0.9,name="spec_at_sens90")])

    model.summary()
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=ES_PATIENCE,restore_best_weights=True)
                 ,tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",factor=RL_FACTOR,patience=RL_PATIENCE, min_lr=RL_MIN_LR)
                 ,tf.keras.callbacks.ModelCheckpoint(str(model_dir/"cnn4_residual_best.h5"),monitor="val_loss",save_best_only=True,verbose=1)]
    
    history=model.fit(ds_bundle["train_ds"],validation_data=ds_bundle["val_ds"],epochs=args.epochs
                      ,steps_per_epoch=ds_bundle["steps_per_epoch"],validation_steps=ds_bundle["val_steps"]
                      ,class_weight=ds_bundle["class_weights"],callbacks=callbacks,verbose=1)

    plot_training_history(history)

    run_eval_and_artifacts(
        model,
        ds_bundle,
        args.threshold,
        npz_path=str(model_dir/"cnn4_residual_predictions.npz"),
        gradcam_dir=True,
        last_conv_layer_name="last_conv",
        save_path=str(model_dir/"cnn4_residual_final.h5"),
    )

if __name__=="__main__":
    main()
