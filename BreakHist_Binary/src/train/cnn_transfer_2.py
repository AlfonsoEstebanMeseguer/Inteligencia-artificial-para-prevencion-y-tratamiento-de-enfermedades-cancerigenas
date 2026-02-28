import os, math, json
import sys
from pathlib import Path
import argparse  # Parametrización CLI sin alterar la lógica
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from src.config.create_dataset import DatasetConfig, BUFFER_DEFAULT, SHUFFLE_DEFAULT, PREFETCH_DEFAULT, CACHE_DEFAULT
from src.config.readDataset import read_binary_breakhis_data
from src.config.split_dataset import split_by_patient
from src.models.models_definitions import build_efficientnetb0_transfer
from src.utils.utils import ensure_splits,get_datasets_basic,run_eval_and_artifacts,resolve_split_dir,plot_training_history

# Defaults centralizados
IMG_SIZE_DEFAULT=[224,224]
BATCH_SIZE_DEFAULT=32
AUG_LEVEL_DEFAULT="advanced"
NORM_MODE_DEFAULT="efficientnet"
USE_CLASS_WEIGHTS_DEFAULT=True
TRAIN_SIZE_DEFAULT=0.8
VAL_SIZE_DEFAULT=0.1
TEST_SIZE_DEFAULT=0.1
EPOCHS_HEAD_DEFAULT=12
EPOCHS_FT_DEFAULT=30
LR_HEAD_DEFAULT=3e-4
LR_FT_DEFAULT=5e-6
DROPOUT_DEFAULT=0.45
LABEL_SMOOTHING_DEFAULT=0.0
ES_HEAD_PATIENCE=8
ES_FT_PATIENCE=10
RL_PATIENCE=4
RL_FACTOR=0.3
RL_MIN_LR=1e-7
SEED=42
# Permite ejecutar este script directamente (`python src/train/...py`) sin romper imports.
PROJECT_ROOT=Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0,str(PROJECT_ROOT))

def parse_arguments():
    """
    Hiperparámetros expuestos vía CLI manteniendo los defaults originales.
    """
    default_base = Path(__file__).resolve().parents[2] / "BreakHist" / "data" / "BreakHis - Breast Cancer Histopathological Database" / "dataset_cancer_v1" / "dataset_cancer_v1" / "classificacao_binaria"
    default_splits = Path(__file__).resolve().parents[2] / "splits"
    parser=argparse.ArgumentParser(description="Transfer learning con EfficientNetB0 sobre BreakHis (binario).")
    parser.add_argument("--base-path",default=str(default_base),help="Ruta raíz del dataset BreakHis (binario).")
    parser.add_argument("--splits-dir",default=str(default_splits),help="Directorio de JSONs de split.")
    parser.add_argument("--img-size",type=int,nargs=2,default=IMG_SIZE_DEFAULT,metavar=("H", "W"),help="Tamaño de imagen (alto ancho).")
    parser.add_argument("--batch-size",type=int,default=BATCH_SIZE_DEFAULT,help=f"Batch size (default {BATCH_SIZE_DEFAULT}).")
    parser.add_argument("--augmentation-level",default=AUG_LEVEL_DEFAULT,help=f"Nivel de augmentación (default {AUG_LEVEL_DEFAULT}).")
    parser.add_argument("--normalization-mode",default=NORM_MODE_DEFAULT,help=f"Modo de normalización (default {NORM_MODE_DEFAULT}).")
    parser.add_argument("--use-class-weights",dest="use_class_weights",action="store_true",default=USE_CLASS_WEIGHTS_DEFAULT,help=f"Usar pesos de clase (default {USE_CLASS_WEIGHTS_DEFAULT}).")
    parser.add_argument("--no-class-weights",dest="use_class_weights",action="store_false",help="Desactiva pesos de clase si se desea.")
    parser.add_argument("--train-size",type=float,default=TRAIN_SIZE_DEFAULT,help=f"Proporción de entrenamiento (default {TRAIN_SIZE_DEFAULT}).")
    parser.add_argument("--val-size",type=float,default=VAL_SIZE_DEFAULT,help=f"Proporción de validación (default {VAL_SIZE_DEFAULT}).")
    parser.add_argument("--test-size",type=float,default=TEST_SIZE_DEFAULT,help=f"Proporción de test (default {TEST_SIZE_DEFAULT}).")
    parser.add_argument("--epochs-head",type=int,default=EPOCHS_HEAD_DEFAULT,help=f"Épocas fase 1 (default {EPOCHS_HEAD_DEFAULT}).")
    parser.add_argument("--epochs-ft",type=int,default=EPOCHS_FT_DEFAULT,help=f"Épocas fase 2 (default {EPOCHS_FT_DEFAULT}).")
    parser.add_argument("--lr-head",type=float,default=LR_HEAD_DEFAULT,help=f"Learning rate fase 1 (default {LR_HEAD_DEFAULT}).")
    parser.add_argument("--lr-ft",type=float,default=LR_FT_DEFAULT,help=f"Learning rate fase 2 (default {LR_FT_DEFAULT}).")
    parser.add_argument("--dropout",type=float,default=DROPOUT_DEFAULT,help=f"Dropout en la cabeza (default {DROPOUT_DEFAULT}).")
    parser.add_argument("--buffer-size",type=int,default=BUFFER_DEFAULT,help=f"Tamaño de buffer para shuffle (default {BUFFER_DEFAULT}).")
    parser.add_argument("--cache",dest="cache",action="store_true",default=CACHE_DEFAULT,help="Activa cache del dataset (default True).")
    parser.add_argument("--no-cache",dest="cache",action="store_false",help="Desactiva cache del dataset.")
    parser.add_argument("--shuffle-train",dest="shuffle_train",action="store_true",default=SHUFFLE_DEFAULT,help="Baraja el dataset de train (default True).")
    parser.add_argument("--no-shuffle-train",dest="shuffle_train",action="store_false",help="No barajar el dataset de train.")
    parser.add_argument("--prefetch",dest="prefetch",action="store_true",default=PREFETCH_DEFAULT,help="Activa prefetch (default True).")
    parser.add_argument("--no-prefetch",dest="prefetch",action="store_false",help="Desactiva prefetch.")
    parser.add_argument("--split-mode",default="patient",choices=["patient","image"],help="patient (sin fuga) o image (estratificado por clase, con fuga).")
    return parser.parse_args()

def main():
    args=parse_arguments()

    # normalizador con preprocess_input(efficientnet)
    config=DatasetConfig(tuple(args.img_size),args.batch_size,args.buffer_size,args.augmentation_level.lower()
                         ,args.normalization_mode.lower(),SEED,args.use_class_weights,args.cache
                         ,args.shuffle_train,args.prefetch)
    config["gradcam_fixed_path"]=str(PROJECT_ROOT/"splits"/"gradcam_fixed.json")
    
    model_dir=PROJECT_ROOT/"models"/Path(__file__).stem
    model_dir.mkdir(parents=True,exist_ok=True)
    split_dir=resolve_split_dir(args.splits_dir,args.split_mode)
    ensure_splits(args.base_path,split_dir,args.train_size,args.val_size,args.test_size,args.split_mode,dataset_type="binary",random_state=SEED,ensure_all_classes=False)
    ds=get_datasets_basic(config,split_dir,False,"binary")
    model,base=build_efficientnetb0_transfer((*config["img_size"],3),args.dropout)
    model.summary()
    # entrenar cabeza
    opt1=tf.keras.optimizers.Adam(learning_rate=args.lr_head,clipnorm=1.0)
    loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=LABEL_SMOOTHING_DEFAULT)
    model.compile(
        optimizer=opt1,loss=loss,
        metrics=[tf.keras.metrics.AUC(name="auc"),tf.keras.metrics.AUC(name="prc",curve="PR")
                 ,tf.keras.metrics.BinaryAccuracy(name="acc"),tf.keras.metrics.Recall(name="recall"),tf.keras.metrics.Precision(name="precision")
                 ,tf.keras.metrics.SensitivityAtSpecificity(0.9,name="sens_at_spec90"),tf.keras.metrics.SpecificityAtSensitivity(0.9,name="spec_at_sens90")])
    
    callbacks=[tf.keras.callbacks.ModelCheckpoint(str(model_dir/"cnn7_effnetb0_best.weights.h5"),monitor="val_auc",mode="max",save_best_only=True,save_weights_only=True, verbose=1),
         tf.keras.callbacks.EarlyStopping(monitor="val_auc",mode="max",patience=ES_HEAD_PATIENCE,restore_best_weights=True)]

    history_head=model.fit(ds["train_ds"],validation_data=ds["val_ds"],epochs=args.epochs_head,steps_per_epoch=ds["steps_per_epoch"],validation_steps=ds["val_steps"]
              ,class_weight=ds["class_weights"],callbacks=callbacks,verbose=1)
    plot_training_history(history_head,None,None,True)

    # fine-tuning 
    base.trainable=True
    # Congelar casi todo y afinar solo top
    for layer in base.layers[:-45]:
        layer.trainable=False
    model.summary()

    opt2=tf.keras.optimizers.Adam(learning_rate=args.lr_ft,clipnorm=1.0)
    model.compile(optimizer=opt2,loss=loss,metrics=[tf.keras.metrics.AUC(name="auc"),tf.keras.metrics.AUC(name="prc",curve="PR"),tf.keras.metrics.BinaryAccuracy(name="acc")
                                                    ,tf.keras.metrics.Recall(name="recall"),tf.keras.metrics.Precision(name="precision")
                                                    ,tf.keras.metrics.SensitivityAtSpecificity(0.9,name="sens_at_spec90"),tf.keras.metrics.SpecificityAtSensitivity(0.9,name="spec_at_sens90")])

    callbacks2=[tf.keras.callbacks.ModelCheckpoint(str(model_dir/"cnn7_effnetb0_finetuned_best.weights.h5"),monitor="val_auc",mode="max",save_best_only=True,save_weights_only=True,verbose=1)
            ,tf.keras.callbacks.EarlyStopping(monitor="val_auc",mode="max",patience=ES_FT_PATIENCE,restore_best_weights=True)
            ,tf.keras.callbacks.ReduceLROnPlateau(monitor="val_auc",mode="max",factor=RL_FACTOR,patience=RL_PATIENCE,min_lr=RL_MIN_LR,verbose=1)]

    history_ft=model.fit(ds["train_ds"],validation_data=ds["val_ds"],epochs=args.epochs_ft,steps_per_epoch=ds["steps_per_epoch"]
              ,validation_steps=ds["val_steps"]
              ,class_weight=ds["class_weights"]
              ,callbacks=callbacks2,verbose=1)
    
    plot_training_history(history_ft,None,None,True)
    run_eval_and_artifacts(model,ds,gradcam_dir=True,last_conv_layer_name="auto",save_path=str(model_dir/"cnn7_effnetb0_transfer_final.weights.h5"),prefix="TEST",force_target_class=None)

if __name__ == "__main__":
    main()
