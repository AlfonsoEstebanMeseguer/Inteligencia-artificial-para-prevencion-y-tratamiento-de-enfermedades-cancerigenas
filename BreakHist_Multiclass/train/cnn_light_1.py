import os
import sys
import argparse
from pathlib import Path
import tensorflow as tf
import numpy as np
from BreakHist_Multiclass.config.create_dataset import DatasetConfig
from BreakHist_Multiclass.utils.utils import (ensure_splits,get_datasets_basic,evaluate_multiclass,resolve_split_dir,
                                              plot_confusion_matrix,plot_metrics_bar,
                                              DEFAULT_TRAIN_SIZE,DEFAULT_VAL_SIZE,DEFAULT_TEST_SIZE,DEFAULT_RANDOM_STATE)
from BreakHist_Binary.src.utils.utils import plot_training_history
from BreakHist_Multiclass.models.models_definitions import build_cnn_light_1

# Ajuste de path para ejecución directa
if __name__=="__main__" and __package__ is None:
    project_root=Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0,str(project_root))

def parse_arguments():
    parser=argparse.ArgumentParser(description="Entrena cnn_light_1 (multiclase simple).")
    project_root=Path(__file__).resolve().parents[2]
    default_base=project_root/"BreakHist_Multiclass"/"BreakHist"/"data"/"BreakHis - Breast Cancer Histopathological Database" /"dataset_cancer_v1"/"dataset_cancer_v1"/"classificacao_multiclasse"
    parser.add_argument("--base-path",default=str(default_base),help="Ruta raíz del dataset.")
    default_splits = Path(__file__).resolve().parents[1]/"splits"
    parser.add_argument("--splits-dir",default=str(default_splits),help="Directorio de JSONs de split (por defecto en BreakHist_Multiclass/splits).")
    parser.add_argument("--img-size",type=int,nargs=2,default=[224,224],metavar=("H","W"),help="Tamaño de imagen.")
    parser.add_argument("--batch-size",type=int,default=32,help="Batch size.")
    parser.add_argument("--epochs",type=int,default=40,help="Número de épocas.")
    parser.add_argument("--lr",type=float,default=1e-3,help="Learning rate.")
    parser.add_argument("--use-class-weights",dest="use_class_weights",action="store_true",default=True,help="Usar pesos de clase (default True).")
    parser.add_argument("--no-class-weights",dest="use_class_weights",action="store_false",help="No usar pesos de clase.")
    parser.add_argument("--train-size",type=float,default=DEFAULT_TRAIN_SIZE,help=f"Proporción de entrenamiento (default {DEFAULT_TRAIN_SIZE}).")
    parser.add_argument("--val-size",type=float,default=DEFAULT_VAL_SIZE,help=f"Proporción de validación (default {DEFAULT_VAL_SIZE}).")
    parser.add_argument("--test-size",type=float,default=DEFAULT_TEST_SIZE,help=f"Proporción de test (default {DEFAULT_TEST_SIZE}).")
    parser.add_argument("--ensure-all-classes",dest="ensure_all_classes",action="store_true",default=True,help="Garantiza presencia de todas las clases en splits (default True).")
    parser.add_argument("--no-ensure-all-classes",dest="ensure_all_classes",action="store_false",help="No fuerza cobertura total de clases.")
    parser.add_argument("--split-mode",default="patient",choices=["patient","image"],help="patient (sin fuga) o image (con fuga).")
    parser.add_argument("--augmentation-level",default="none",choices=["none","low","medium","advanced","expert"],help="Nivel de augmentación para train.")
    parser.add_argument("--normalization-mode",default="standard",choices=["standard","efficientnet"],help="Normalización.")
    parser.add_argument("--seed",type=int,default=42,help="Semilla para reproducibilidad.")
    return parser.parse_args()

def main():
    args=parse_arguments()
    tf.keras.utils.set_random_seed(args.seed)
    np.random.seed(args.seed)
    model_dir=Path("models")/Path(__file__).stem
    os.makedirs(model_dir,exist_ok=True)
    config=DatasetConfig(tuple(args.img_size),args.batch_size,args.use_class_weights,args.normalization_mode,augmentation_level=args.augmentation_level,seed=args.seed)
    split_dir=resolve_split_dir(args.splits_dir,args.split_mode)
    ensure_splits(args.base_path,split_dir,train_size=args.train_size,val_size=args.val_size,test_size=args.test_size
                  ,split_mode=args.split_mode,dataset_type="multiclass",random_state=DEFAULT_RANDOM_STATE
                  ,ensure_all_classes=args.ensure_all_classes)
    
    ds_bundle=get_datasets_basic(config,split_dir,include_labels=True,dataset_type="multiclass")
    num_classes=ds_bundle["num_classes"]
    model=build_cnn_light_1(input_shape=(*config["img_size"],3),num_classes=num_classes)
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=["accuracy"])
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_accuracy",patience=8,restore_best_weights=True)
               ,tf.keras.callbacks.ModelCheckpoint(str(model_dir/"best.h5"),monitor="val_accuracy",mode="max",save_best_only=True,verbose=1)]

    history=model.fit(ds_bundle["train_ds"],validation_data=ds_bundle["val_ds"],epochs=args.epochs,steps_per_epoch=ds_bundle["steps_per_epoch"]
                      ,validation_steps=ds_bundle["val_steps"],class_weight=ds_bundle["class_weights"],callbacks=callbacks,verbose=1)
    
    plot_training_history(history)
    metrics,cm,report=evaluate_multiclass(model,ds_bundle)
    print("\nMétricas finales:",metrics)
    print("\nMatriz de confusión:\n",cm)
    print("\nClassification report:\n",report)
    plot_confusion_matrix(cm)
    plot_metrics_bar(metrics,title="Métricas finales (cnn_light_1)")
    model.save(str(model_dir/"final.h5"))
    print(f"Modelo guardado en {model_dir/'final.h5'}")

if __name__=="__main__":
    main()
