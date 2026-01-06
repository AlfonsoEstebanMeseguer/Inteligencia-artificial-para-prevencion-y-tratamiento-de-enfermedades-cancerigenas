import os
import sys
import argparse
from pathlib import Path
import tensorflow as tf
from BreakHist_Multiclass.config.create_dataset import DatasetConfig
from BreakHist_Multiclass.utils.utils import ensure_splits, get_datasets_basic, evaluate_multiclass, resolve_split_dir
from BreakHist_Multiclass.models.models_definitions import build_cnn_transfer_1

# Ajuste de path para ejecución directa
if __name__=="__main__" and __package__ is None:
    project_root=Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0,str(project_root))

def parse_arguments():
    parser=argparse.ArgumentParser(description="Transfer learning EfficientNetB0 multiclase.")
    project_root = Path(__file__).resolve().parents[2]  # .../BreakHist_Multiclass
    default_base = project_root / "BreakHist_Multiclass" / "BreakHist" / "data" / "BreakHis - Breast Cancer Histopathological Database" / "dataset_cancer_v1" / "dataset_cancer_v1" / "classificacao_multiclasse"
    parser.add_argument("--base-path",default=str(default_base),help="Ruta raíz del dataset.")
    default_splits = Path(__file__).resolve().parents[1] / "splits"
    parser.add_argument("--splits-dir",default=str(default_splits),help="Directorio de JSONs de split (por defecto en BreakHist_Multiclass/splits).")
    parser.add_argument("--img-size",type=int,nargs=2,default=[224,224],metavar=("H","W"),help="Tamaño de imagen.")
    parser.add_argument("--batch-size",type=int,default=32,help="Batch size.")
    parser.add_argument("--epochs-head",type=int,default=10,help="Épocas fase 1 (cabeza).")
    parser.add_argument("--epochs-ft",type=int,default=20,help="Épocas fase 2 (fine-tuning).")
    parser.add_argument("--lr-head",type=float,default=3e-4,help="LR fase 1.")
    parser.add_argument("--lr-ft",type=float,default=1e-5,help="LR fase 2.")
    parser.add_argument("--dropout",type=float,default=0.4,help="Dropout final.")
    parser.add_argument("--trainable-at",type=int,default=200,help="Índice de capa para comenzar a entrenar en EfficientNet.")
    parser.add_argument("--use-class-weights",dest="use_class_weights",action="store_true",default=True,help="Usar pesos de clase.")
    parser.add_argument("--no-class-weights",dest="use_class_weights",action="store_false",help="No usar pesos de clase.")
    parser.add_argument("--split-mode",default="patient",choices=["patient","image"],help="patient (sin fuga) o image (con fuga).")
    return parser.parse_args()

def main():
    args=parse_arguments()
    model_dir=Path("models")/Path(__file__).stem
    os.makedirs(model_dir,exist_ok=True)
    config=DatasetConfig(tuple(args.img_size),args.batch_size,args.use_class_weights,"efficientnet")
    split_dir=resolve_split_dir(args.splits_dir,args.split_mode)
    ensure_splits(args.base_path,split_dir,args.split_mode)
    ds_bundle=get_datasets_basic(config,split_dir,True)
    num_classes=ds_bundle["num_classes"]
    model,base=build_cnn_transfer_1((*config["img_size"],3),num_classes,args.dropout,args.trainable_at)
    # cabeza
    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr_head),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=["accuracy"])
    
    cbs1=[tf.keras.callbacks.ModelCheckpoint(str(model_dir/"best_head.weights.h5"),monitor="val_accuracy",mode="max",save_best_only=True,verbose=1,save_weights_only=True),
          tf.keras.callbacks.EarlyStopping(monitor="val_accuracy",patience=5,restore_best_weights=True)]
    
    model.fit(ds_bundle["train_ds"],validation_data=ds_bundle["val_ds"],
              epochs=args.epochs_head,steps_per_epoch=ds_bundle["steps_per_epoch"],
              validation_steps=ds_bundle["val_steps"],
              class_weight=ds_bundle["class_weights"],
              callbacks=cbs1,verbose=1)

    # fine-tuning
    base.trainable=True
    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr_ft),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=["accuracy"])
    
    cbs2=[tf.keras.callbacks.ModelCheckpoint(str(model_dir/"best_ft.weights.h5"),monitor="val_accuracy",mode="max",save_best_only=True,verbose=1,save_weights_only=True),
          tf.keras.callbacks.EarlyStopping(monitor="val_accuracy",patience=8,restore_best_weights=True),
          tf.keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy",factor=0.3,patience=3,min_lr=1e-6,verbose=1)]
    
    model.fit(ds_bundle["train_ds"],validation_data=ds_bundle["val_ds"],
              epochs=args.epochs_ft,steps_per_epoch=ds_bundle["steps_per_epoch"],
              validation_steps=ds_bundle["val_steps"],
              class_weight=ds_bundle["class_weights"],
              callbacks=cbs2,verbose=1)

    metrics,cm,report=evaluate_multiclass(model,ds_bundle)
    print("\nMétricas finales:",metrics)
    print("\nMatriz de confusión:\n",cm)
    print("\nClassification report:\n",report)
    try:
        from utils.utils import plot_confusion_matrix
        plot_confusion_matrix(cm)
    except Exception:
        pass

    model.save_weights(str(model_dir/"final.weights.h5"))
    print(f"Pesos guardados en {model_dir/'final.weights.h5'}")

if __name__=="__main__":
    main()
