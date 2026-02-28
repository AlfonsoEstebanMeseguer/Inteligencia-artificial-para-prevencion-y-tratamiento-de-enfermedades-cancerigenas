import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf

project_root=Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0,str(project_root))

from BRACS.bracs_models import build_bracs_cnn_transfer_1
from BRACS.config_bracs import (DEFAULT_BASE
                                ,compute_multiclass_metrics,compute_weights,load_splits_from_dir,make_dataset
                                ,plot_confusion_matrix_gray,plot_eval_metrics,plot_split_distribution,plot_training_history_all
                                ,read_bracs_dataset,save_splits_to_dir,split_by_patient)

# Ruta explícita al dataset redimensionado de BRACS.
DEFAULT_BASE_RESIZED=Path(__file__).resolve().parents[1]/"BRACS"/"BRACS_RoI"/"latest_version_resized"
if not DEFAULT_BASE_RESIZED.exists():
    DEFAULT_BASE_RESIZED=Path(DEFAULT_BASE)

def _json_default(obj):
    if isinstance(obj,np.ndarray):
        return obj.tolist()
    if isinstance(obj,(np.floating,np.integer)):
        return obj.item()
    if isinstance(obj,tf.Tensor):
        return obj.numpy().tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def format_confusion_matrix(cm,class_names):
    header="\t"+"\t".join(class_names)
    rows=[header]
    for i,row in enumerate(cm):
        rows.append(f"{class_names[i]}\t"+"\t".join(f"{v:3d}" for v in row))
    return "\n".join(rows)

# Defaults más simples (similar a BreakHist)
IMG_SIZE_DEFAULT=256
BATCH_SIZE_DEFAULT=24
TRAIN_SIZE_DEFAULT=0.8
VAL_SIZE_DEFAULT=0.1
TEST_SIZE_DEFAULT=0.1
EPOCHS_HEAD_DEFAULT=10
EPOCHS_FT_DEFAULT=60
LR_HEAD_DEFAULT=3e-4
LR_FT_DEFAULT=5e-5
DROPOUT_DEFAULT=0.4
TRAINABLE_AT_DEFAULT=150
SHUFFLE_BUFFER_DEFAULT=8000
USE_CLASS_WEIGHTS_DEFAULT=True
USE_CACHE_DEFAULT=False
EARLYSTOP_PATIENCE_DEFAULT=16
REDUCE_PATIENCE_DEFAULT=6
SEED_DEFAULT=42

def parse_args():
    parser=argparse.ArgumentParser("Entrena bracs_cnn_transfer_1 (EfficientNet con transfer learning).")
    parser.add_argument("--base-dir",type=str,default=str(DEFAULT_BASE_RESIZED),help="Raíz del dataset BRACS (por defecto latest_version_resized).")
    parser.add_argument("--train-size",type=float,default=TRAIN_SIZE_DEFAULT,help="Proporción train (resto val/test).")
    parser.add_argument("--val-size",type=float,default=VAL_SIZE_DEFAULT,help="Proporción val.")
    parser.add_argument("--test-size",type=float,default=TEST_SIZE_DEFAULT,help="Proporción test.")
    parser.add_argument("--seed",type=int,default=SEED_DEFAULT,help="Semilla.")
    parser.add_argument("--batch-size",type=int,default=BATCH_SIZE_DEFAULT,help="Batch size.")
    parser.add_argument("--image-size",type=int,default=IMG_SIZE_DEFAULT,help="Tamaño cuadrado de entrada.")
    parser.add_argument("--head-epochs",type=int,default=EPOCHS_HEAD_DEFAULT,help="Épocas de entrenamiento del head.")
    parser.add_argument("--finetune-epochs",type=int,default=EPOCHS_FT_DEFAULT,help="Épocas de fine-tuning.")
    parser.add_argument("--lr-head",type=float,default=LR_HEAD_DEFAULT,help="Learning rate fase head.")
    parser.add_argument("--lr-ft",type=float,default=LR_FT_DEFAULT,help="Learning rate fase fine-tuning.")
    parser.add_argument("--trainable-at",type=int,default=TRAINABLE_AT_DEFAULT,help="Índice de capa a partir del cual se descongela.")
    parser.add_argument("--dropout",type=float,default=DROPOUT_DEFAULT,help="Dropout en la cabeza densa.")
    parser.add_argument("--cache",action="store_true",default=USE_CACHE_DEFAULT,help="Usar cache() en tf.data.")
    parser.add_argument("--no-cache",dest="cache",action="store_false")
    parser.add_argument("--shuffle-buffer",type=int,default=SHUFFLE_BUFFER_DEFAULT,help="Buffer de shuffle para train.")
    parser.add_argument("--use-class-weights",dest="use_class_weights",action="store_true",default=USE_CLASS_WEIGHTS_DEFAULT)
    parser.add_argument("--no-class-weights",dest="use_class_weights",action="store_false")
    parser.add_argument("--earlystop-patience",type=int,default=EARLYSTOP_PATIENCE_DEFAULT,help="Paciencia de EarlyStopping.")
    parser.add_argument("--reduce-patience",type=int,default=REDUCE_PATIENCE_DEFAULT,help="Paciencia de ReduceLROnPlateau.")
    parser.add_argument("--splits-file",type=str,default=str(Path(__file__).resolve().parents[1]/"splits"),help="Carpeta con JSON de split.")
    parser.add_argument("--overwrite-splits",action="store_true",help="Regenera los JSON de split aunque ya existan.")
    parser.add_argument("--output-dir",type=str,default="BRACS/models/bracs_cnn_transfer_1",help="Ruta para checkpoints y reportes.")
    return parser.parse_args()

def prepare_data(args):
    splits_dir=Path(args.splits_file)
    split_train_path=splits_dir/"split_train.json" 
    # Accedemos a los splits y comprobamos que si ya existe un split entrenable, 
    # fue generado para la misma base_dir. Si no, se evita cargar un split incompatible y 
    # se pide regenerarlo o usar otro. 
    if split_train_path.exists() and not args.overwrite_splits:
        try:
            with open(split_train_path,"r",encoding="utf-8") as f:
                payload=json.load(f)
            meta_base=payload.get("meta",{}).get("base_dir")
        except Exception:
            meta_base=None
        if meta_base is not None:
            cur=os.path.normcase(os.path.abspath(args.base_dir))
            prev=os.path.normcase(os.path.abspath(meta_base))
            if cur!=prev:
                raise SystemExit(f"El split existente en {splits_dir} fue generado para base_dir={meta_base}, pero has pasado --base-dir {args.base_dir}. Usa --overwrite-splits o un --splits-file distinto.")
        print(f"Cargando split desde {splits_dir}")
        splits,label_map=load_splits_from_dir(splits_dir)
    else:
        print(f"No se encontró split, generando en {splits_dir}")
        _,images,labels,patients,label_map=read_bracs_dataset(Path(args.base_dir),verbose=True)
        splits=split_by_patient(images,labels,patients,args.train_size,args.val_size,args.test_size,args.seed)
        save_splits_to_dir(splits,label_map,splits_dir,Path(args.base_dir))

    num_classes=len(label_map)
    class_names=[]
    i=0
    while i<num_classes:
        for cls in label_map:
            if label_map[cls]==i:
                class_names.append(cls)
                break
        i+=1
    # Creamos los tf.data.Dataset para cada split
    train_ds=make_dataset(splits["train"]["images"],splits["train"]["labels"],args.image_size,args.batch_size,training=True,cache=args.cache,shuffle_buffer=args.shuffle_buffer)
    val_ds=make_dataset(splits["val"]["images"],splits["val"]["labels"],args.image_size,args.batch_size,training=False,cache=args.cache,shuffle_buffer=args.shuffle_buffer)
    test_ds=make_dataset(splits["test"]["images"],splits["test"]["labels"],args.image_size,args.batch_size,training=False,cache=False,shuffle_buffer=args.shuffle_buffer)

    return splits,label_map,num_classes,class_names,train_ds,val_ds,test_ds


def main():
    args=parse_args()
    np.random.seed(args.seed) #generador de semilla por defecto 42 para rereproducibilidad
    tf.random.set_seed(args.seed) 
    splits,label_map,num_classes,class_names,train_ds,val_ds,test_ds=prepare_data(args)
    class_weights=None
    if args.use_class_weights:
        class_weights=compute_weights(splits["train"]["labels"],num_classes)

    metrics=["accuracy"]
    model,base=build_bracs_cnn_transfer_1(input_shape=(args.image_size,args.image_size,3),num_classes=num_classes,dropout=args.dropout,trainable_at=args.trainable_at)
    model.summary()
    # Donde guardaremos los modelos y reportes
    model_dir=Path(args.output_dir)
    os.makedirs(model_dir,exist_ok=True)
    ckpt_path_head=model_dir/"best_head.h5"
    ckpt_path_ft=model_dir/"best_finetune.h5"

    cb_head=[tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=args.earlystop_patience,restore_best_weights=True)
             ,tf.keras.callbacks.ModelCheckpoint(str(ckpt_path_head),monitor="val_loss",save_best_only=True,verbose=1)]

    # Entrenamos solo la cabeza con el backbone congelado
    for layer in base.layers:
        layer.trainable=False

    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr_head),loss=tf.keras.losses.SparseCategoricalCrossentropy(),metrics=metrics)
    history_head=model.fit(train_ds,validation_data=val_ds,epochs=args.head_epochs,callbacks=cb_head,class_weight=class_weights,verbose=1)
    plot_training_history_all(history_head,title="Curvas head",show=True) # ploteo de entrenamiento de la cabeza

    # Entrenamiento fine-tuning
    base.trainable=True
    i=0
    for layer in base.layers:
        if i>=args.trainable_at:
            layer.trainable=True
        else:
            layer.trainable=False
        i+=1
    
    model.summary()
    cb_ft=[tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=args.earlystop_patience,restore_best_weights=True)
           ,tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",factor=0.5,patience=args.reduce_patience,min_lr=1e-6,verbose=1)
           ,tf.keras.callbacks.ModelCheckpoint(str(ckpt_path_ft),monitor="val_loss",save_best_only=True,verbose=1)]
    
    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr_ft),loss=tf.keras.losses.SparseCategoricalCrossentropy(),metrics=metrics)
    history_ft=model.fit(train_ds,validation_data=val_ds,epochs=args.finetune_epochs,callbacks=cb_ft,class_weight=class_weights,verbose=1)
    plot_training_history_all(history_ft,title="Curvas fine-tuning",show=True) # ploteo de entrenamiento de fine-tuning
    
    y_true=splits["test"]["labels"]
    probs=model.predict(test_ds) # obtenemos las preddiciones del test para las metricas (aqui a diferencia de breakhist no vamos a declarar
    # una funcion como run_and_evalualte porque solo usaremos un modelo, por eso se hace instantaneo)
    metrics_dict,report,cm,y_pred=compute_multiclass_metrics(y_true,probs,class_names)

    print("\nMétricas test:",json.dumps(metrics_dict,indent=2,default=_json_default))
    print("\nClassification report:\n",report)
    print("\nMétricas agregadas:"
          f"\n- Precisión macro: {metrics_dict['precision_macro']:.3f} | Recall macro: {metrics_dict['recall_macro']:.3f} | F1 macro: {metrics_dict['f1_macro']:.3f}"
          f"\n- Precisión micro: {metrics_dict['precision_micro']:.3f} | Recall micro: {metrics_dict['recall_micro']:.3f} | F1 micro: {metrics_dict['f1_micro']:.3f}"
          f"\n- Precisión ponderada: {metrics_dict['precision_weighted']:.3f} | Recall ponderado: {metrics_dict['recall_weighted']:.3f} | F1 ponderado: {metrics_dict['f1_weighted']:.3f}")
    print("\nMétricas por clase:")
    for cls_name in class_names:
        cls_metrics=metrics_dict["per_class"].get(cls_name,{})
        print(f"- {cls_name}: precision {cls_metrics.get('precision',0.0):.3f}, recall {cls_metrics.get('recall',0.0):.3f}, f1 {cls_metrics.get('f1',0.0):.3f}, soporte {cls_metrics.get('support',0)}")
    
    print("\nMatriz de confusión:\n",format_confusion_matrix(cm,class_names))
    plot_confusion_matrix_gray(cm,class_names,title="Matriz de confusión",show=True)
    plot_eval_metrics(metrics_dict,class_names,title="Métricas de evaluación",show=True)
    plot_split_distribution(splits,label_map,title="Distribución de imágenes por split y clase",show=True)
    model.save(model_dir/"final.h5")
    print(f"Modelo guardado en {model_dir/'final.h5'}")

if __name__=="__main__":
    main()
