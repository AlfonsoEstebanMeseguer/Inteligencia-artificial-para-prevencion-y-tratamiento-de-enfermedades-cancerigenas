import argparse
import json
import os
from functools import partial
from collections import Counter,defaultdict
from pathlib import Path
from typing import Dict,List,Sequence,Tuple

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report,confusion_matrix,precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers,models,regularizers
from tensorflow.keras.applications import efficientnet,resnet50

# Ruta por defecto al BRACS RoI
DEFAULT_BASE=(Path(__file__).resolve().parent/"BRACS"/"BRACS_RoI"/"latest_version_resized")

# Augmentación y normalización (fijas)

# Caches de capas aleatorias para no crear variables dentro de tf.function
ROT_LAYERS:Dict[float,layers.Layer]={}
ZOOM_LAYERS:Dict[float,layers.Layer]={}
TRANS_LAYERS:Dict[float,layers.Layer]={}

def parse_args():
    parser=argparse.ArgumentParser("Utilidades BRACS: lectura y split por paciente (sin entrenar).")
    parser.add_argument("--base-dir",type=str,default=str(DEFAULT_BASE),help="Raíz del dataset BRACS_RoI.")
    parser.add_argument("--train-size",type=float,default=0.8,help="Proporción train (resto val/test).")
    parser.add_argument("--val-size",type=float,default=0.1,help="Proporción val.")
    parser.add_argument("--test-size",type=float,default=0.1,help="Proporción test.")
    parser.add_argument("--seed",type=int,default=42,help="Semilla para reproducibilidad.")
    parser.add_argument("--max-images-per-patient",type=int,default=15,help="Tope de imágenes por paciente en train (sólo reporte).")
    parser.add_argument("--splits-dir",type=str,default=str(Path(__file__).resolve().parent/"splits"),help="Carpeta donde guardar/cargar los JSON de split (train/val/test).")
    parser.add_argument("--overwrite-splits",action="store_true",help="Sobrescribe los JSON de split si ya existen.")
    return parser.parse_args()

"""
En Bracs, el ID de paciente se encuentra en el nombre del archivo al igual que en breakhist, pero se usa un formato de nombre diferente a este, para extraer el patient_id 
del nombre del archivo, se asume que el formato es "img_{patient_id}_{image_id}" (por ejemplo, "img_12345_1.png" para la primera imagen del paciente 12345). 
"""
def parse_patient_id(filename):
    stem=Path(filename).stem
    parts=stem.split("_")
    if len(parts)>=3:
        return parts[1]
    return stem

"""
Misma lógica que la función ReadDataset.py de config de breakhist, recorremos los directorios jerarquicamente (aunque ahora sin augmentaciones) y
vamos obteniendo los labels (las distintas carpetas de clase) y las imágenes que hay dentro de cada una, formando un diccionario con la estructura total
del dataset asi como el mapeo de labels y el id de pacientes (importante para split por paciente)
"""
def read_bracs_dataset(base_dir,verbose):
    if not base_dir.exists():
        raise FileNotFoundError(f"No existe la ruta base:{base_dir}")
    splits=["train","val","test"]
    data={}
    class_names=[]
    records=[]
    split_class_counts={}
    split_class_patients={}

    for split in splits:
        split_dir=base_dir/split
        if not split_dir.is_dir():
            continue
        data[split]={}
        cls_list=[]
        for d in os.listdir(split_dir):
            if (split_dir/d).is_dir():
                cls_list.append(d)
        cls_list.sort()
        for cls in cls_list:
            if cls not in class_names:
                class_names.append(cls)
            cls_dir=split_dir/cls
            paths=[]
            for fname in os.listdir(cls_dir):
                if not fname.lower().endswith((".png",".jpg",".jpeg",".tif",".tiff")):
                    continue
                full_path=str(cls_dir/fname)
                pid=parse_patient_id(fname)
                paths.append(full_path)
                records.append((full_path,cls,pid,split))
                key_count=(split,cls)
                if key_count in split_class_counts:
                    split_class_counts[key_count]+=1
                else:
                    split_class_counts[key_count]=1
                key=(split,cls)
                if key not in split_class_patients:
                    split_class_patients[key]=[]
                if pid not in split_class_patients[key]:
                    split_class_patients[key].append(pid)
            data[split][cls]=paths

    if not records:
        raise ValueError(f"No se encontraron imágenes en {base_dir}")

    class_names.sort()
    label_map={}
    i=0
    while i<len(class_names):
        label_map[class_names[i]]=i
        i+=1

    all_images=[]
    all_labels=[]
    patient_ids=[]
    for r in records:
        all_images.append(r[0])
        all_labels.append(label_map[r[1]])
        patient_ids.append(r[2])

    if verbose:
        counts={}
        for lbl in all_labels:
            if lbl in counts:
                counts[lbl]+=1
            else:
                counts[lbl]=1
        print("\nLECTURA DATASET BRACS")
        print(f"Total imágenes:{len(all_images)}")
        unique_patients=[]
        for pid in patient_ids:
            if pid not in unique_patients:
                unique_patients.append(pid)
        print(f"Pacientes únicos:{len(unique_patients)}")

        print("\nDETALLE POR SPLIT Y CLASE")
        for split in splits:
            if split not in data:
                continue
            print(f"\n{split.upper()}")
            cls_keys=[]
            for k in data[split].keys():
                cls_keys.append(k)
            cls_keys.sort()
            for cls in cls_keys:
                img_cnt=split_class_counts[(split,cls)]
                patient_cnt=len(split_class_patients[(split,cls)])
                print(f"Clase {cls}(label {label_map[cls]}):Imágenes {img_cnt}|Pacientes únicos {patient_cnt}")

        print("\nRESUMEN FINAL")
        ordered_classes=[]
        i=0
        while i<len(label_map):
            for cls in label_map:
                if label_map[cls]==i:
                    ordered_classes.append((cls,i))
                    break
            i+=1
        for cls,idx in ordered_classes:
            patient_list=[]
            for rec in records:
                if rec[1]==cls:
                    pid=rec[2]
                    if pid not in patient_list:
                        patient_list.append(pid)
            patient_cls=len(patient_list)
            img_count=counts.get(idx,0)
            print(f"{cls}(label {idx}):Imágenes {img_count}|Pacientes únicos {patient_cls}")
        print(f"Total imágenes:{len(all_images)}")
        unique_patients=[]
        for pid in patient_ids:
            if pid not in unique_patients:
                unique_patients.append(pid)
        print(f"Total pacientes únicos:{len(unique_patients)}")
        print(f"Label map:{label_map}")

    return data,all_images,all_labels,patient_ids,label_map

# Función para guardar los splits en json (al igual que en breakhist)
def save_splits_to_dir(splits,label_map,out_dir,base_dir):
    out_dir.mkdir(parents=True,exist_ok=True)
    for split_name in splits:
        data=splits[split_name]
        payload={"meta":{"base_dir":str(base_dir),"split":split_name}
                 ,"label_map":label_map
                 ,"images":data["images"]
                 ,"labels":data["labels"]
                 ,"patients":data["patients"]}
        out_file=out_dir/f"split_{split_name}.json"
        with open(out_file,"w",encoding="utf-8") as f:
            json.dump(payload,f,indent=2)
        print(f"Split {split_name} guardado en {out_file}")

# Función para cargar los splits desde json (al igual que en breakhist)
def load_splits_from_dir(out_dir):
    splits={}
    label_map=None
    for split_name in ["train","val","test"]:
        fpath=out_dir/f"split_{split_name}.json"
        if not fpath.exists():
            raise FileNotFoundError(f"No se encontró {fpath}")
        with open(fpath,"r",encoding="utf-8") as f:
            payload=json.load(f)
        splits[split_name]={"images":payload["images"],"labels":payload["labels"],"patients":payload["patients"]}
        if label_map is None:
            label_map=payload["label_map"]
    return splits,label_map

"""
Función muy parecida al split de breakhist por paciente PERO OJO, AQUI SI HAY PACIENTES MIXTOS, por eso debemos cuantificar cual es la etiqueta 
mayoritaria de cada paciente para poder hacer el split por paciente de forma estratificada, y así asegurar que cada split tenga una distribución similar de clases.
"""
def split_by_patient(images,labels,patients,train_size,val_size,test_size,seed):
    if abs(train_size+val_size+test_size-1.0)>1e-6: #comprobamos que las proporciones sumen 1.0
        raise ValueError("Las proporciones deben sumar 1.0q")
    patient_to_idx={}
    # Aquí en este for vamos creando las entradas del diccionario de pacientes que mapea cada patient_id a la lista de índices de 
    # imágenes que pertenecen a ese paciente, para luego hacer el split por paciente
    for i,pid in enumerate(patients):
        if pid not in patient_to_idx:
            patient_to_idx[pid]=[]
        patient_to_idx[pid].append(i)
    patient_keys=list(patient_to_idx.keys())
    patient_keys.sort()
    patient_labels=[]
    for pid in patient_keys: # Entonces ahora para cada paciente obtenemos la 
        #lista de índices de imágenes que le corresponden, y a partir de ahí obtenemos los labels de esas imágenes, 
        # y asignamos al paciente el label mayoritario entre sus imágenes (si hay empate, se asigna el label con 
        # menor valor numérico, es decir, el que aparece primero en el label map)

        # ESTA COMPROBACIÓN NO SE HIZO EN BREAKHIST PORQUE: OJO, BREAKHIST NO TIENE PACIENTES MIXTOS, PERO BRACS SI, 
        # POR ESO DEBEMOS SABER CUAL ES LA ETIQUETA MAYORITARIA DE CADA PACIENTE PARA PODER HACER EL SPLIT POR PACIENTE DE FORMA ESTRATIFICADA

        lbls=[]
        for i in patient_to_idx[pid]:
            lbls.append(labels[i])
        majority=int(np.bincount(lbls).argmax())
        patient_labels.append(majority)

    train_p,temp_p,y_train,y_temp=train_test_split(patient_keys,patient_labels,train_size=train_size,stratify=patient_labels,random_state=seed)
    val_ratio=val_size/(val_size+test_size)
    val_p,test_p,y_val,y_test=train_test_split(temp_p,y_temp,train_size=val_ratio,stratify=y_temp,random_state=seed)

    train_patients=set(train_p)
    val_patients=set(val_p)
    test_patients=set(test_p)
    train_images=[]
    train_labels=[]
    train_patients_out=[]
    val_images=[]
    val_labels=[]
    val_patients_out=[]
    test_images=[]
    test_labels=[]
    test_patients_out=[]
    i=0
    # Agregamos a cada split las imágenes, labels y pacientes correspondientes según el patient_id de cada imagen y el split al que corresponda ese patient_id
    while i<len(patients):
        pid=patients[i]
        if pid in train_patients:
            train_images.append(images[i])
            train_labels.append(labels[i])
            train_patients_out.append(patients[i])
        elif pid in val_patients:
            val_images.append(images[i])
            val_labels.append(labels[i])
            val_patients_out.append(patients[i])
        elif pid in test_patients:
            test_images.append(images[i])
            test_labels.append(labels[i])
            test_patients_out.append(patients[i])
        i+=1

    splits={
        "train":{"images":train_images,"labels":train_labels,"patients":train_patients_out},
        "val":{"images":val_images,"labels":val_labels,"patients":val_patients_out},
        "test":{"images":test_images,"labels":test_labels,"patients":test_patients_out},
    }
    # Verificación de no fuga de datos 
    train_patients=set(splits["train"]["patients"])
    val_patients=set(splits["val"]["patients"])
    test_patients=set(splits["test"]["patients"])
    leakage_train_val=list(train_patients & val_patients)
    leakage_train_test=list(train_patients & test_patients)
    leakage_val_test=list(val_patients & test_patients)
    if leakage_train_val or leakage_train_test or leakage_val_test:
        raise ValueError(f"Fuga de datos detectada: train_val={leakage_train_val}, train_test={leakage_train_test}, val_test={leakage_val_test}")
    return splits

"""
Función que imprime estadísticas detalladas de cada split incluyendo número de imágenes, pacientes únicos, distribución por clase y ratio entre 
clases. Recibe el diccionario de splits y el label_map para mostrar los nombres de las clases. Esto en Breakhist lo haciamos en el eda pero aquí lo 
dejamos como función aparte para poder llamarla desde el script de entrenamiento también, y así tener un reporte rápido de la distribución del split 
generado o cargado.
"""
def print_split_stats(splits,label_map):
    label_names=[] # Primero la lista de clases ordenadas por su índice
    i=0
    while i<len(label_map):
        for cls in label_map:
            if label_map[cls]==i:
                label_names.append(cls)
                break
        i+=1
    for split_name in ["train","val","test"]:
        if split_name not in splits:
            continue
        info=splits[split_name]
        images=info["images"]
        labels=info["labels"]
        patients=info["patients"]
        patient_list=[] # Conteo de pacientes únicos en cada split
        for pid in patients:
            if pid not in patient_list:
                patient_list.append(pid)
        class_counts={}  # Conteo de  imágenes de cada clase
        for lbl in labels:
            if lbl in class_counts:
                class_counts[lbl]+=1
            else:
                class_counts[lbl]=1

        print(f"\nESTADÍSTICAS SPLIT {split_name.upper()}")
        print(f"Imágenes:{len(images)}")
        print(f"Pacientes:{len(patient_list)}")

        img_lines=[]
        idx=0
        while idx<len(label_names):
            cls_name=label_names[idx]
            img_lines.append(f"{cls_name}:{class_counts.get(idx,0)}")
            idx+=1
        print(f"Imágenes por clase:{', '.join(img_lines)}")

"""
Función que plotea gráfico de barras con la distribución de imágenes por clase para cada split. 
Se anotan los conteos sobre cada barra (cada barra es uan subclase (hay 8) y cada conjunto split tendrá las 8 barras respectivas).
"""
def plot_split_distribution(splits,label_map,title,show):
    split_names=[]
    for s in ["train","val","test"]:
        if s in splits:
            split_names.append(s)
    if not split_names:
        return None
    # Iuglua que en print_split_stats construimos la lista de clases ordenadas según label map 
    class_names=[]
    i=0
    while i<len(label_map):
        for cls in label_map:
            if label_map[cls]==i:
                class_names.append(cls)
                break
        i+=1
    # Matrices de conteos (filas->splits, columnas->clases)
    img_counts=np.zeros((len(split_names),len(class_names)),dtype=np.int32)
    patient_counts=np.zeros((len(split_names),len(class_names)),dtype=np.int32)
    i=0
    while i<len(split_names):
        split=split_names[i]
        labels=splits[split]["labels"]
        patients=splits[split].get("patients",[])
        per_class_patients=[]
        j=0
        while j<len(class_names):
            per_class_patients.append(set())
            j+=1
        k=0
        while k<len(labels):
            lbl=int(labels[k])
            if 0<=lbl<len(class_names):
                img_counts[i,lbl]+=1
                if k<len(patients):
                    per_class_patients[lbl].add(str(patients[k]))
            k+=1
        j=0
        while j<len(class_names):
            patient_counts[i,j]=len(per_class_patients[j])
            j+=1
        i+=1

    x=np.arange(len(split_names))
    width=0.8/max(1,len(class_names))
    fig,ax=plt.subplots(figsize=(12,6))
    j=0
    while j<len(class_names):
        cls_name=class_names[j]
        offset=(j-(len(class_names)-1)/2)*width
        vals=img_counts[:,j]
        bars=ax.bar(x+offset,vals,width=width,label=cls_name)
        i=0
        while i<len(bars):
            bar=bars[i]
            img_val=int(vals[i])
            pat_val=int(patient_counts[i,j])
            y_text=bar.get_height()+max(1,img_counts.max()*0.01)
            ax.text(bar.get_x()+bar.get_width()/2,y_text,f"{img_val}\nP:{pat_val}",ha="center",va="bottom",fontsize=7)
            i+=1
        j+=1

    split_labels=[]
    i=0
    while i<len(split_names):
        split_labels.append(split_names[i].upper())
        i+=1
    ax.set_xticks(x)
    ax.set_xticklabels(split_labels)
    ax.set_xlabel("Split")
    ax.set_ylabel("Imágenes")
    ax.set_title(title)
    ax.grid(True,axis="y",alpha=0.25)
    ax.legend(loc="upper right",ncol=2,frameon=True)
    fig.tight_layout()
    if show:
        plt.show()
    plt.close(fig)
    return fig

"""
Función que recibe una lista de métricas de un historial de entrenamiento y combina sus métricas en un solo dict
Función que auxilia al plot de las metricas de entrenamiento
"""
def merge_histories(histories):
    merged={}
    for history in histories:
        if hasattr(history,"history"):
            hist=history.history
        else:
            hist=history
        for key in hist:
            values=hist[key]
            if key not in merged:
                merged[key]=[]
            merged[key].extend(list(values))
    return merged


"""
Función que plotea las curvas de entrenamiento y validación de todas las métricas disponibles en el historial de entrenamiento.
Las métricas de entrenamiento se grafican con líneas sólidas y las de validación con líneas discontinuas del mismo color para facilitar la comparación. 
(al igual que en Breakhist pero aquí separamos la función de merge de historiales para poder usarla también en el script de entrenamiento, 
y así tener un gráfico combinado de todas las fases del entrenamiento).
"""
def plot_training_history_all(history,title,show):
    if hasattr(history,"history"): #History de Keras
        hist=history.history
    else:
        hist=history
    # metricas train menos val_*
    metrics=[]
    for k in hist.keys():
        if not k.startswith("val_"):
            metrics.append(k)
    if not metrics: # Si no métricas, no grafica 
        return None
    metrics.sort() # orden para gráfico
    fig,ax=plt.subplots(figsize=(12,6))
    color_cycle=plt.rcParams["axes.prop_cycle"].by_key()["color"]
    idx=0
    for metric in metrics:
        color=color_cycle[idx%len(color_cycle)]
        ax.plot(hist[metric],label=f"train_{metric}",color=color,linestyle="-") # línea train continua
        val_key=f"val_{metric}"
        if val_key in hist:
            ax.plot(hist[val_key],label=val_key,color=color,linestyle="--") #línea val discontinua
        idx+=1
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Valor")
    ax.grid(True,alpha=0.2)
    # Leyenda debajo para no tapar las curvas
    ax.legend(loc="upper center",bbox_to_anchor=(0.5,-0.12),ncol=min(4,len(metrics)*2),frameon=True)
    fig.tight_layout()  # ajusto margenes
    if show:
        plt.show()
    plt.close(fig)
    return fig

"""
Igual que plotMatrix de breajkhist multiclase pero orientado a bracs
"""
def plot_confusion_matrix_gray(cm,class_names,title,show):
    cm=np.asarray(cm)  # Aseguramos numpy
    fig,ax=plt.subplots(figsize=(max(5,len(class_names)*0.9),max(5,len(class_names)*0.9))) # figura proporcional al número de clases
    im=ax.imshow(cm,cmap="Blues") # escala de azules
    ax.figure.colorbar(im,ax=ax,fraction=0.046,pad=0.04)
    # Etiquetas de ejes con nombres de clase
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names,rotation=30,ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_title(title)
    if cm.size: # umbral para camnio de color
       thresh=cm.max()/2.0  
    else:
        thresh=0
    i=0
    while i<cm.shape[0]:
        j=0
        while j<cm.shape[1]:
            if cm[i,j]>thresh:
                color="white" 
            else:
                color="black"
            ax.text(j,i,f"{cm[i,j]}",ha="center",va="center",color=color,fontsize=8)
            j+=1
        i+=1
    fig.tight_layout() # ajustamos layout
    if show:
        plt.show()
    plt.close(fig)
    return fig

"""
Función que devuelve las métricas de evaluación multiclase. Calcula métricas (accuracy, log_loss, precision/recall/f1 macro, micro y weighted) y 
métricas por clase (precision/recall/f1/support). También genera el reporte de clasificación y la matriz de confusión.
[PARECIDO A EVAL_MULTICLASS DE BREAKHIST PERO ORIENTADO A BRACS]
"""
def compute_multiclass_metrics(y_true,probs,class_names):
    y_true=np.array(y_true,dtype=np.int32)
    probs=np.asarray(probs)
    y_pred=np.argmax(probs,axis=1)
    report=classification_report(y_true,y_pred,target_names=class_names,digits=3,zero_division=0)
    cm=confusion_matrix(y_true,y_pred)
    precision_macro,recall_macro,f1_macro,_=precision_recall_fscore_support(y_true,y_pred,average="macro",zero_division=0)
    precision_weighted,recall_weighted,f1_weighted,_=precision_recall_fscore_support(y_true,y_pred,average="weighted",zero_division=0)
    precision_micro,recall_micro,f1_micro,_=precision_recall_fscore_support(y_true,y_pred,average="micro",zero_division=0)
    per_prec,per_rec,per_f1,per_sup=precision_recall_fscore_support(y_true,y_pred,labels=list(range(len(class_names))),zero_division=0)
    metrics_dict={}
    metrics_dict["accuracy"]=float(np.mean(y_true==y_pred))
    metrics_dict["log_loss"]=float(np.mean(tf.keras.losses.sparse_categorical_crossentropy(y_true,probs).numpy()))
    metrics_dict["precision_macro"]=float(precision_macro)
    metrics_dict["recall_macro"]=float(recall_macro)
    metrics_dict["f1_macro"]=float(f1_macro)
    metrics_dict["precision_micro"]=float(precision_micro)
    metrics_dict["recall_micro"]=float(recall_micro)
    metrics_dict["f1_micro"]=float(f1_micro)
    metrics_dict["precision_weighted"]=float(precision_weighted)
    metrics_dict["recall_weighted"]=float(recall_weighted)
    metrics_dict["f1_weighted"]=float(f1_weighted)
    metrics_dict["per_class"]={}
    i=0
    while i<len(class_names):
        cls_name=class_names[i]
        metrics_dict["per_class"][cls_name]={"precision":float(per_prec[i]),"recall":float(per_rec[i]),"f1":float(per_f1[i]),"support":int(per_sup[i])}
        i+=1
    return metrics_dict,report,cm,y_pred

"""
Función de ploteo de las métricas de eval multiclase (como breakhist pero orientado a brcas)
"""
def plot_eval_metrics(metrics_dict,class_names,title,show):
    if not metrics_dict: # Si no métricas, no grafica 
        return None
    agg_keys=["accuracy","precision_macro","recall_macro","f1_macro","precision_micro","recall_micro","f1_micro","precision_weighted","recall_weighted","f1_weighted"]
    agg_vals=[]
    agg_names=[]
    for k in agg_keys:
        if k in metrics_dict:
            agg_vals.append(float(metrics_dict[k]))
            agg_names.append(k)
    # Métricas por clase 
    per_class=metrics_dict.get("per_class",{})
    per_f1=[]
    for c in class_names:
        per_f1.append(float(per_class.get(c,{}).get("f1",0.0)))
    log_loss=metrics_dict.get("log_loss",None)  # Log-loss
    fig,axes=plt.subplots(1,2,figsize=(14,6))
    axes[0].barh(agg_names,agg_vals,color="tab:gray")
    # Escala invertida (de izq a drch)
    axes[0].invert_xaxis()
    axes[0].set_xlim(1.0,0.0)
    axes[0].set_title("Métricas agregadas")
    axes[0].grid(True,axis="x",alpha=0.2)
    i=0
    while i<len(agg_vals):
        val=agg_vals[i]
        axes[0].text(val-0.02,i,f"{val:.3f}",ha="right",va="center",fontsize=8)
        i+=1  # Etiquetas numéricas de cada barra

    axes[1].barh(class_names,per_f1,color="tab:blue")
    axes[1].invert_xaxis() # Misma escala 
    axes[1].set_xlim(1.0,0.0)
    axes[1].set_title("F1 por clase")
    axes[1].grid(True,axis="x",alpha=0.2)
    i=0
    while i<len(per_f1):
        val=per_f1[i]
        axes[1].text(val-0.02,i,f"{val:.3f}",ha="right",va="center",fontsize=8)
        i+=1

    subtitle=f"{title}"
    if log_loss is not None:
        subtitle+=f"|log_loss={float(log_loss):.3f}"
    fig.suptitle(subtitle)
    # ajustamos margenes
    fig.tight_layout()
    if show:
        plt.show()
    plt.close(fig)
    return fig

"""
Igual que función random bool de breakhist
"""
def random_bool(prob):
    return tf.less(tf.random.uniform([],0.0,1.0),prob)

"""
Función que aplica augmentaciones YA AVANZADAS (NO hay niveles (low, none, medium,etc) como en breakhist porque no es necesario, ya que ni con 
augmentacion avanzada ni transfer llegaremos a resultados altos), asi que solo se usa augmentacion avanzada. No voy a repetir que hace cada cosa
porque es muy parecida a la función de augmnetación de breakhist 
"""
def apply_augmentations(image,img_size):
    h,w=img_size
    key_rot=25.0
    layer=ROT_LAYERS.get(key_rot)
    if layer is None:
        layer=layers.RandomRotation(key_rot/360.0)
        ROT_LAYERS[key_rot]=layer
    image=layer(image,training=True)

    key_zoom=0.15
    layer=ZOOM_LAYERS.get(key_zoom)
    if layer is None:
        layer=layers.RandomZoom(key_zoom,key_zoom)
        ZOOM_LAYERS[key_zoom]=layer
    image=layer(image,training=True)

    key_shift=0.15
    layer=TRANS_LAYERS.get(key_shift)
    if layer is None:
        layer=layers.RandomTranslation(key_shift,key_shift)
        TRANS_LAYERS[key_shift]=layer
    image=layer(image,training=True)
    image=tf.where(tf.random.uniform([],0.0,1.0)<0.7,tf.image.flip_left_right(image),image)
    image=tf.where(tf.random.uniform([],0.0,1.0)<0.35,tf.image.flip_up_down(image),image)
    image=tf.image.random_brightness(image,0.15)
    image=tf.image.random_contrast(image,1-0.15,1+0.15)
    image=tf.image.random_hue(image,0.08)
    image=tf.image.random_saturation(image,1-0.12,1+0.12)

    if h>100 and w>100:
        cut_h=tf.cast(tf.round(tf.cast(h,tf.float32)*0.2),tf.int32)
        cut_w=tf.cast(tf.round(tf.cast(w,tf.float32)*0.2),tf.int32)
        y=tf.random.uniform([],0,h-cut_h,dtype=tf.int32)
        x=tf.random.uniform([],0,w-cut_w,dtype=tf.int32)
        zeros=tf.zeros((cut_h,cut_w,3),dtype=image.dtype)
        paddings=[[y,h-y-cut_h],[x,w-x-cut_w],[0,0]]
        cut_mask=tf.pad(zeros,paddings,constant_values=1.0)
        cutout_img=image*cut_mask
        image=tf.where(tf.random.uniform([],0.0,1.0)<0.5,cutout_img,image)
    return image

"""
Función de normalización que solo se especializa en efficientNet (en bracs solo entrenaremos transfer efficientNet)
"""
def normalize_image(image):
    image=tf.cast(image,tf.float32)
    return efficientnet.preprocess_input(image*255.0)

def load_image(path,label,img_size,training):
    img_bytes=tf.io.read_file(path)
    # Decodificación nativa (sin py_function) para mantener pipeline más estable y eficiente.
    img=tf.io.decode_image(img_bytes,channels=3,expand_animations=False)
    img.set_shape([None,None,3])
    # Redimensionamos primero para evitar cast explícito de imágenes originales gigantes.
    img=tf.image.resize(img,img_size)
    # Tras resize (rango 0..255), normalizamos manualmente a 0..1.
    img=tf.cast(img,tf.float32)/255.0
    if training:
        img=apply_augmentations(img,img_size)
    img=normalize_image(img)
    return img,label


def make_dataset(paths,labels,image_size,batch_size,training,cache,shuffle_buffer):
    img_size=(image_size,image_size)
    ds=tf.data.Dataset.from_tensor_slices((list(paths),list(labels)))
    if training: #suffle en train
        ds=ds.shuffle(buffer_size=min(len(paths)*2,shuffle_buffer),reshuffle_each_iteration=True)
    map_fn=partial(load_image,img_size=img_size,training=training)
    # Limitamos paralelismo para reducir picos de RAM al decodificar imágenes grandes.
    ds=ds.map(map_fn,num_parallel_calls=4,deterministic=False)
    if cache: # cache en train
        ds=ds.cache()
    ds=ds.batch(batch_size,drop_remainder=False).prefetch(tf.data.AUTOTUNE) # obligatorio batch porque las imagenes son grandes y
    #no caben en memoria, prefetch para rendimiento
    return ds

# Class weight
def compute_weights(labels,num_classes):
    classes=np.arange(num_classes)
    weights=compute_class_weight(class_weight="balanced",classes=classes,y=np.array(labels))
    res={}
    i=0
    while i<len(classes):
        res[int(classes[i])]=float(weights[i])
        i+=1
    return res

# Main nos sirve para inspeccionar splits o generarlos al gusto pero realmente las funciones son accesibles a través de los scripts de entrenamiento, 
# así que no es necesario ejecutar este script main previamente para entrenar.
def main():
    args=parse_args()
    base_dir=Path(args.base_dir)
    splits_dir=Path(args.splits_dir)

    if splits_dir.exists() and not args.overwrite_splits:
        try: # cargamos splits
            splits,label_map=load_splits_from_dir(splits_dir)
            print(f"Cargando splits existentes desde {splits_dir}")
        except FileNotFoundError:
            # si hay algun problema, generamos splits nuevos y por si acaso directorio de splits por si no existe
            splits_dir.mkdir(parents=True,exist_ok=True)
            _,images,labels,patients,label_map=read_bracs_dataset(base_dir,verbose=True)
            splits=split_by_patient(images,labels,patients,args.train_size,args.val_size,args.test_size,args.seed)
            save_splits_to_dir(splits,label_map,splits_dir,base_dir)
    else:
        # generamos splits nuevos (en teoria el directorio existe y es accesible)
        _,images,labels,patients,label_map=read_bracs_dataset(base_dir,verbose=True)
        splits=split_by_patient(images,labels,patients,args.train_size,args.val_size,args.test_size,args.seed)
        save_splits_to_dir(splits,label_map,splits_dir,base_dir)

    print_split_stats(splits,label_map)
    plot_split_distribution(splits,label_map,title="Distribución de imágenes por split y clase",show=True)

if __name__=="__main__":
    tf.get_logger().setLevel("ERROR")
    main()
