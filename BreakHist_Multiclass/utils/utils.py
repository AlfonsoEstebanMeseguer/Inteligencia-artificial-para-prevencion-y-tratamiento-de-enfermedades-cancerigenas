import os,math,json
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import (classification_report,confusion_matrix,accuracy_score,balanced_accuracy_score,top_k_accuracy_score,roc_auc_score,log_loss,)
from BreakHist_Binary.src.utils.utils import (ensure_splits,get_datasets_basic,make_gradcam_heatmap,show_gradcam_example,resolve_split_dir,plot_metrics_bar,)

# Valores por defecto para splits multiclase
DEFAULT_TRAIN_SIZE=0.7
DEFAULT_VAL_SIZE=0.15
DEFAULT_TEST_SIZE=0.15
DEFAULT_RANDOM_STATE=42

"""
La única diferencia es que esta evaluación no necesita un umbral para saber si es 1/0 para binario, además de que
proporciona otras métricas más específicas para clasificación multiclase
"""
def evaluate_multiclass(model,ds_bundle):
    y_true=[]
    y_prob=[]
    y_pred=[]
    for batch_imgs,batch_labels in ds_bundle["test_ds"].take(ds_bundle["test_steps"]):
        preds=model.predict(batch_imgs,verbose=0)
        y_true.append(batch_labels.numpy())
        y_prob.append(preds)
    y_true=np.concatenate(y_true)
    y_prob=np.concatenate(y_prob)
    y_pred=np.argmax(y_prob,axis=1)
    acc=accuracy_score(y_true,y_pred)
    cm=confusion_matrix(y_true,y_pred)
    num_classes=y_prob.shape[1]
    # Hasta ahora todo "igual" que binary adaptado a Multiclase
    # Promedios macro/weighted/micro
    report_dict=classification_report(y_true,y_pred,digits=3,zero_division=0,output_dict=True)
    metrics={"accuracy":acc,"balanced_accuracy":balanced_accuracy_score(y_true,y_pred),"precision_macro":report_dict["macro avg"]["precision"]
             ,"recall_macro":report_dict["macro avg"]["recall"],"f1_macro":report_dict["macro avg"]["f1-score"],"precision_weighted":report_dict["weighted avg"]["precision"]
             ,"recall_weighted":report_dict["weighted avg"]["recall"],"f1_weighted":report_dict["weighted avg"]["f1-score"]}
    if "micro avg" in report_dict:
        metrics.update({"precision_micro":report_dict["micro avg"]["precision"],"recall_micro":report_dict["micro avg"]["recall"]
                        ,"f1_micro":report_dict["micro avg"]["f1-score"]})

    """
    Top‑k te dice si la clase correcta está entre las k más probables, útil cuando hay clases parecidas y necesito
    medir "casi aciertos". Log_loss evalúa la calidad de las probabilidades, penalizando mucho predicciones con alta confianza y 
    erróneas. Es útil para comparar modelos aunque tengan accuracies similares.

    Básicamente que si 8 clases son parecidas y el modelo acierta 6 de ellas pero se equivoca con 2, 
    el top-3 podría ser alto (porque la clase correcta está entre las 3 más probables) aunque el accuracy sea 
    bajo (porque solo acierta 6/8). Log_loss penalizaría mucho esos errores con alta confianza, mostrando que el 
    modelo no es tan bueno como parece con el top-k. Por eso es importante usar varias métricas para tener una 
    visión completa del rendimiento del modelo.
    """
    for k in (2,3,5): #top-k con 2,3,5 clases 
        if k <= num_classes:
            metrics[f"top{k}_accuracy"]=top_k_accuracy_score(y_true,y_prob,k=k,labels=np.arange(num_classes))
    try:
        metrics["log_loss"]=log_loss(y_true,y_prob,labels=np.arange(num_classes))
    except ValueError:
        pass

    # AUCs multiclase (se omiten si no hay probas válidas)
    for average in ("macro","weighted"):
        try:
            metrics[f"roc_auc_ovr_{average}"]=roc_auc_score(y_true,y_prob,multi_class="ovr",average=average)
        except ValueError:
            pass
        try:
            metrics[f"roc_auc_ovo_{average}"]=roc_auc_score(y_true,y_prob,multi_class="ovo",average=average)
        except ValueError:
            pass

    return metrics,cm,classification_report(y_true,y_pred,digits=3,zero_division=0)

"""
Función para plotear la matriz de confusión para multiclase, similar a función binaria
"""
def plot_confusion_matrix(cm,class_names,title):
    n=cm.shape[0]
    if class_names is None:
        class_names=[]
        i=0
        while i < n:
            class_names.append(str(i))
            i+=1
    fig,ax=plt.subplots(figsize=(6,6))
    im=ax.imshow(cm,cmap="Blues")
    ax.figure.colorbar(im,ax=ax)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names,rotation=45,ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    if cm.size:
        thresh=cm.max()/2.0 
    else:
        thresh=0
    for i in range(n):
        for j in range(n):
            val=cm[i,j]
            if val>thresh:
                color="white"
            else:
                color="black"
            ax.text(j,i,f"{val}",ha="center",va="center",color=color,fontsize=8)
    plt.tight_layout()
    plt.show()
    return fig
