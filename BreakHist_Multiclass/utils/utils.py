import os,math,json
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    top_k_accuracy_score,
    roc_auc_score,
    log_loss,
)
from BreakHist_Binary.src.utils.utils import (
    ensure_splits,
    get_datasets_basic,
    make_gradcam_heatmap,
    show_gradcam_example,
    resolve_split_dir,
)

# Valores por defecto para splits multiclase
DEFAULT_TRAIN_SIZE=0.8
DEFAULT_VAL_SIZE=0.1
DEFAULT_TEST_SIZE=0.1
DEFAULT_RANDOM_STATE=42

# Reexport helpers para que estén disponibles desde este módulo
__all__=[
    "ensure_splits",
    "get_datasets_basic",
    "make_gradcam_heatmap",
    "show_gradcam_example",
    "resolve_split_dir",
    "evaluate_multiclass",
    "DEFAULT_TRAIN_SIZE",
    "DEFAULT_VAL_SIZE",
    "DEFAULT_TEST_SIZE",
    "DEFAULT_RANDOM_STATE",
]

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

    # Promedios macro/weighted/micro
    report_dict=classification_report(y_true,y_pred,digits=3,zero_division=0,output_dict=True)
    metrics={
        "accuracy":acc,
        "balanced_accuracy":balanced_accuracy_score(y_true,y_pred),
        "precision_macro":report_dict["macro avg"]["precision"],
        "recall_macro":report_dict["macro avg"]["recall"],
        "f1_macro":report_dict["macro avg"]["f1-score"],
        "precision_weighted":report_dict["weighted avg"]["precision"],
        "recall_weighted":report_dict["weighted avg"]["recall"],
        "f1_weighted":report_dict["weighted avg"]["f1-score"],
    }
    if "micro avg" in report_dict:
        metrics.update({
            "precision_micro":report_dict["micro avg"]["precision"],
            "recall_micro":report_dict["micro avg"]["recall"],
            "f1_micro":report_dict["micro avg"]["f1-score"],
        })

    # Top-k y log-loss
    for k in (2,3,5):
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
