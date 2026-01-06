import os,math,json
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from BreakHist_Multiclass.config.readDataset import read_multiclass_breakhis_data
from BreakHist_Multiclass.config.split_dataset import split_by_patient,split_by_image
from BreakHist_Multiclass.config.create_dataset import load_split,create_dataset,compute_class_weights,decode_image,preprocess_image
from BreakHist_Binary.config.utils.utils import resolve_split_dir

def ensure_splits(base_path,split_dir,split_mode,train_size,val_size,test_size,random_state):
    split_dir=resolve_split_dir(split_dir,split_mode)
    missing=[]
    for s in ["train","val","test"]:
        split_path=os.path.join(split_dir,f"{s}.json")
        if not os.path.exists(split_path):
            missing.append(split_path)

    regenerate=False
    if missing:
        regenerate=True
        print(f"Faltan splits, se regenerarán: {','.join(missing)}")
    else:
        for s in ["train","val","test"]:
            split_path=os.path.join(split_dir,f"{s}.json")
            try:
                with open(split_path,"r",encoding="utf-8") as f:
                    data=json.load(f)
                sample_paths=data.get("images",[])[:20]
                for p in sample_paths:
                    if not os.path.exists(p):
                        print(f"Ruta inexistente detectada en {split_path}: {p}")
                        regenerate=True
                        break
                if regenerate:
                    break
            except Exception as e:
                print(f"No se pudo validar {split_path}: {e}")
                regenerate=True
                break

    if regenerate:
        os.makedirs(split_dir,exist_ok=True)
        _,all_images,all_labels,_,slides=read_multiclass_breakhis_data(base_path,verbose=False)
        if split_mode=="patient":
            splits,_=split_by_patient(all_images,all_labels,slides,train_size=train_size,val_size=val_size,test_size=test_size,random_state=random_state)
        else:
            splits,_=split_by_image(all_images,all_labels,slides,train_size=train_size,val_size=val_size,test_size=test_size,random_state=random_state)
        for split_name,split_data in splits.items():
            with open(os.path.join(split_dir,f"{split_name}.json"),"w",encoding="utf-8") as f:
                json.dump(split_data,f,indent=2)
    else:
        return

"""
Copia idéntica a utils binario pero este usa funciones de create-dataset.multicase, no es lo mismo realmente sus funciones
aun con mismo nombre, son distintas
"""
def get_datasets_basic(config,split_dir,include_labels):
    train_imgs,train_labels=load_split(split_dir,"train")
    val_imgs,val_labels=load_split(split_dir,"val")
    test_imgs,test_labels=load_split(split_dir,"test")
    if len(train_imgs)==0 or len(val_imgs)==0 or len(test_imgs)==0:
        raise RuntimeError("Algún split está vacío")
    train_ds=create_dataset(train_imgs,train_labels,training=True,config=config)
    val_ds=create_dataset(val_imgs,val_labels,training=False,config=config)
    test_ds=create_dataset(test_imgs,test_labels,training=False,config=config)
    steps_per_epoch=math.ceil(len(train_imgs)/config["batch_size"])
    val_steps=math.ceil(len(val_imgs)/config["batch_size"])
    test_steps=math.ceil(len(test_imgs)/config["batch_size"])
    if config["use_class_weights"]:
        class_weights=compute_class_weights(train_labels)
    else:
        class_weights=None
    num_classes=int(np.max(train_labels))+1
    out={"config":config,"train_ds":train_ds,"val_ds":val_ds,"test_ds":test_ds,
         "steps_per_epoch":steps_per_epoch,"val_steps":val_steps,"test_steps":test_steps,
         "class_weights":class_weights,"test_imgs":test_imgs,"num_classes":num_classes,"test_labels":test_labels}
    if include_labels:
        out.update({"train_labels":np.array(train_labels,dtype=np.int32),"val_labels":np.array(val_labels,dtype=np.int32),"test_labels":np.array(test_labels,dtype=np.int32)})
    return out

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
    report_dict=classification_report(y_true,y_pred,digits=3,zero_division=0,output_dict=True)
    metrics={"accuracy":acc,"precision_macro":report_dict["macro avg"]["precision"],
             "recall_macro":report_dict["macro avg"]["recall"],"f1_macro":report_dict["macro avg"]["f1-score"],
             "precision_weighted":report_dict["weighted avg"]["precision"],
             "recall_weighted":report_dict["weighted avg"]["recall"],
             "f1_weighted":report_dict["weighted avg"]["f1-score"]}
    return metrics,cm,classification_report(y_true,y_pred,digits=3,zero_division=0)

def plot_confusion_matrix(cm,class_names=None):
    num_classes=cm.shape[0]
    if class_names is None:
        class_names=[f"c{i}" for i in range(num_classes)]
    fig,ax=plt.subplots(figsize=(5,5))
    im=ax.imshow(cm,cmap="Blues")
    ax.figure.colorbar(im,ax=ax)
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(class_names,rotation=45,ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    thresh=cm.max()/2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color="white" if cm[i,j]>thresh else "black"
            ax.text(j,i,f"{cm[i,j]}",ha="center",va="center",color=color)
    plt.tight_layout()
    plt.show()
    plt.close(fig)
    return fig

def make_gradcam_heatmap(model,img_array,last_conv_layer_name):
    grad_model=tf.keras.models.Model([model.inputs],[model.get_layer(last_conv_layer_name).output,model.output])
    with tf.GradientTape() as tape:
        conv_outputs,predictions=grad_model(img_array)
        loss_vals=[]
        for pred in predictions:
            loss_vals.append(pred[0])
        loss=tf.stack(loss_vals)
    grads=tape.gradient(loss,conv_outputs)[0]
    pooled_grads=tf.reduce_mean(grads,axis=(0,1))
    conv_outputs=conv_outputs[0]
    heatmap=tf.reduce_sum(tf.multiply(pooled_grads,conv_outputs),axis=-1)
    heatmap=tf.maximum(heatmap,0)/(tf.reduce_max(heatmap)+1e-8)
    return heatmap.numpy()

def show_gradcam_example(model,config,image_path,last_conv_layer_name):
    img_decoded=decode_image(tf.constant(image_path))
    img=preprocess_image(img_decoded,False,config)
    img_array=tf.expand_dims(img,axis=0)
    heatmap=make_gradcam_heatmap(model,img_array,last_conv_layer_name)
    heatmap=np.uint8(255*heatmap)
    heatmap=np.expand_dims(heatmap,axis=-1)
    heatmap=tf.image.resize(heatmap,config["img_size"]).numpy().astype(np.uint8)
    fig,axes=plt.subplots(1,2,figsize=(8,4))
    axes[0].imshow(img_decoded)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(heatmap[...,0],cmap="jet")
    axes[1].set_title("Heatmap")
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()
    return fig
