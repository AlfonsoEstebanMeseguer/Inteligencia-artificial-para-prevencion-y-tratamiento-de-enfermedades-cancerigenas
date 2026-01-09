import os, math, json, traceback
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import (confusion_matrix,classification_report,roc_auc_score,average_precision_score,accuracy_score)
from src.config.readDataset import read_binary_breakhis_data
from src.config.split_dataset import split_by_patient as split_by_patient_shared, split_by_image as split_by_image_shared
from src.config.create_dataset import load_split as load_split_binary,create_dataset as create_dataset_binary,compute_class_weights as compute_class_weights_binary,decode_image,preprocess_image
from BreakHist_Multiclass.config.readDataset import read_multiclass_breakhis_data
from BreakHist_Multiclass.config.create_dataset import load_split as load_split_multiclass,create_dataset as create_dataset_multiclass,compute_class_weights as compute_class_weights_multiclass

"""
Esta función se encarga únicamente de crear cualquier.json que pueda faltar tras split. Simplemente ofrece mayor eficacia al programa en caso de que borremos los scripts
previamente al entrenamiento
"""
def resolve_dataset_helpers(dataset_type):
    if dataset_type=="binary":
        return (read_binary_breakhis_data, split_by_patient_shared, split_by_image_shared,
                load_split_binary, create_dataset_binary, compute_class_weights_binary)
    if dataset_type=="multiclass":
        return (read_multiclass_breakhis_data, split_by_patient_shared, split_by_image_shared,
                load_split_multiclass, create_dataset_multiclass, compute_class_weights_multiclass)
    raise ValueError(f"Tipo de dataset no soportado: {dataset_type}")

def ensure_splits(base_path,split_dir,train_size=0.8,val_size=0.1,test_size=0.1,split_mode="patient",dataset_type="binary",random_state=None):
    # Normalizamos la carpeta de splits según el modo solicitado solo si aún no incluye el sufijo esperado.
    target_suffix="split_patient" if split_mode=="patient" else "split_imagen"
    if Path(split_dir).name != target_suffix:
        split_dir=resolve_split_dir(split_dir,split_mode)
    read_data,split_by_patient_fn,split_by_image_fn,_,_,_= resolve_dataset_helpers(dataset_type)
    missing=[]
    for s in ["train","val","test"]:
        split_path=os.path.join(split_dir,f"{s}.json")
        if not os.path.exists(split_path):
            missing.append(split_path)

    regenerate=False
    if missing:
        regenerate=True
        print(f"Faltan splits, se regenerarán: {', '.join(missing)}")
    else:
        # Validamos que las rutas existentes realmente existen (evita rutas obsoletas)
        for s in ["train","val","test"]:
            split_path=os.path.join(split_dir,f"{s}.json")
            try:
                with open(split_path,"r",encoding="utf-8") as f:
                    data=json.load(f)
                sample_paths=data.get("images",[])[:20]  # comprobamos unas pocas rutas
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
        # Preparamos carpeta y leemos dataset completo para dividir
        os.makedirs(split_dir,exist_ok=True)
        if dataset_type=="multiclass":
            _,all_images,all_labels,label_map,slides=read_data(base_path,verbose=False)
        else:
            _,all_images,all_labels,label_map,slides=read_data(base_path,verbose=False)
        if split_mode=="patient":
            # Split estratificado por paciente (sin fuga)
            if dataset_type=="multiclass":
                splits,_=split_by_patient_fn(all_images,all_labels,slides,train_size=train_size,val_size=val_size,test_size=test_size
                                             ,random_state=random_state,dataset_type="multiclass",label_map=label_map)
            else:
                splits,_=split_by_patient_fn(all_images,all_labels,slides,train_size,val_size,test_size,random_state=random_state
                                             ,dataset_type="binary",label_map=label_map)
        else:
            # Split estratificado por clase a nivel de imagen (permite fuga)
            if dataset_type=="multiclass":
                splits,_=split_by_image_fn(all_images,all_labels,slides,train_size=train_size,val_size=val_size,test_size=test_size
                                           ,random_state=random_state,dataset_type="multiclass",label_map=label_map)
            else:
                splits,_=split_by_image_fn(all_images,all_labels,slides,train_size,val_size,test_size,random_state=random_state
                                           ,dataset_type="binary",label_map=label_map)
        # Guardamos cada subconjunto en JSON (se reescriben si falta alguno o si se regeneró por rutas inválidas)
        for split in ["train","val","test"]:
            with open(os.path.join(split_dir,f"{split}.json"),"w",encoding="utf-8") as f:
                json.dump(splits[split],f,indent=2)
    else:
        return

"""
Función simple, básicamente añade a split dir que no es más que el directorio "split" (únicamente) la ruta
que va a al split imagen o paciente dependiendo de lo que se ponga en "split_mode"
"""
def resolve_split_dir(split_dir,split_mode):
    base=Path(split_dir) # base = dir split ahora tenemos que ver el resto
                         # es decir split+split_image / split+split_patient
    if split_mode=="patient":
        target="split_patient"
    else:
        target="split_imagen"
    return str(base/target)

"""
Esta función crea datasets tf.data listos para model.fit con steps calculados. Obtiene los datos listos para tesorflow (imagenes, labels), con ellos crea los datasets,
calcula los steps con: ceil(redondeo hacia arriba) y NO división exacta (porque que, por ejemplo para 100 imágenes y tamaño de batch 32, habrían 3,125 pasos, que hacemos con el último?
, si no redondeamos hacia arriba, estaremos perdiendo imágenes por cada step, por ello forzamos a usar más num steps antes que menos).

Importante: aunque SOLO train será infinito, (no tiene sentido en val y test, solo buscamos validar), también calculo steps en val y test porque forzamos a que se evalúen todas
las imágenes, incluso si el batch es incompleto.

Tras obtener los datasets, los steps y los pesos (si procede), devuelve un diccionario con la información de configuración lista para entrenamiento.
"""
def get_datasets_basic(config,split_dir,include_labels=False,dataset_type="binary"):
    _,_,_,load_split_fn,create_dataset_fn,compute_class_weights_fn=resolve_dataset_helpers(dataset_type)
    train_imgs,train_labels=load_split_fn(split_dir,"train")
    val_imgs,val_labels=load_split_fn(split_dir,"val")
    test_imgs,test_labels=load_split_fn(split_dir,"test")
    if len(train_imgs)==0 or len(val_imgs)==0 or len(test_imgs)==0:
        raise RuntimeError("Algún split está vacío")
    if dataset_type=="multiclass":
        train_ds=create_dataset_fn(train_imgs,train_labels,training=True,config=config)
        val_ds=create_dataset_fn(val_imgs,val_labels,training=False,config=config)
        test_ds=create_dataset_fn(test_imgs,test_labels,training=False,config=config)
    else:
        train_ds=create_dataset_fn(train_imgs,train_labels,True,config)
        val_ds=create_dataset_fn(val_imgs,val_labels,False,config)
        test_ds=create_dataset_fn(test_imgs,test_labels,False,config)
    steps_per_epoch=math.ceil(len(train_imgs)/config["batch_size"])
    val_steps=math.ceil(len(val_imgs)/config["batch_size"])
    test_steps=math.ceil(len(test_imgs)/config["batch_size"])
    if config["use_class_weights"]:
        class_weights=compute_class_weights_fn(train_labels)
    else:
        class_weights=None
    num_classes=int(np.max(train_labels))+1
    out={"config":config,"train_ds":train_ds,"val_ds":val_ds,"test_ds":test_ds,"steps_per_epoch":steps_per_epoch,
         "val_steps":val_steps,"test_steps":test_steps,"class_weights":class_weights,"test_imgs":test_imgs,
         "test_labels":test_labels,"num_classes":num_classes}
    if include_labels:
        out.update({"train_labels":np.array(train_labels,dtype=np.int32),"val_labels":np.array(val_labels,dtype=np.int32),"test_labels":np.array(test_labels,dtype=np.int32)})
    return out

def plot_training_history(history, metrics=None, save_path=None, show=True):
    history_dict=getattr(history,"history",history)
    if metrics is None:
        metrics=[m for m in history_dict.keys() if not m.startswith("val_")]
    metrics=[m for m in metrics if m in history_dict]
    if not metrics:
        print("No hay métricas para graficar.")
        return None
    fig,ax=plt.subplots(figsize=(10,5))
    color_cycle=plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for idx,metric in enumerate(metrics):
        color=color_cycle[idx % len(color_cycle)]
        ax.plot(history_dict[metric],label=f"train_{metric}",color=color,linestyle="-")
        val_key=f"val_{metric}"
        if val_key in history_dict:
            ax.plot(history_dict[val_key],label=val_key,color=color,linestyle="--")
    ax.set_title("Curvas de entrenamiento")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Valor")
    handles,labels=ax.get_legend_handles_labels()
    ax.grid(True)
    plt.tight_layout()
    legend_fig,legend_ax=plt.subplots(figsize=(10,1.5))
    legend_ax.axis("off")
    legend_ax.legend(handles,labels,loc="center",ncol=min(len(labels),4),frameon=True)
    if show:
        plt.show()
    plt.close(fig)
    plt.close(legend_fig)
    return fig,legend_fig


"""
Función que evalúa en test y devuelve métricas y predicciones. Obtenemos todo test y comprobamos predicción del modelo ya entrenado con 'predict' donde
y_true es la lista de valores reales y y_prob la lista de valores predecidos, calculamos la matriz de confusión y el reporte de clasificación proporcionados por
skearn-metrics con métricas:
Accuracy: Porcentaje de predicciones correctas (medida global de rendimiento). Problema: engañosa con datos desbalanceados.
Precision: De todos los casos predichos, cuántos lo son realmente (Controlar falsos positivos)
Recall: De todos los casos reales, cuántos detecta el modelo (Minimizar falsos negativos) -> busca no dejar cánceres sin detectar (my importante en rama médica)
F1-score: Media armónica entre precision y recall (compromiso entre ambas)
ROC-AUC: Evalúa el modelo independientemente del umbral (Capacidad de separar clases) (1.0 -> perrfecto (irreal), 0.5 -> azar)
PR-AUC:Evalúa modelos con clases desbalanceadas (compromiso entre precision/sensibilidad cuando existe clase mayoritaria)
Specificity: Proporción de casos correctamente clasificados(Controlar falsos positivos) -> util para no alarmar a pacientes sanos
Sensitivity: Detecta  el máximo número de cánceres
"""
def evaluate_binary(model,test_ds,test_steps,threshold):
    y_true=[]
    y_prob=[]
    y_pred=[]
    for batch_imgs,batch_labels in test_ds.take(test_steps):
        preds=model.predict(batch_imgs,verbose=1).ravel()
        y_true.append(batch_labels.numpy().ravel())
        y_prob.append(preds)
    y_true=np.concatenate(y_true)
    y_prob=np.concatenate(y_prob)
    """
    COMPROBACIÓN POSIBLMENENTE REDUDANTE:
        # Forzamos etiquetas binarias ante cualquier valor extraño
        y_true = (y_true > 0).astype(np.int32)
        # Limpieza de valores inválidos
        y_prob=np.nan_to_num(y_prob,nan=0.0,posinf=1.0,neginf=0.0)
        y_prob=np.clip(y_prob,0.0,1.0)
    """
    for prob in y_prob:
        if prob >= threshold:
            y_pred.append(1)
        else:
            y_pred.append(0)
    y_pred=np.array(y_pred,dtype=np.int32)
    cm=confusion_matrix(y_true,y_pred,labels=[0,1])
    rep=classification_report(y_true,y_pred,labels=[0,1],target_names=["benign","malignant"],output_dict=True,zero_division=0)
    acc=accuracy_score(y_true,y_pred)
    roc_auc=roc_auc_score(y_true, y_prob) 
    pr_auc=average_precision_score(y_true,y_prob)
    metrics = {"accuracy":acc,"precision":rep["weighted avg"]["precision"],"recall":rep["weighted avg"]["recall"]
               ,"f1":rep["weighted avg"]["f1-score"],"roc_auc":roc_auc,"pr_auc":pr_auc
               ,"specificity":cm[0,0]/(cm[0,0]+cm[0,1]+1e-8),"sensitivity":cm[1,1]/(cm[1,1]+cm[1,0]+1e-8)}
    
    return metrics, cm, y_true, y_prob, y_pred

"""
Función que guarda matriz de confusión. 
"""
def plot_confusion_matrix(cm,labels=("Benign","Malignant")):
    # cm -> matriz de confusión
    fig,ax=plt.subplots(figsize=(4, 4)) # figura tamaño 4x4
    im=ax.imshow(cm,cmap="Blues") # visualizamos matriz como imagen, mapa de color en escala de azules
    ax.figure.colorbar(im,ax=ax) # barra que valor numérico corresponde a cada valor azul
    ax.set_xticks([0,1]) # índices validos para tamaño 2x2
    ax.set_yticks([0,1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    thresh=cm.max()/2.0 
    # Esto básicamente es para controlar si es mejor escribir texto blanco sobre fondo oscuro o al revés. (Legilibilidad)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i,j] > thresh:
                col="white"
            else:
                col="black"     
            ax.text(j,i,f"{cm[i, j]}",ha="center",va="center",color=col)
    plt.tight_layout()
    plt.show()
    return fig

"""
Función que genera heatmap Grad-CAM para una imagen preprocesada. Lo primero es construir un modelo que devuelva la predicción final del modelo, (model.output)
y la última capa convolucional del modelo en cuestión. Este modelo es necesario pues necesitamos la predicción real (no la aproximada por alguna entropía o modificación
, ej sigmoide o binarycrossentropy), es decir, el valor real predicho por la última capa convolucional. Y, necesitamos el resultado final del modelo, para calcular gradientes de 
calor-frio. Usamos GradientTape, proporcionado por tensorflow para registrar todos los cálculos realizados en una cinta, que se usará para la representación. De seguido,
"""
def find_layer_recursive(model, layer_name):
    """Busca una capa por nombre, devolviendo (layer, parent_model) si se encuentra."""
    for layer in model.layers:
        if layer.name==layer_name:
            return layer, model
        if isinstance(layer, tf.keras.Model) and hasattr(layer, "layers"):
            found=find_layer_recursive(layer,layer_name)
            if found[0] is not None:
                return found
    return None, None

def make_gradcam_heatmap(model, img_array,last_conv_layer_name):
    target_layer,parent_model=find_layer_recursive(model,last_conv_layer_name)
    if target_layer is None:
        raise ValueError(f"No se encontró la capa {last_conv_layer_name} en el modelo para Grad-CAM.")
    # Si la capa está en un modelo anidado (p.ej. resnet50), necesitamos reencaminar al input del modelo principal
    if parent_model is model:
        target_tensor=target_layer.output
    else:
        intermediate=tf.keras.models.Model(parent_model.input,target_layer.output)
        target_tensor=intermediate(model.input)
    grad_model=tf.keras.models.Model(model.input,[target_tensor,model.output])
    with tf.GradientTape() as tape:
        conv_outputs,predictions=grad_model(img_array) # salidas de la última capa conv y pred final
        loss_vals=[]
        for pred in predictions:  # recorre el batch
            loss_vals.append(pred[0]) 
        loss=tf.stack(loss_vals)
        """
        grads es el valor de la sensibilidad de cada filtro de los de la última capa de convolución 
        pooled graps es la agrupación de la media de cada filtro:
        grads [[...], [...]..,num filtros]
        poledgrads [mean[...],mean[...0],etc]
        """
    grads=tape.gradient(loss,conv_outputs)[0] # calculamos gradientes con la misma cinta
    pooled_grads=tf.reduce_mean(grads,axis=(0,1)) # umbralizamos, si filtro es importante en muchas clasificaciones, entonces es importante, si no no lo es
    conv_outputs=conv_outputs[0] # eliminamos columna batch
    #Cada gradiente se multiplica por su peso de la última capa
    #Se suman todos los canales
    """
    Lo que hacemos en el heatmap es superponer todos los filtros, de forma que los que tengan más sensibilidad
    pesarán o tendrán más presencia en el heatmap que los que no, entonces para hacer eso multiplicamos cada filtro
    por su peso, es como bajar la opacidad en una imagen o cuando hacemos máscaras 
    """
    heatmap=tf.reduce_sum(tf.multiply(pooled_grads,conv_outputs),axis=-1)
    # Elimino valores negativos, porque solo buscamos regiones cercanas, no regiones leganas, no normalizamos 0-1 y evitamos que se divida por 0 con 1e-8
    heatmap=tf.maximum(heatmap,0)/(tf.reduce_max(heatmap)+1e-8) # aqui se normaliza, por eso en show_gradcam desnormalizamos
    return heatmap.numpy() #convertimos a numpy poder representarlo con opencv

"""
Función que muestra un overlay Grad-CAM para una imagen dada. Antes guardábamos 3 tipos de imágenes (original, heatmap y overlay); ahora solo las mostramos.
"""
def label_to_text(label):
    try:
        return "malignant" if int(label)==1 else "benign"
    except Exception:
        return str(label)

def show_gradcam_example(model,config,image_path,last_conv_layer_name,label=None,threshold=0.5):
    # Sabemos que vamos a trabajar con imágenes, por ello debemos convertirlas de tensores a imágenes,
    # hacemos lo mismo que en la función decode_image de create dataset
    img_decoded=decode_image(tf.constant(image_path))
    # Imagen preprocesada (igual que dataset)
    img=preprocess_image(img_decoded,False,config)
    #El overlay debe hacerse sobre valores de imagen reales, no normalizados.
    img_array=tf.expand_dims(img,axis=0) # expandimos a una columna para inferir con grdacam, recordemos que en la función de 
    # make_gradcam_heatmap se toman 4 dimensiones (1,H,W,C) pero aqui de primeras tenemos (H,W,C) tras decodificar y procesasr.
    heatmap=make_gradcam_heatmap(model,img_array,last_conv_layer_name)
    heatmap=np.uint8(255*heatmap) # Desnormalizamos -> se normalizó en make_gradcam_heatmap
    heatmap=np.expand_dims(heatmap,axis=-1) #make_graccam_heatmap devuelve 2 dimensiones (Wc,Wc), expando a (Hc,Wc,1) para operaciones como resize (no permite con 2 dimnensiones)
    heatmap=tf.image.resize(heatmap,config["img_size"]).numpy().astype(np.uint8)
    # Muy importante ya que en Grad-CAM se calcula a resolución de la capa convolucional, aqui devolvemos la dimensiíon orignial
    # Predicción para mostrar en la cabecera
    pred_prob=float(model.predict(img_array,verbose=0).ravel()[0])
    pred_label=1 if pred_prob>=threshold else 0
    fig,axes=plt.subplots(1,2,figsize=(8,4)) # original y heatmap
    axes[0].imshow(img_decoded)
    if label is not None:
        axes[0].set_title(f"Original (label: {label_to_text(label)})")
    else:
        axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(heatmap[..., 0],cmap="jet") # jet -> azul baja importancia, rojo alta importancia
    axes[1].set_title(f"Heatmap (pred: {label_to_text(pred_label)} | p={pred_prob:.2f})")
    axes[1].axis("off")
    plt.tight_layout() # evitamos cortar textos 
    plt.show()
    return fig

"""
Función que se encarga de evaluar e imprimir cualquier métrica o representación adecuada para el análisis del modelo. 
Muestra la matriz de confusión y Grad-CAMs en pantalla; no guarda imágenes.
"""
def run_eval_and_artifacts(model,ds_bundle,threshold,npz_path=None,gradcam_dir=None,last_conv_layer_name=None,save_path=None,prefix="TEST"):
    print(f"\nEvaluación en {prefix}:")
    model.evaluate(ds_bundle["test_ds"],steps=ds_bundle["test_steps"], verbose=2)
    metrics,cm,y_true,y_prob,y_pred=evaluate_binary(model,ds_bundle["test_ds"],ds_bundle["test_steps"],threshold)
    print("\nMétricas finales:",json.dumps(metrics,indent=2))
    print(classification_report(y_true,y_pred,target_names=["benign","malignant"]))
    plot_confusion_matrix(cm)
    if npz_path:
        os.makedirs(os.path.dirname(npz_path),exist_ok=True)
        np.savez(npz_path,y_true=y_true,y_prob=y_prob,y_pred=y_pred,threshold=threshold)
        print(f"Probabilidades y etiquetas guardadas en {npz_path}")
    if gradcam_dir:
        sample_paths=ds_bundle["test_imgs"]
        sample_labels=ds_bundle.get("test_labels",[None]*len(sample_paths))
        n_samples=min(3,len(sample_paths))
        if n_samples==0:
            print("No hay imágenes de test para Grad-CAM.")
        else:
            rng=np.random.default_rng()
            selected_idx=rng.choice(len(sample_paths),size=n_samples,replace=False)
            for i,idx in enumerate(selected_idx,1):
                img_path=sample_paths[idx]
                label=sample_labels[idx] if idx < len(sample_labels) else None
                try:
                    show_gradcam_example(model,ds_bundle["config"],img_path,last_conv_layer_name,label=label,threshold=threshold)
                except Exception:
                    print("No se pudo generar Grad-CAM")
                    print(traceback.format_exc())
    if save_path:
        os.makedirs(os.path.dirname(save_path),exist_ok=True)
        # Permite guardar sólo pesos si se pasa un nombre de archivo de pesos
        if str(save_path).endswith(".weights.h5"):
            model.save_weights(save_path)
        else:
            model.save(save_path)
        print(f"Modelo guardado en {save_path}")
    return metrics,cm,y_true,y_prob,y_pred

def focal_loss(y_true,y_pred,alpha=0.25,gamma=2.0,label_smoothing=0.05):
    y_true=tf.cast(y_true,tf.float32)
    y_pred=tf.clip_by_value(tf.cast(y_pred,tf.float32),1e-7,1.0-1e-7)
    if label_smoothing>0:
        y_true=y_true*(1.0-label_smoothing)+0.5*label_smoothing
    p_t=y_true*y_pred+(1.0-y_true)*(1.0-y_pred)
    alpha_t=y_true*alpha+(1.0-y_true)*(1.0-alpha)
    loss=-alpha_t*tf.pow(1.0-p_t,gamma)*tf.math.log(p_t)
    return tf.reduce_mean(loss)
