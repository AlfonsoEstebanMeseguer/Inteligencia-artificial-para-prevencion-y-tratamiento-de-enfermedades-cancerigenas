import os, math, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import (confusion_matrix,classification_report,roc_auc_score,average_precision_score,accuracy_score)
from src.config.readDataset import read_binary_breakhis_data
from src.config.split_dataset import split_by_patient, split_by_image
from src.config.create_dataset import load_split,create_dataset,compute_class_weights,decode_image,preprocess_image

"""
Esta función se encarga únicamente de crear cualquier.json que pueda faltar tras split. Simplemente ofrece mayor eficacia al programa en caso de que borremos los scripts
previamente al entrenamiento
"""
def ensure_splits(base_path,split_dir,train_size=0.8,val_size=0.1,test_size=0.1,split_mode="patient"):
    # Normalizamos la carpeta de splits según el modo solicitado.
    split_dir=resolve_split_dir(split_dir,split_mode)
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
        _,all_images,all_labels,_,slides=read_binary_breakhis_data(base_path,verbose=False)
        if split_mode=="patient":
            # Split estratificado por paciente (sin fuga)
            splits,_=split_by_patient(all_images,all_labels,slides,train_size,val_size,test_size)
        else:
            # Split estratificado por clase a nivel de imagen (permite fuga)
            splits,_=split_by_image(all_images,all_labels,slides,train_size,val_size,test_size)
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
def get_datasets_basic(config,split_dir,include_labels=False):
    train_imgs,train_labels=load_split(split_dir,"train")
    val_imgs,val_labels=load_split(split_dir,"val")
    test_imgs,test_labels=load_split(split_dir,"test")
    if len(train_imgs)==0 or len(val_imgs)==0 or len(test_imgs)==0:
        raise RuntimeError("Algún split está vacío")
    train_ds=create_dataset(train_imgs,train_labels,True,config)
    val_ds=create_dataset(val_imgs,val_labels,False,config)
    test_ds=create_dataset(test_imgs,test_labels,False,config)
    steps_per_epoch=math.ceil(len(train_imgs)/config["batch_size"])
    val_steps=math.ceil(len(val_imgs)/config["batch_size"])
    test_steps=math.ceil(len(test_imgs)/config["batch_size"])
    if config["use_class_weights"]:
        class_weights=compute_class_weights(train_labels)
    else:
        class_weights=None
    num_classes=int(np.max(train_labels))+1
    out={"config":config,"train_ds":train_ds,"val_ds":val_ds,"test_ds":test_ds,"steps_per_epoch":steps_per_epoch,
         "val_steps":val_steps,"test_steps":test_steps,"class_weights":class_weights,"test_imgs":test_imgs,
         "test_labels":test_labels,"num_classes":num_classes}
    if include_labels:
        out.update({"train_labels":np.array(train_labels,dtype=np.int32),"val_labels":np.array(val_labels,dtype=np.int32),"test_labels":np.array(test_labels,dtype=np.int32)})
    return out

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
    plt.close(fig)
    return fig

"""
Función que genera heatmap Grad-CAM para una imagen preprocesada. Lo primero es construir un modelo que devuelva la predicción final del modelo, (model.output)
y la última capa convolucional del modelo en cuestión. Este modelo es necesario pues necesitamos la predicción real (no la aproximada por alguna entropía o modificación
, ej sigmoide o binarycrossentropy), es decir, el valor real predicho por la última capa convolucional. Y, necesitamos el resultado final del modelo, para calcular gradientes de 
calor-frio. Usamos GradientTape, proporcionado por tensorflow para registrar todos los cálculos realizados en una cinta, que se usará para la representación. De seguido,
"""
def make_gradcam_heatmap(model, img_array,last_conv_layer_name):
    grad_model=tf.keras.models.Model([model.inputs],[model.get_layer(last_conv_layer_name).output,model.output])
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
def show_gradcam_example(model,config,image_path,last_conv_layer_name):
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
    fig,axes=plt.subplots(1,2,figsize=(8,4)) # original y heatmap
    axes[0].imshow(img_decoded)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(heatmap[..., 0],cmap="jet") # jet -> azul baja importancia, rojo alta importancia
    axes[1].set_title("Heatmap")
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
        sample_paths=ds_bundle["test_imgs"][:3]
        for i,img_path in enumerate(sample_paths,1):
            try:
                show_gradcam_example(model,ds_bundle["config"],img_path,last_conv_layer_name)
            except Exception:
                print("No se pudo generar Grad-CAM")
    if save_path:
        os.makedirs(os.path.dirname(save_path),exist_ok=True)
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
