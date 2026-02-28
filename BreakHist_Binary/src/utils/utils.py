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
Función que garantiza la existencia de splits en la ruta dada, con el modo y tipo de dataset solicitados. 
Si falta algún split o si las rutas dentro de los splits no son válidas, se regeneran completamente a partir del dataset original. 
Además, si se solicita cobertura de clases en multiclase, se valida que cada split contenga al menos una muestra de cada clase 
(solo en modo paciente, ya que en modo imagen es más difícil garantizarlo sin fuga). 
Esto asegura que el pipeline de entrenamiento siempre tenga datos válidos y completos para funcionar correctamente.
"""
def ensure_splits(base_path,split_dir,train_size,val_size,test_size,split_mode,dataset_type,random_state,ensure_all_classes):
    if random_state is None:
        random_state=42 #seed reproducible 
    # Normalizamos la carpeta de splits según el modo solicitado solo si aún no incluye el sufijo esperado.
    if split_mode=="patient": 
        target_suffix="split_patient" 
    else:
        target_suffix="split_imagen"
    # gestión de rutas
    if Path(split_dir).name != target_suffix:
        split_dir=resolve_split_dir(split_dir,split_mode)
    read_data,split_by_patient_fn,split_by_image_fn,_,_,_= resolve_dataset_helpers(dataset_type)
    missing=[]
    for s in ["train","val","test"]:
        split_path=os.path.join(split_dir,f"{s}.json")
        if not os.path.exists(split_path):
            missing.append(split_path) # agregamos split no encontrado

    regenerate=False
    if missing:
        regenerate=True
        print(f"Faltan splits, se regenerarán: {', '.join(missing)}")
    else:
        # Validamos rutas existentes r
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

    # Si se solicita cobertura de clases en multiclase, comprobamos los splits existentes.
    if not regenerate and ensure_all_classes and dataset_type=="multiclass" and split_mode=="patient":
        try:
            _,_,_,label_map,_=read_data(base_path,verbose=False)
            expected_classes=set(label_map.values())
        except Exception as e:
            print(f"No se pudo cargar label_map para validar clases: {e}")
            expected_classes=None
        coverages={}
        for s in ["train","val","test"]:
            split_path=os.path.join(split_dir,f"{s}.json")
            try:
                with open(split_path,"r",encoding="utf-8") as f:
                    data=json.load(f)
                coverages[s]=set(int(x) for x in data.get("labels",[]))
            except Exception as e:
                print(f"No se pudo validar cobertura en {split_path}: {e}")
                regenerate=True
                break
        if not regenerate and coverages:
            if expected_classes is None:
                expected_classes=set().union(*coverages.values())
            missing_per_split={s:sorted(list(expected_classes-coverages[s])) for s in coverages if expected_classes-coverages[s]}
            if missing_per_split:
                print(f"Faltan clases en splits ({missing_per_split}). Se regenerarán.")
                regenerate=True

    if regenerate:
        # Preparamos carpeta y leemos dataset completo para dividir
        os.makedirs(split_dir,exist_ok=True)
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
Esta función devuelve un conjunto de funciones específicas para el tipo de dataset solicitado (binario o multiclase). 
Esto permite que el resto del pipeline pueda usar funciones genéricas sin preocuparse por las diferencias entre tipos de 
dataset, ya que cada función se adapta internamente a las necesidades de su tipo específico.
"""
def resolve_dataset_helpers(dataset_type):
    if dataset_type=="binary":
        return (read_binary_breakhis_data, split_by_patient_shared, split_by_image_shared,
                load_split_binary, create_dataset_binary, compute_class_weights_binary)
    if dataset_type=="multiclass":
        return (read_multiclass_breakhis_data, split_by_patient_shared, split_by_image_shared,
                load_split_multiclass, create_dataset_multiclass, compute_class_weights_multiclass)
    raise ValueError(f"Tipo de dataset no soportado: {dataset_type}")

"""
Esta función crea datasets tf.data listos para model.fit con steps calculados. Obtiene los datos listos para tesorflow (imagenes, labels), con ellos crea los datasets,
calcula los steps con: ceil(redondeo hacia arriba) y NO división exacta (porque que, por ejemplo para 100 imágenes y tamaño de batch 32, habrían 3,125 pasos, que hacemos con el último?
, si no redondeamos hacia arriba, estaremos perdiendo imágenes por cada step, por ello forzamos a usar más num steps antes que menos).

Importante: aunque SOLO train será infinito, (no tiene sentido en val y test, solo buscamos validar), también calculo steps en val y test porque forzamos a que se evalúen todas
las imágenes, incluso si el batch es incompleto.

Tras obtener los datasets, los steps y los pesos (si procede), devuelve un diccionario con la información de configuración lista para entrenamiento.
"""
def get_datasets_basic(config,split_dir,include_labels,dataset_type):
    _,_,_,load_split_fn,create_dataset_fn,compute_class_weights_fn=resolve_dataset_helpers(dataset_type)
    train_imgs,train_labels=load_split_fn(split_dir,"train")
    val_imgs,val_labels=load_split_fn(split_dir,"val")
    test_imgs,test_labels=load_split_fn(split_dir,"test")
    # check rapido para binario, solo etiquetas 0,1
    if dataset_type=="binary":
        all_labels=train_labels+val_labels+test_labels
        unique_labels=[]
        for x in all_labels:
            x=int(x) # aseguramos entero
            if x not in unique_labels:
                unique_labels.append(x)
        invalid_labels=[]
        for x in unique_labels: #chequeamos que solo haya 0 y 1, si hay algo más, error
            if x!=0 and x!=1:
                invalid_labels.append(x)
        if invalid_labels:
            raise ValueError(f"Labels no binarios detectados en splits: {sorted(invalid_labels)}. "f"Usa splits binarios o el pipeline multiclase.")
    if len(train_imgs)==0 or len(val_imgs)==0 or len(test_imgs)==0:
        raise RuntimeError("Algún split está vacío")

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
    out={"config":config,"train_ds":train_ds,"val_ds":val_ds,"test_ds":test_ds,"steps_per_epoch":steps_per_epoch
         ,"val_steps":val_steps,"test_steps":test_steps,"class_weights":class_weights,"test_imgs":test_imgs
         ,"test_labels":test_labels,"num_classes":num_classes}
    if include_labels:
        out.update({"train_labels":np.array(train_labels,dtype=np.int32),"val_labels":np.array(val_labels,dtype=np.int32),"test_labels":np.array(test_labels,dtype=np.int32)})
    return out


"""
Función que plotea gráficas del entrenamiento .Cada métrica de train se muestra con una línea para el entrenamiento y una 
línea discontinua para la validación, usando el mismo color para ambas. Además, se se plotea tambiñén una leyendapara mejorar la legibilidad.
"""
def plot_training_history(history,metrics,save_path,show):
    # history de Keras o diccionario con métricas por epoch
    if hasattr(history,"history"):
        history_dict=history.history
    else:
        history_dict=history
    if metrics is None:
        # Si no se especifican métricas, usamos las de entrenamiento 
        metrics=[]
        for m in history_dict.keys():
            if not m.startswith("val_"):
                metrics.append(m)
    # Filtramos métricas que existan en el historial
    filtered_metrics=[]
    for m in metrics:
        if m in history_dict:
            filtered_metrics.append(m)
    metrics=filtered_metrics
    if len(metrics)==0:
        print("No hay métricas para graficar.")
        return None
    fig,ax=plt.subplots(figsize=(10,5))
    color_cycle=plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for idx,metric in enumerate(metrics):
        color=color_cycle[idx % len(color_cycle)]
        # Curva de entrenamiento
        ax.plot(history_dict[metric],label=f"train_{metric}",color=color,linestyle="-")
        val_key=f"val_{metric}"
        if val_key in history_dict:
            # Curva de validación
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
Función que evalúa en test y devuelve métricas y predicciones. Obtenemos todo test y comprobamos predicción del modelo ya entrenado con predict donde
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
def evaluate_binary(model,test_ds,test_steps):
    y_true=[]
    y_prob=[]
    y_pred=[]
    for batch_imgs,batch_labels in test_ds.take(test_steps):
        preds=model.predict(batch_imgs,verbose=1).ravel()
        y_true.append(batch_labels.numpy().ravel())
        y_prob.append(preds)
    y_true=np.concatenate(y_true)
    y_prob=np.concatenate(y_prob)
    unique_labels=np.unique(y_true)
    if len(unique_labels) > 2:
        raise ValueError(f"evaluate_binary recibió etiquetas multiclase: {unique_labels}. "f"Usa el pipeline multiclase o splits binarios.")
    for prob in y_prob:
        if prob >= 0.5:
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
    
    return metrics,cm,y_true,y_prob,y_pred

"""
Función que plotea la matriz de confusión (escala de azules). Esta visualización facilita la interpretación de los resultados del modelo, 
mostrando claramente las verdaderas positivas, verdaderas negativas, falsas positivas y falsas negativas.
"""
def plot_confusion_matrix(cm,labels):
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
Función que plotea métricas finales en un gráfico de barras y devuelve un diccionario con todas las métricas calculadas,
la matriz de confusión y el reporte de clasificación.
"""
def plot_metrics_bar(metrics,title,figsize,show):
    # Solo métricas numéricas
    keys=[]
    for k in metrics:
        v=metrics[k]
        if isinstance(v,(int,float)):
            keys.append(k)
    if len(keys)==0:
        print("No hay métricas numéricas para graficar.")
        return None
    values=[]
    for k in keys:
        values.append(float(metrics[k]))
    fig,ax=plt.subplots(figsize=figsize)
    x=np.arange(len(keys))
    bars=ax.bar(x,values,color="steelblue",edgecolor="black")
    ax.set_title(title)
    ax.set_ylabel("Valor")
    ax.set_xticks(x)
    ax.set_xticklabels(keys,rotation=45,ha="right")
    ax.grid(True,axis="y",alpha=0.3)
    i=0
    for bar in bars:
        h=bar.get_height()
        offset=0.01
        if h*0.02 > offset:
            offset=h*0.02
        ax.text(bar.get_x()+bar.get_width()/2,h+offset,f"{values[i]:.3f}",ha="center",va="bottom",fontsize=8)
        i+=1
    plt.tight_layout()
    if show:
        plt.show()
    return fig

"""
Función que genera heatmap Grad-CAM para una imagen preprocesada. Lo primero es construir un modelo que devuelva la predicción final del modelo, (model.output)
y la última capa convolucional del modelo en cuestión. Este modelo es necesario pues necesitamos la predicción real (no la aproximada por alguna entropía o modificación
, ej sigmoide o binarycrossentropy), es decir, el valor real predicho por la última capa convolucional. Y, necesitamos el resultado final del modelo, para calcular gradientes de 
calor-frio. Usamos GradientTape, proporcionado por tensorflow para registrar todos los cálculos realizados en una cinta, que se usará para la representación. De seguido,
"""
def make_gradcam_heatmap(model, img_array,last_conv_layer_name,target_class):
    if last_conv_layer_name in (None, "auto"):
        # Elige la Conv2D con mayor varianza de activación para evitar mapas planos.
        conv_layers=[layer for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
        if not conv_layers:
            target_layer,parent_model=None,None
        else:
            outputs=[layer.output for layer in conv_layers]
            probe=tf.keras.models.Model(model.input,outputs)
            acts=probe(img_array,training=False)
            best_idx=0
            best_score=-1.0
            for i,act in enumerate(acts):
                # Varianza global + tamaño espacial (favorece mapas menos genéricos)
                h,w=int(act.shape[1]),int(act.shape[2])
                score=float(tf.math.reduce_std(act).numpy()) * max(h*w,1)
                if score > best_score:
                    best_score=score
                    best_idx=i
            target_layer,parent_model=conv_layers[best_idx],model
    else:
        # Busca una capa por nombre en el modelo (incluye modelos anidados).
        target_layer=None
        parent_model=None
        stack=[model]
        while stack and target_layer is None:
            current=stack.pop()
            for layer in current.layers:
                if layer.name==last_conv_layer_name:
                    target_layer=layer
                    parent_model=current
                    break
                if isinstance(layer, tf.keras.Model) and hasattr(layer, "layers"):
                    stack.append(layer)
    if target_layer is None:
        # Fallback: elige la conv con mayor varianza de activación
        conv_layers=[layer for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
        if not conv_layers:
            target_layer,parent_model=None,None
        else:
            outputs=[layer.output for layer in conv_layers]
            probe=tf.keras.models.Model(model.input,outputs)
            acts=probe(img_array,training=False)
            best_idx=0
            best_score=-1.0
            for i,act in enumerate(acts):
                # Varianza global + tamaño espacial (favorece mapas menos genéricos)
                h,w=int(act.shape[1]),int(act.shape[2])
                score=float(tf.math.reduce_std(act).numpy()) * max(h*w,1)
                if score > best_score:
                    best_score=score
                    best_idx=i
            target_layer,parent_model=conv_layers[best_idx],model
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
        p=predictions[:,0]
        # Logit para evitar saturación de la sigmoide en gradientes
        logit=tf.math.log(p+1e-7) - tf.math.log(1.0-p+1e-7)
        loss=logit if target_class==1 else -logit
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
    # Elimino valores negativos
    heatmap=tf.maximum(heatmap,0)
    heatmap=heatmap.numpy()
    # Normalización robusta por percentiles (evita "todo rojo")
    p_low=np.percentile(heatmap,20)
    p_high=np.percentile(heatmap,99)
    if p_high <= p_low:
        denom=(np.max(heatmap)-np.min(heatmap))+1e-8
        heatmap=(heatmap-np.min(heatmap))/denom
    else:
        heatmap=np.clip((heatmap-p_low)/(p_high-p_low+1e-8),0,1)
    # Gamma para realzar contraste local
    heatmap=np.power(heatmap,1.2)
    return heatmap

"""
Función que plotea un Grad-CAM para una imagen dada, mostrando la imagen original junto al heatmap. 
Permite forzar una clase objetivo específica o usar la predicción del modelo para decidirla.
"""
def show_gradcam_example(model,config,image_path,last_conv_layer_name,label,force_target_class):
    # Sabemos que vamos a trabajar con imágenes, por ello debemos convertirlas de tensores a imágenes,
    # hacemos lo mismo que en la función decode_image de create dataset
    img_decoded=decode_image(tf.constant(image_path))
    # Imagen preprocesada (igual que dataset)
    img=preprocess_image(img_decoded,False,config)
    #El overlay debe hacerse sobre valores de imagen reales, no normalizados.
    img_array=tf.expand_dims(img,axis=0) # expandimos a una columna para inferir con grdacam, recordemos que en la función de 
    # make_gradcam_heatmap se toman 4 dimensiones (1,H,W,C) pero aqui de primeras tenemos (H,W,C) tras decodificar y procesasr.
    # Predicción para decidir la clase objetivo del Grad-CAM
    pred_prob=float(model.predict(img_array,verbose=0).ravel()[0])
    if force_target_class is None:
        if label is not None:
            target_class=int(label) # aseguramos entero
        else: 
            if pred_prob>=0.5: # si la probabilidad de maligno es >=0.5, se considera maligno, si no benigno, 
                #sabemos que la salida del modelo puede no ser exactamente 0 o 1, sino un valor continuo entre ambos, por ello el umbral de 0.5 para decidir la clase objetivo del Grad-CAM
                target_class=1 
            else:
                target_class=0
    else:
        target_class=int(force_target_class)
    heatmap=make_gradcam_heatmap(model,img_array,last_conv_layer_name,target_class=target_class)
    heatmap=np.uint8(255*heatmap) # Desnormalizamos -> se normalizó en make_gradcam_heatmap
    heatmap=np.expand_dims(heatmap,axis=-1) #make_graccam_heatmap devuelve 2 dimensiones (Wc,Wc), expando a (Hc,Wc,1) para operaciones como resize (no permite con 2 dimnensiones)
    heatmap=tf.image.resize(heatmap,config["img_size"]).numpy().astype(np.uint8)
    # Muy importante ya que en Grad-CAM se calcula a resolución de la capa convolucional, aqui devolvemos la dimensiíon orignial
    # Predicción para mostrar en la cabecera
    pred_label=target_class
    fig,axes=plt.subplots(1,2,figsize=(8,4)) # original y heatmap
    axes[0].imshow(img_decoded)
    if label is not None:
        try:
            if int(label)==1:
                label_text="malignant" 
            else :
                label_text="benign"
        except Exception:
            label_text=str(label)
        axes[0].set_title(f"Original (label: {label_text})")
    else:
        axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(heatmap[..., 0],cmap="jet") # jet -> azul baja importancia, rojo alta importancia
    if int(pred_label)==1:
        pred_text="malignant"
    else:
        pred_text="benign"
    axes[1].set_title(f"Heatmap (pred: {pred_text})")
    axes[1].axis("off")
    plt.tight_layout() # evitamos cortar textos 
    plt.show()
    return fig

"""
Función que ejecuta la evaluación completa en test, incluyendo métricas, matriz de confusión, reporte de clasificación y Grad-CAM. 
Permite permite guardar el modelo si se especifica una ruta.  Se muestran los mejores ejemplos para Grad-CAM según confianza, 
o una lista fija si se proporciona. Se maneja cualquier error en la generación de Grad-CAM para asegurar que la evaluación no se interrumpa.
"""
def run_eval_and_artifacts(model,ds_bundle,gradcam_dir,last_conv_layer_name,save_path,prefix,force_target_class):
    print(f"\nEvaluación en {prefix}:")
    # Evaluación sobre el dataset de test 
    model.evaluate(ds_bundle["test_ds"],steps=ds_bundle["test_steps"], verbose=2)
    # Métricas y predicciones en bruto 
    metrics,cm,y_true,y_prob,y_pred=evaluate_binary(model,ds_bundle["test_ds"],ds_bundle["test_steps"])
    print("\nMétricas finales:",json.dumps(metrics,indent=2))
    print(classification_report(y_true,y_pred,target_names=["benign","malignant"]))
    plot_confusion_matrix(cm,("Benign","Malignant"))
    plot_metrics_bar(metrics,title=f"Métricas finales ({prefix})",figsize=(12,4),show=True)
    if gradcam_dir:
        sample_paths=ds_bundle["test_imgs"]
        #Si no existen todos los labels see rellenan con None
        sample_labels=ds_bundle.get("test_labels",[None]*len(sample_paths))
        gradcam_fixed_path=Path(ds_bundle["config"].get("gradcam_fixed_path",""))
        fixed_active=False
        if gradcam_fixed_path and gradcam_fixed_path.exists():
            try:
                with open(gradcam_fixed_path,"r",encoding="utf-8") as f:
                    fixed=json.load(f)
                fixed_paths=fixed.get("images",[])
                fixed_labels=fixed.get("labels",[])
                if fixed_paths:
                    # Si hay lista fija, se prioriza frente a muestreo por confianza
                    sample_paths=fixed_paths
                    if fixed_labels:
                        sample_labels=fixed_labels
                    else:
                        sample_labels=[None]*len(sample_paths)
                    fixed_active=True
            except Exception as e:
                print(f"No se pudo cargar gradcam_fixed: {e}")
        if len(sample_paths)==0:
            print("No hay imágenes de test para Grad-CAM.")
        else:
            # En caso de que queramos comparar las mismas imagenes con distintos modelos, pasaremos la lista de imagenes iguales
            if fixed_active:
                selected=list(range(len(sample_paths)))
            else:
                # Si no buscamos mejores candidatos por clase, se toman las primeras 6 imágenes 
                # (4 benignas y 2 malignas) para mostrar variedad sin sesgo de confianza.
                benign_candidates=[]
                malignant_candidates=[]
                for idx,prob in enumerate(y_prob):
                    # si probabilidad de maligno es >=0.5, se considera maligno, si no benigno
                    if prob>=0.5:
                        pred_label=1
                    else:
                        pred_label=0
                    if pred_label==0:
                        benign_candidates.append((idx,1.0-prob))
                    else:
                        malignant_candidates.append((idx,prob))
                # Elegimos ejemplos mejor predichos por clase
                top_benign=[]
                for cand in benign_candidates:
                    inserted=False
                    i=0
                    while i < len(top_benign):
                        if cand[1] > top_benign[i][1]:
                            top_benign.insert(i,cand)
                            inserted=True
                            break
                        i+=1
                    if not inserted:
                        top_benign.append(cand)
                    while len(top_benign) > 4:
                        top_benign.pop()
                top_malignant=[]
                for cand in malignant_candidates:
                    inserted=False
                    i=0
                    while i < len(top_malignant):
                        if cand[1] > top_malignant[i][1]:
                            top_malignant.insert(i,cand)
                            inserted=True
                            break
                        i+=1
                    if not inserted:
                        top_malignant.append(cand)
                    while len(top_malignant) > 2:
                        top_malignant.pop()
                selected=top_benign+top_malignant
            if not selected:
                print("No hay predicciones para Grad-CAM.")
            else:
                for rank,item in enumerate(selected,1):
                    if fixed_active: # si hay lista fija, el item es directamente el índice, si no, el item es una tupla 
                        # (índice, confianza de preddct)
                        idx=item
                    else:
                        idx,_=item
                    img_path=sample_paths[idx]
                    if idx < len(sample_labels):
                        label=sample_labels[idx]
                    else:
                        label=None
                    try:
                        # Grad-CAM puede fallar si no encuentra la capa (ya que tenemos muchos modelos posibles), 
                        # o por cualquier otro motivo, por ello lo envolvemos en un try-except para asegurar que el proceso de evaluación no se interrumpa.
                        show_gradcam_example(model,ds_bundle["config"],img_path,last_conv_layer_name,label=label,force_target_class=force_target_class)
                    except Exception:
                        print("No se pudo generar Grad-CAM")
                        print(traceback.format_exc())
    if save_path:
        os.makedirs(os.path.dirname(save_path),exist_ok=True)
        # guardar sólo pesos si se pasa un nombre de archivo de pesos
        if str(save_path).endswith(".weights.h5"):
            model.save_weights(save_path)
        else:
            model.save(save_path)
        print(f"Modelo guardado en {save_path}")
    return metrics,cm,y_true,y_prob,y_pred
