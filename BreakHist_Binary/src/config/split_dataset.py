import os
import json
import argparse 
import numpy as np
import sys
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit, train_test_split
import matplotlib.pyplot as plt
# Imports diferidos: requieren que `sys.path` incluya PROJECT_ROOT y BINARY_ROOT
from src.config.readDataset import read_binary_breakhis_data
from BreakHist_Multiclass.config.readDataset import read_multiclass_breakhis_data
# Gestion de rutas para importar módulos de src y BreakHist_Multiclass sin importar desde dónde se ejecute el script
PROJECT_ROOT=Path(__file__).resolve().parents[3]
BINARY_ROOT=Path(__file__).resolve().parents[2]

for p in (PROJECT_ROOT,BINARY_ROOT):
    sp=str(p)
    if sp not in sys.path:
        sys.path.insert(0,sp)



if __name__=="__main__" and __package__ is None:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

DEFAULT_BASE_PATH_BINARY = os.environ.get("BREAKHIS_BASE",os.path.join(str(PROJECT_ROOT),"BreakHist_Binary","BreakHist","data"
                                                                       ,"BreakHis - Breast Cancer Histopathological Database","dataset_cancer_v1"
                                                                       ,"dataset_cancer_v1","classificacao_binaria"))
DEFAULT_BASE_PATH_MULTICLASS = os.environ.get(
    "BREAKHIS_BASE_MULTICLASS",os.path.join(str(PROJECT_ROOT),"BreakHist_Multiclass""BreakHist","data"
                                            ,"BreakHis - Breast Cancer Histopathological Database","dataset_cancer_v1","dataset_cancer_v1","classificacao_multiclasse"))

"""
Función que construye estadísticas detalladas en función del tipo de clasificación (binaria /mutliclase) de cada split
"""
def build_stats(splits,dataset_type,all_labels,label_map):
    if dataset_type=="multiclass":
        if label_map is None:
            raise ValueError("label_map requerido para estadísticas multiclase")
        num_classes=int(np.max(all_labels))+1 # para stats y plots interesa contar de 1-8 que de 0-7 para subclases
        inv_map={}
        for k in label_map:
            inv_map[label_map[k]]=k #mapa inverso para mostrar nombres de clases en vez de índices
        stats={}
        for split_name in splits:
            split_data=splits[split_name]
            labels=split_data["labels"]
            slides_split=split_data["slides"]
            counts=[0]*num_classes
            # realizamos conteo de imágenes por clase 
            for y in labels:
                idx=int(y) # aseguramos entero
                counts[idx]+=1
            # Conteo de pacientes por clase
            patient_counts=[0]*num_classes
            patient_label={}
            for pid,y in zip(slides_split,labels):
                y_int=int(y) # aseguramos entero
                if pid not in patient_label:
                    patient_label[pid]=y_int # creamos entrada de paciente si no existe, asignamos su etiqueta (se asume que cada paciente tiene una única etiqueta, se vio en EDA y documentación del dataset)
            for pid in patient_label: # conteo
                lbl=patient_label[pid]
                patient_counts[int(lbl)]+=1
            class_counts_named={}
            i=0
            # Construimos diccionario con nombres de clases para mostrar en estadísticas y gráficas, si no hay nombre, se muestra class_i
            while i < num_classes:
                name=inv_map.get(i,"class_%s" % i)
                class_counts_named[name]=counts[i]
                i+=1
            class_patient_counts_named={}
            i=0
            while i < num_classes:
                name=inv_map.get(i,"class_%s" % i)
                # asignamos conteo de pacientes por clase con nombre
                class_patient_counts_named[name]=patient_counts[i]
                i+=1
            total_imgs=len(labels)
            class_ratios_named={}
            for name in class_counts_named:
                count=class_counts_named[name]
                # si no hay imágenes en el split, evitamos división por cero y asignamos ratio 0.0, 
                # lo dejamos así para evitar errores y mostrar que no hay representación de esa clase en el split
                if total_imgs:
                    class_ratios_named[name]=count/total_imgs
                else:
                    class_ratios_named[name]=0.0
            unique_patients={}
            for pid in slides_split:
                if pid not in unique_patients:
                    unique_patients[pid]=1
            stats[split_name]={"num_images":total_imgs,"num_patients":len(unique_patients),"slides":list(slides_split)
                               ,"class_counts":counts,"class_counts_named":class_counts_named,"class_patient_counts":patient_counts
                               ,"class_patient_counts_named":class_patient_counts_named,"class_ratios_named":class_ratios_named}
    else:
        stats={}
        for split_name in splits:
            split_data=splits[split_name]
            labels=split_data["labels"]
            slides_split=split_data["slides"]
            unique_patients={}
            # Conteo de pacientes únicos en el split para comprobaciones de fugas 
            for pid in slides_split:
                if pid not in unique_patients:
                    unique_patients[pid]=1
            num_benign=0
            num_malignant=0
            # Conteo de labels
            for y in labels:
                if int(y)==0:
                    num_benign+=1
                else:
                    num_malignant+=1
            # Conteo de pacientes por clase (se asume una sola etiqueta por paciente)
            patient_label={}
            for pid,y in zip(slides_split,labels):
                y_int=int(y)
                if pid not in patient_label:
                    patient_label[pid]=y_int
            num_benign_patients=0
            num_malignant_patients=0
            for pid in patient_label:
                if patient_label[pid]==0:
                    num_benign_patients+=1
                else:
                    num_malignant_patients+=1
            # Control para que si no hay benignos en el split, no se pueda dividir entre 0
            if num_benign > 0:
                denom=num_benign 
            else:  
                denom=1 
            stats[split_name]={"num_images":len(split_data["images"]),"num_patients":len(unique_patients)
                            ,"num_benign":num_benign,"num_malignant":num_malignant
                            ,"num_benign_patients":num_benign_patients,"num_malignant_patients":num_malignant_patients
                            ,"ratio_malig_benign":(num_malignant/denom)}
    return stats

"""
Función que asigna la cantidad de pacientes a cada split según las proporciones indicadas.
Se asegura que la suma de pacientes asignados a cada split sea igual al total, incluso si hay restos al convertir a enteros.
"""
def allocate_split_counts(total,train_size,val_size,test_size):
    if total < 0:
        raise ValueError("Total de pacientes inválido.")
    if total == 0:
        raise ValueError("Total de pacientes 0.")
    # numeros totales de pacientes por split
    raw_train=total*train_size
    raw_val=total*val_size
    raw_test=total*test_size
    counts=[int(raw_train),int(raw_val),int(raw_test)] # aseguramos enteros
    remainder=total-(counts[0]+counts[1]+counts[2])
    if remainder > 0: # si sobra, lo asignamos al split con mayor fracción decimal
        fracs=[raw_train-counts[0],raw_val-counts[1],raw_test-counts[2]] #restos
        for x in range(remainder):
            max_idx=0 
            # encontramos el split con mayor fracción
            if fracs[1] > fracs[max_idx]: 
                max_idx=1
            if fracs[2] > fracs[max_idx]:
                max_idx=2
            # asignamos 1 al split correspondiente
            counts[max_idx]+=1
            fracs[max_idx]=-1
    return counts

"""
Mapeo de etiquetas de imágenes a idpaciente, esto lo usaremos a la hora de estratificar por paciente, la idea es que los splits pueden salir un poco ineficientes, ej
de 81 pacientes de breakhist, 24 son beningos, al hacer el tipico split ya sea 80/10/10 o 70/15/15, puede que en el conjunto de validación&test solo caigan 6 
pacientes benignos, lo que hace que el ratio maligno/benigno se dispare y de resultados no representativos ni reproducibles (depende de que tan fáciles de predecir
sean esos pocos pacientes benignos que caen en val/test). Para ello, usaremos estratificación por clase a nivel de paciente, de 
forma que cada split tenga una distribución más ""estable"" de clases. Para ello debemos saber la etiqueta correspondiente a cada paciente, y se sabe que 
cada paciente tiene una única etiqueta (benigno o maligno), (se vio en EDA y en la documentación del dataset).
"""
def patient_level_labels(slides,labels):
    patient_to_label={}
    for patient,label in zip(slides,labels):
        label=int(label) # aseguramos entero
        if patient not in patient_to_label:
            patient_to_label[patient]=label # creamos entrada de paciente si no existe 
    patient_ids=[]
    patient_labels=[]
    for patient,label in patient_to_label.items():
        patient_ids.append(patient)
        patient_labels.append(label)
    return patient_ids,patient_labels

"""
Función que realiza un split estratificado por clases a nivel de paciente. Importante destacar que se busca mantener la distribución de clases en cada split, no la
proporción exacta de pacientes. Esto con el fin de evitar que en splits pequeños (ej: val/test) haya muy pocos pacientes de una clase y se dispare la distribución.
Además, hacemos shuffle de los pacientes antes de asignarlos a cada split, para evitar sesgos en el orden original, lo que significa que habrá mas variabilidad en los 
splits generados (importante para evitar sesgos en los resultados finales), si por falta de representacion por ejemplo en val hay muchos benignos de alguna subclase que 
el modelo NO ha aprendido tan bien (debido a los datos OJO), lo que se buscaría ahora sería intentar que almenos no siempre estén la misma clase de imágenes.

Importante entender que 
si, poor falta de representación, el modelo tiende a predecir peor benignos, dentro de esos benignos,  habrán más fáciles y más difíciles, 
si por mala suerte en val/test caen los más difíciles SEGUIDOS, el rendimiento se verá muy afectado. 
Esto es inevitable, pero al menos con el shuffle y la estratificación por clase a nivel de paciente, se busca minimizar este efecto.
"""
def split_patients_stratified(patient_ids,patient_labels,train_size,val_size,test_size,random_state):
    seed=np.random.RandomState(random_state) # generador de semilla (usamos 42 por defecto para que sea reproducible (mismo split siempre))
    class_to_patients={} 
    for pid,label in zip(patient_ids,patient_labels):
        if label not in class_to_patients:
            class_to_patients[label]=[] # diccionario con entrada por clase
        class_to_patients[label].append(pid) # pacientes de cada clase
    splits={}
    splits["train"]=[]
    splits["val"]=[]
    splits["test"]=[]
    for label in class_to_patients:
        patients=class_to_patients[label]
        seed.shuffle(patients) # muy immportante, no buscamos sesgar ultimos resultados
        counts=allocate_split_counts(len(patients),train_size,val_size,test_size)
        train_count=counts[0]
        val_count=counts[1]
        test_count=counts[2]
        splits["train"].extend(patients[:train_count])
        splits["val"].extend(patients[train_count:train_count+val_count])
        splits["test"].extend(patients[train_count+val_count:train_count+val_count+test_count])
    return splits

"""
Función para dividir el dataset BreakHis por paciente en train/val/test según las proporciones indicadas.
Estratificamos por clase a nivel de paciente para mantener la distribución y evitar fuga.

Esto es necesario sobretodo para trabajos de histologías, ya que:
Si tenemos muchas imágenes pero pocos pacientes, realmente tenemos varias imágenes del mismo TUMOR del paciente. Imágenes además tomadas en unas ciertas 
cirtunstancias (misma preparación de la muestra, mismo microscopio, mismo técnico, etc). Más que el tumor tendrá características del paciente
Por tanto, si separamos por imagen, es muy probable que tengamos imágenes del mismo pacietumor en train y en val/test, lo que provoca fuga de datos
y sobreestimación del rendimiento real del modelo.

Problema:
En el conjunto val, hay muy pocos pacientes benignos (24 en total), con solo 8 pacientes en validación, basta con que:
entren 1–2 pacientes benignos y varios malignos “grandes” -> el ratio se dispara. Esto significa que el conjunto de validación no 
es representativo del conjunto completo.

Quizás los resultados no sean tan buenos, pero al menos evitamos la fuga de datos, esto ya es mejor que cualquier otro paper que ofrezca
mejores resultados pero FALSOS.

devuelve: 
    splits: diccionario con los splits generados
    stats: estadísticas de cada split (número de imágenes, pacientes, distribución de clases, ratio maligno/benigno)
"""
def split_by_patient(all_images,all_labels,slides,train_size,val_size,test_size,random_state,dataset_type,label_map
                     ,stratify_patients=True):
    # Comprobamos que las proporciones suman 1.0
    if abs(train_size+val_size+test_size) < 1.0:
        raise ValueError("Las proporciones deben sumar 1.0")
    # Convertimos a arrays para indexar de forma vectorizada sin alterar datos
    all_images=np.array(all_images)
    all_labels=np.array(all_labels)
    slides=np.array(slides)
    if stratify_patients:
        patient_ids,patient_labels=patient_level_labels(slides,all_labels)
        patient_splits=split_patients_stratified(patient_ids,patient_labels,train_size,val_size,test_size,random_state)
        train_patients=set(patient_splits["train"])
        val_patients=set(patient_splits["val"])
        test_patients=set(patient_splits["test"])
        # where devuelve los índices donde se cumple la condición de la lista de pacientes del subjconjunto dado con los pacientes de slides (DONDE SE ENCUENTRA LA INTERSECCIÓN)
        train_idx=np.where(np.isin(slides,list(train_patients)))[0]
        val_idx=np.where(np.isin(slides,list(val_patients)))[0]
        test_idx=np.where(np.isin(slides,list(test_patients)))[0]
    else:
        # GroupSuffleSplit no garantiza proporciones exactas de imágenes, pero sí de grupos.
        # Agrupa imágenes por paciente, las trata todas como una unidad y no las separa nunca.
        gss_1=GroupShuffleSplit(n_splits=1,train_size=train_size,random_state=random_state)
        train_idx,temp_idx=next(gss_1.split(all_images,all_labels,slides)) #Índices de train y (val+test) -> aquí indicamos los grupos (pacientes)
        val_ratio=val_size/(val_size+test_size) # donde cae el split val/test
        gss_2=GroupShuffleSplit(n_splits=1,train_size=val_ratio,random_state=random_state) # segundo split
        # Los índices val y test
        val_idx_rel, test_idx_rel=next(gss_2.split(all_images[temp_idx],all_labels[temp_idx],slides[temp_idx]))
        val_idx=temp_idx[val_idx_rel]
        test_idx=temp_idx[test_idx_rel]
    # Construccion del diccionario de splits
    splits={"train":{"images": all_images[train_idx].tolist(),"labels":all_labels[train_idx].tolist(),"slides":slides[train_idx].tolist()}
            ,"val":{"images": all_images[val_idx].tolist(),"labels":all_labels[val_idx].tolist(),"slides":slides[val_idx].tolist()}
            ,"test":{"images": all_images[test_idx].tolist(),"labels":all_labels[test_idx].tolist(),"slides":slides[test_idx].tolist()}}
    stats=build_stats(splits,dataset_type,all_labels=all_labels,label_map=label_map)
    # Verificación de no fuga de datos 
    train_patients=set(splits["train"]["slides"])
    val_patients=set(splits["val"]["slides"])
    test_patients=set(splits["test"]["slides"])
    leakage_train_val=list(train_patients & val_patients)
    leakage_train_test=list(train_patients & test_patients)
    leakage_val_test=list(val_patients & test_patients)
    if leakage_train_val or leakage_train_test or leakage_val_test:
        raise ValueError(f"Data leakage detectado: train_val={leakage_train_val}, train_test={leakage_train_test}, val_test={leakage_val_test}")
    return splits,stats

"""
Función que realiza un split estratificado por clases a nivel de imagen (permite fuga de datos entre pacientes).
Se usa solo para comparar resultados con data leakage.
"""
def split_by_image(all_images,all_labels,slides,train_size,val_size,test_size,random_state,dataset_type,label_map):
    if abs(train_size+val_size+test_size-1.0) >= 1e-6:
        raise ValueError("Las proporciones deben sumar 1.0")
    all_images=np.array(all_images)
    all_labels=np.array(all_labels)
    slides=np.array(slides)
    img_train,img_temp,y_train,y_temp,slides_train,slides_temp=train_test_split(all_images,all_labels,slides,train_size=train_size,stratify=all_labels,random_state=random_state)
    val_ratio=val_size/(val_size+test_size)
    img_val,img_test,y_val,y_test,slides_val,slides_test=train_test_split(img_temp,y_temp,slides_temp,test_size=val_ratio,stratify=y_temp,random_state=random_state)
    splits={"train":{"images":img_train.tolist(),"labels":y_train.tolist(),"slides":slides_train.tolist()}
            ,"val":{"images":img_val.tolist(),"labels":y_val.tolist(),"slides":slides_val.tolist()}
            ,"test":{"images":img_test.tolist(),"labels":y_test.tolist(),"slides":slides_test.tolist()}}
    stats=build_stats(splits,dataset_type,all_labels=all_labels,label_map=label_map)
    # Comprobamos de fuga de datos
    train_patients=set(splits["train"]["slides"])
    val_patients=set(splits["val"]["slides"])
    test_patients=set(splits["test"]["slides"])
    leakage_train_val=list(train_patients & val_patients)
    leakage_train_test=list(train_patients & test_patients)
    leakage_val_test=list(val_patients & test_patients)
    if leakage_train_val or leakage_train_test or leakage_val_test:
        print(f"Data leakage detectado (esperado en split por imagen): train_val={len(leakage_train_val)}, train_test={len(leakage_train_test)}, val_test={len(leakage_val_test)}")
    else:
        print("No se detectó data leakage en split por imagen (inusual; revisa los parámetros).")
    return splits,stats

# Función para plotear las distribuciones de cada split
def plot_split_distributions(stats,dataset_type,class_names=None):
    # Extraemos métricas precalculadas de forma directa
    splits=list(stats.keys())
    if dataset_type=="multiclass":
        num_classes=len(stats[splits[0]]["class_counts"])
        if class_names is None:
            class_names=[f"c{i}" for i in range(num_classes)]
        x=np.arange(len(splits))
        total_width=0.8
        bar_width=total_width/num_classes
        fig,ax=plt.subplots(figsize=(12,6))
        for c in range(num_classes):
            heights=[]
            for s in splits:
                heights.append(stats[s]["class_counts"][c])
            offsets=x - total_width/2 + (c+0.5)*bar_width
            ax.bar(offsets,heights,width=bar_width,label=class_names[c])
            for i,split_name in enumerate(splits):
                h=heights[i]
                if h <= 0:
                    continue
                patient_counts=stats[split_name].get("class_patient_counts")
                patients=patient_counts[c] if patient_counts is not None else stats[split_name]["num_patients"]
                ax.text(offsets[i],h+max(1,h*0.01),f"{h}\nP:{patients}",ha="center",va="bottom",fontsize=7)
        ax.set_xticks(x)
        labels=[]
        for s in splits:
            labels.append(s.upper())
        ax.set_xticklabels(labels)
        ax.set_xlabel("Split")
        ax.set_ylabel("Número de imágenes")
        ax.set_title("Distribución de clases por split")
        ax.legend(ncol=2)
        ax.grid(True,axis="y",alpha=0.3)
        plt.tight_layout()
        plt.show()
        fig2,ax2=plt.subplots(figsize=(8,5))
        patients=[]
        for s in splits:
            patients.append(stats[s]["num_patients"])
        ax2.bar(splits,patients,color="steelblue",edgecolor="black")
        ax2.set_ylabel("Pacientes únicos")
        ax2.set_title("Pacientes únicos por split")
        ax2.grid(True,axis="y",alpha=0.3)
        for i,v in enumerate(patients):
            ax2.text(i,v+0.5,f"{v}",ha="center",fontweight="bold")
        plt.tight_layout()
        plt.show()
    else:
        benign_counts=[]
        malignant_counts=[]
        benign_patients=[]
        malignant_patients=[]
        for split_name in splits:
            benign_counts.append(stats[split_name]['num_benign'])
            malignant_counts.append(stats[split_name]['num_malignant'])
            benign_patients.append(stats[split_name].get("num_benign_patients",stats[split_name]["num_patients"]))
            malignant_patients.append(stats[split_name].get("num_malignant_patients",stats[split_name]["num_patients"]))
        # Barras agrupadas por split (una por clase)
        x=np.arange(len(splits))
        width=0.35
        fig,ax=plt.subplots(figsize=(10, 6))
        bars_b=ax.bar(x-width/2,benign_counts,width,label='Benign',color='tab:blue') # azul -> benignas
        bars_m=ax.bar(x+width/2,malignant_counts,width,label='Malignant',color='tab:orange') # naranja -> maligna
        ax.set_xlabel('Split')
        ax.set_ylabel('Número de imágenes')
        ax.set_title('Distribución de clases por split')
        ax.set_xticks(x)
        labels=[]
        for s in splits:
            labels.append(s.upper())
        ax.set_xticklabels(labels)
        ax.legend()
        ax.grid(True,axis='y',alpha=0.3)
        y_pad=max(1,max(benign_counts+malignant_counts)*0.01)
        for idx,bar in enumerate(bars_b):
            h=benign_counts[idx]
            if h>0:
                ax.text(bar.get_x()+bar.get_width()/2,h+y_pad,f"{h}\nP:{benign_patients[idx]}",ha='center',va='bottom',fontsize=8)
        for idx,bar in enumerate(bars_m):
            h=malignant_counts[idx]
            if h>0:
                ax.text(bar.get_x()+bar.get_width()/2,h+y_pad,f"{h}\nP:{malignant_patients[idx]}",ha='center',va='bottom',fontsize=8)
        plt.tight_layout()
        # mostramos en pantalla
        plt.show()
        # obtenemos ratio 
        ratios=[]
        for split_name in splits:
            ratios.append(stats[split_name]['ratio_malig_benign'])

        fig,ax=plt.subplots(figsize=(8,5))
        x_pos=list(range(len(splits)))
        bars=ax.bar(x_pos, ratios,color='steelblue',edgecolor='black')
        # barra por split
        ax.axhline(2.19,color='red',linestyle='--',label='Ratio global (2.19:1)')
        # ratio maligno/benigno del dataset completo
        # Si hay algun split que no corresponda con la distribución del dataset -> entonces los splits no son representativos
        ax.set_title('Ratio maligno/benigno por split')
        ax.set_ylabel('Ratio')
        ax.set_xlabel('Split')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([s.upper() for s in splits])
        ax.legend()
        ax.grid(True,axis='y',alpha=0.3)
        for idx,bar in enumerate(bars):
            ratio_val=ratios[idx]
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.1,f'{ratio_val:.2f}:1',ha='center',fontweight='bold')
        # ratio numérico
        plt.tight_layout()
        # mostramos en pantalla
        plt.show()

"""
Función para parsear argumentos de línea de comandos 
"""
def parse_arguments():
    parser = argparse.ArgumentParser("Split del dataset BreakHis por paciente (por defecto 70/15/15).")
    parser.add_argument("--dataset-type",default="binary",choices=["binary","multiclass"],help="Tipo de dataset: binary o multiclass.")
    parser.add_argument("--base-path",default=None,help="Ruta raíz del dataset BreakHis.")
    parser.add_argument("--output-dir",default="splits",help="Directorio de salida para los JSON y las gráficas.")
    parser.add_argument("--train-size",type=float, default=0.7,help="Proporción de entrenamiento (default 0.7).")
    parser.add_argument("--val-size",type=float, default=0.15,help="Proporción de validación (default 0.15).")
    parser.add_argument("--test-size",type=float, default=0.15,help="Proporción de test (default 0.15).")
    parser.add_argument("--random-state",type=int, default=42,help="Reproducibilidad (default 42).")
    parser.add_argument("--split-mode",default="patient",choices=["patient","image"],help="patient (sin fuga, por paciente) o image (estratificado por clase, con fuga).")
    parser.add_argument("--no-stratify-patients",action="store_true",help="Desactiva la estratificación por clase al dividir por paciente.")
    parser.add_argument("--no-plot",action="store_true",help="No mostrar gráficas.")
    return parser.parse_args()

if __name__=="__main__":
    args=parse_arguments()
    # Resolución de rutas
    if args.base_path is None:
        if args.dataset_type=="multiclass":
            args.base_path=DEFAULT_BASE_PATH_MULTICLASS
        else:
            args.base_path=DEFAULT_BASE_PATH_BINARY
    # modo split
    if args.split_mode=="patient":
        split_subdir="split_patient" 
    else:
        split_subdir="split_imagen"

    # Configuración de salida
    base_output=Path(args.output_dir)
    if not base_output.is_absolute():
        base_output=PROJECT_ROOT/base_output
    output_root=base_output/split_subdir
    output_root.mkdir(parents=True,exist_ok=True)
    
    # delegacion de lectura del dataset
    if args.dataset_type=="multiclass":
        data,all_images,all_labels,label_map,slides=read_multiclass_breakhis_data(base_path=args.base_path,verbose=False)
    else:
        data,all_images,all_labels,label_map,slides=read_binary_breakhis_data(args.base_path)

    # Delegación del split según modo seleccionado
    if args.split_mode=="patient":
        splits,stats=split_by_patient(all_images,all_labels,slides,args.train_size,args.val_size,args.test_size,args.random_state
                                      ,dataset_type=args.dataset_type,label_map=label_map,stratify_patients=(not args.no_stratify_patients))
    else:
        splits,stats=split_by_image(all_images,all_labels,slides,args.train_size,args.val_size,args.test_size,args.random_state
                                    ,dataset_type=args.dataset_type,label_map=label_map)
        
    print("\nESTADÍSTICAS DEL SPLIT:")
    if args.dataset_type=="multiclass":
        inv_map={}
        for k,v in label_map.items():
            inv_map[v]=k
        for split,s in stats.items():
            print(f"\n{split.upper()}")
            print(f"Imágenes:{s['num_images']}")
            print(f"Pacientes:{s['num_patients']}")
            for cls_idx,count in enumerate(s["class_counts"]):
                cls_name=inv_map.get(cls_idx,f"class_{cls_idx}")
                print(f"{cls_name}({cls_idx}):{count}")
            print("Ratios por clase:")
            for cls_name in s["class_ratios_named"]:
                ratio=s["class_ratios_named"][cls_name]
                print(f"{cls_name}:{ratio:.2f}%")
    else:
        for split,s in stats.items():
            print(f"\n{split.upper()}")
            print(f"Imágenes:{s['num_images']}")
            print(f"Pacientes:{s['num_patients']}")
            print(f"Benignas:{s['num_benign']}")
            print(f"Malignas:{s['num_malignant']}")
            print(f"Ratio maligno/benigno: {s['ratio_malig_benign']:.2f}:1")

    for split_name,split_data in splits.items():
        # Creamos el split para cargarlo en ejecución, si no lo guardáramos en un archivo tendria que crearse en tiempo de ejecución -> techo por límite hardware.
        # Se ha elegido guardarlo en .json por comodidad de lectura, podría haber sido un csv con rutas y labels
        path=output_root/f"{split_name}.json"
        with open(path,"w",encoding="utf-8") as f:
            json.dump(split_data,f,indent=2)

    print(f"Archivos guardados en: {output_root}")
    if not args.no_plot:
        class_names=None
        # En caso de multiclase, obtenemos nombres de clases para gráficas
        if args.dataset_type=="multiclass":
            num_classes=None
            for split_name in stats:
                num_classes=len(stats[split_name]["class_counts"])
                break
            inv_map={}
            for k in label_map:
                inv_map[label_map[k]]=k
            class_names=[]
            i=0
            while i < num_classes:
                name=inv_map.get(i,"c%s" % i)
                class_names.append(name)
                i+=1
        plot_split_distributions(stats,dataset_type=args.dataset_type,class_names=class_names)
