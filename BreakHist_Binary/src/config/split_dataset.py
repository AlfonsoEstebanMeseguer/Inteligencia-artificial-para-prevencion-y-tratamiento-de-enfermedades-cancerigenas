import os
import json
import argparse 
import numpy as np
import sys
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit, train_test_split
import matplotlib.pyplot as plt

PROJECT_ROOT=Path(__file__).resolve().parents[3]
BINARY_ROOT=Path(__file__).resolve().parents[2]

for p in (PROJECT_ROOT,BINARY_ROOT):
    sp=str(p)
    if sp not in sys.path:
        sys.path.insert(0,sp)

from src.config.readDataset import read_binary_breakhis_data
from BreakHist_Multiclass.config.readDataset import read_multiclass_breakhis_data

DEFAULT_BASE_PATH_BINARY = os.environ.get("BREAKHIS_BASE",os.path.join(os.getcwd(),"BreakHist_Binary","BreakHist","data"
                                                                       ,"BreakHis - Breast Cancer Histopathological Database"
                                                                       ,"dataset_cancer_v1","dataset_cancer_v1"
                                                                       ,"classificacao_binaria"))

DEFAULT_BASE_PATH_MULTICLASS = os.environ.get("BREAKHIS_BASE_MULTICLASS",os.path.join(os.getcwd(),"BreakHist","data"
                                                                                     ,"BreakHis - Breast Cancer Histopathological Database"
                                                                                     ,"dataset_cancer_v1","dataset_cancer_v1"
                                                                                     ,"classificacao_multiclasse"))

def _build_stats(splits,dataset_type,all_labels=None,label_map=None):
    if dataset_type=="multiclass":
        if label_map is None:
            raise ValueError("label_map requerido para estadísticas multiclase")
        num_classes=int(np.max(all_labels))+1 if all_labels is not None else len(label_map)
        inv_map={v:k for k,v in label_map.items()}
        stats={}
        for split_name,split_data in splits.items():
            labels=split_data["labels"]
            slides_split=split_data["slides"]
            counts=[0]*num_classes
            for y in labels:
                counts[int(y)]+=1
            class_counts_named={}
            for i in range(num_classes):
                name=inv_map.get(i,f"class_{i}")
                class_counts_named[name]=counts[i]
            total_imgs=len(labels)
            class_ratios_named={}
            for name,count in class_counts_named.items():
                class_ratios_named[name]=count/total_imgs if total_imgs else 0.0
            stats[split_name]={"num_images":total_imgs,"num_patients":len(set(slides_split)),"slides":list(slides_split),
                               "class_counts":counts,"class_counts_named":class_counts_named,
                               "class_ratios_named":class_ratios_named}
        return stats
    stats={}
    for split_name,split_data in splits.items():
        labels=split_data["labels"]
        stats[split_name]={"num_images":len(split_data["images"]),"num_patients":len(set(split_data["slides"]))
                           ,"num_benign":labels.count(0),"num_malignant":labels.count(1)
                           ,"ratio_malig_benign":(labels.count(1)/max(labels.count(0),1))}
    return stats

"""
Función para dividir el dataset BreakHis por paciente en train/val/test según las proporciones indicadas.
Usamos GroupShuffleSplit de sklearn para evitar fuga de datos entre splits.

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
def split_by_patient(all_images,all_labels,slides,train_size,val_size,test_size,random_state,dataset_type,label_map):
    # Comprobamos que las proporciones suman 1.0
    if abs(train_size+val_size+test_size-1.0) >= 1e-6:
        raise ValueError("Las proporciones deben sumar 1.0")
    # Convertimos a arrays para indexar de forma vectorizada sin alterar datos
    all_images=np.array(all_images)
    all_labels=np.array(all_labels)
    slides=np.array(slides)
    # GroupSuffleSplit no garantiza proporciones exactas de imágenes, pero sí de grupos, aunque ahora mismo 
    # no se indica el grupo, solo el tamaño del primer split (train).
    # Agrupa imágenes por paciente, las trata todas como una unidad y no las separa nunca.
    # Necesario porque queremos evitar fuga
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
    stats=_build_stats(splits,dataset_type,all_labels=all_labels,label_map=label_map)

    # Verificación de no fuga de datos (forma sencilla)
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
    img_train,img_temp,y_train,y_temp,slides_train,slides_temp=train_test_split(all_images,all_labels,slides
                                                                                ,train_size=train_size,stratify=all_labels,random_state=random_state)
    val_ratio=val_size/(val_size+test_size)
    img_val,img_test,y_val,y_test,slides_val,slides_test=train_test_split(img_temp,y_temp,slides_temp,test_size=val_ratio
                                                                          ,stratify=y_temp,random_state=random_state)
    splits={"train":{"images":img_train.tolist(),"labels":y_train.tolist(),"slides":slides_train.tolist()}
            ,"val":{"images":img_val.tolist(),"labels":y_val.tolist(),"slides":slides_val.tolist()}
            ,"test":{"images":img_test.tolist(),"labels":y_test.tolist(),"slides":slides_test.tolist()}}
    stats=_build_stats(splits,dataset_type,all_labels=all_labels,label_map=label_map)

    # Comprobamos explícitamente la fuga de datos (esperada) entre splits a nivel de imagen
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
        bottom=np.zeros(len(splits))
        fig,ax=plt.subplots(figsize=(10,6))
        for c in range(num_classes):
            heights=[]
            for s in splits:
                heights.append(stats[s]["class_counts"][c])
            ax.bar(x,heights,bottom=bottom,label=class_names[c])
            bottom=bottom+np.array(heights)
        ax.set_xticks(x)
        labels=[]
        for s in splits:
            labels.append(s.upper())
        ax.set_xticklabels(labels)
        ax.set_xlabel("Split")
        ax.set_ylabel("Número de imágenes (apilado por clase)")
        ax.set_title("Distribución de clases por split")
        ax.legend()
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
        for split_name in splits:
            benign_counts.append(stats[split_name]['num_benign'])
            malignant_counts.append(stats[split_name]['num_malignant'])
        # num benignas y num malignas
        x=range(len(splits)) # pos bar
        width=0.35 #ancho bar
        fig,ax=plt.subplots(figsize=(10, 6))
        ax.bar(x,benign_counts,width,label='Benign',color='tab:blue') # azul -> benignas
        ax.bar(x,malignant_counts,width,bottom=benign_counts,label='Malignant',color='tab:orange') # naranja -> maligna 
        # bottom = benign_counts para que aparezca encima: decisión de diseño -> total columna = benigna + maligna
        ax.set_xlabel('Split')
        ax.set_ylabel('Número de imágenes')
        ax.set_title('Distribución de clases por split (imágenes)')
        ax.set_xticks(x)
        labels=[]
        for s in splits:
            labels.append(s.upper())
        ax.set_xticklabels(labels)
        ax.legend()
        ax.grid(True,axis='y',alpha=0.3)
        for idx in range(len(splits)):
            total=benign_counts[idx]+malignant_counts[idx]
            ax.text(idx,total+50,f'{total}',ha='center',fontweight='bold')
        # total de imágenes por split encima de cada columna.
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
python split_dataset.py --base-path "D:\\TFG_BREAKHIST\\BreakHist\\data\\BreakHis - Breast Cancer Histopathological Database\\dataset_cancer_v1\\dataset_cancer_v1\\classificacao_binaria" --output-dir splits --train-size 0.8 --val-size 0.1 --test-size 0.1 --random-state 42 --no-verbos
"""
def parse_arguments():
    parser = argparse.ArgumentParser("Split del dataset BreakHis por paciente (por defecto 80/10/10).")
    parser.add_argument("--dataset-type",default="binary",choices=["binary","multiclass"],help="Tipo de dataset: binary o multiclass.")
    parser.add_argument("--base-path",default=None,help="Ruta raíz del dataset BreakHis.")
    parser.add_argument("--output-dir",default="splits",help="Directorio de salida para los JSON y las gráficas.")
    parser.add_argument("--train-size",type=float, default=0.8,help="Proporción de entrenamiento (default 0.8).")
    parser.add_argument("--val-size",type=float, default=0.1,help="Proporción de validación (default 0.1).")
    parser.add_argument("--test-size",type=float, default=0.1,help="Proporción de test (default 0.1).")
    parser.add_argument("--random-state",type=int, default=42,help="Reproducibilidad (default 42).")
    parser.add_argument("--split-mode",default="patient",choices=["patient","image"],help="patient (sin fuga, por paciente) o image (estratificado por clase, con fuga).")
    parser.add_argument("--no-plot",action="store_true",help="No mostrar gráficas.")
    return parser.parse_args()

if __name__=="__main__" and __package__ is None:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

if __name__=="__main__":
    args=parse_arguments()
    if args.base_path is None:
        if args.dataset_type=="multiclass":
            args.base_path=DEFAULT_BASE_PATH_MULTICLASS
        else:
            args.base_path=DEFAULT_BASE_PATH_BINARY
    split_subdir="split_patient" if args.split_mode=="patient" else "split_imagen"
    base_output=Path(args.output_dir)
    if not base_output.is_absolute():
        base_output=Path(__file__).resolve().parents[2]/base_output
    output_root=base_output/split_subdir
    output_root.mkdir(parents=True,exist_ok=True)

    if args.dataset_type=="multiclass":
        data,all_images,all_labels,label_map,slides=read_multiclass_breakhis_data(base_path=args.base_path,verbose=False)
    else:
        data,all_images,all_labels,label_map,slides=read_binary_breakhis_data(args.base_path)
    if args.split_mode=="patient":
        splits,stats=split_by_patient(all_images,all_labels,slides,args.train_size,args.val_size,args.test_size,args.random_state
                                      ,dataset_type=args.dataset_type,label_map=label_map)
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
        if args.dataset_type=="multiclass":
            num_classes=len(stats[splits[0]]["class_counts"])
            inv_map={}
            for k,v in label_map.items():
                inv_map[v]=k
            class_names=[]
            for i in range(num_classes):
                class_names.append(inv_map.get(i,f"c{i}"))
        plot_split_distributions(stats,dataset_type=args.dataset_type,class_names=class_names)
