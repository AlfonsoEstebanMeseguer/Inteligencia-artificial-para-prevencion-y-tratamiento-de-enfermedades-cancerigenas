import os
import json
import argparse
import numpy as np
import sys
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit,train_test_split
import matplotlib.pyplot as plt
from BreakHist_Multiclass.config.readDataset import read_multiclass_breakhis_data

DEFAULT_BASE_PATH=os.environ.get("BREAKHIS_BASE_MULTICLASS",os.path.join(os.getcwd(),"BreakHist","data"
                                ,"BreakHis - Breast Cancer Histopathological Database","dataset_cancer_v1"
                                ,"dataset_cancer_v1","classificacao_multiclasse"))

# Solo se comentarán diferencias respecto a versión binary
def split_by_patient(all_images,all_labels,slides,label_map,train_size=0.8,val_size=0.1,test_size=0.1,random_state=42):
    if abs(train_size+val_size+test_size-1.0)>=1e-6:
        raise ValueError("Las proporciones deben sumar 1.0")
    
    all_images=np.array(all_images)
    all_labels=np.array(all_labels)
    slides=np.array(slides)
    gss_1=GroupShuffleSplit(n_splits=1,train_size=train_size,random_state=random_state)
    train_idx,temp_idx=next(gss_1.split(all_images,all_labels,groups=slides))
    val_ratio=val_size/(val_size+test_size)
    gss_2=GroupShuffleSplit(n_splits=1,train_size=val_ratio,random_state=random_state)
    val_idx_rel,test_idx_rel=next(gss_2.split(all_images[temp_idx],all_labels[temp_idx],groups=slides[temp_idx]))
    val_idx=temp_idx[val_idx_rel]
    test_idx=temp_idx[test_idx_rel]
    splits={"train":{"images":all_images[train_idx].tolist(),"labels":all_labels[train_idx].tolist(),"slides":slides[train_idx].tolist()}
            ,"val":{"images":all_images[val_idx].tolist(),"labels":all_labels[val_idx].tolist(),"slides":slides[val_idx].tolist()}
            ,"test":{"images":all_images[test_idx].tolist(),"labels":all_labels[test_idx].tolist(),"slides":slides[test_idx].tolist()}}
    """
    en el caso binario sabíamos que eran 2 y era más fácil sacar proporciones, pero aqui hay más entonces 
    debemos sacar el número de clases y hacer operaciones con ellas

    inv_map es el inverso a un label map,
    ej      label_map = {"benign": 0}
            inv_map = {0: "benign"
    Lo primero que hacemos es crear una lista de 0 de tamaño num_clases (8) en la que iremos sumando por cada aparición
    de subclase. Luego más adelante lo que hacemos es que una vez que tenemos para cada num de label sus conteos 
    obtenemos el nombre del label y le ponemos el conteos de su label correspondiente. Para esto se requiere un label map inverso
    porque va d indice ->  a nombre es decir get num = nombre, y no al revés
    """
    num_classes=int(all_labels.max())+1 
    inv_map={} 
    for k,v in label_map.items():
        inv_map[v]=k
    stats={}
    for split_name,split_data in splits.items():
        labels=split_data["labels"]
        slides_split=split_data["slides"]
        counts=[0]*num_classes
        for y in labels:
            counts[int(y)]+=1 # sumamos 1 a cada label encontrado
        class_counts_named={}
        for i in range(num_classes):
            name=inv_map.get(i) # aqui obtenemos el nombre a partir del índice
            class_counts_named[name]=counts[i]
        total_imgs=len(labels) 
        class_ratios_named={}
        # Encontramos ratio de tal forma
        # si todas son el 100% , que % es cada una
        for name,count in class_counts_named.items():
            if total_imgs:
                class_ratios_named[name]=count/total_imgs
            else:
                class_ratios_named[name]=0.0
        stats[split_name]={"num_images":total_imgs,"num_patients":len(set(slides_split)),"slides":list(slides_split),
                           "class_counts":counts,"class_counts_named":class_counts_named
                           ,"class_ratios_named":class_ratios_named}

    # igual que en caso binario
    train_patients=set(splits["train"]["slides"])
    val_patients=set(splits["val"]["slides"])
    test_patients=set(splits["test"]["slides"])
    leakage_train_val=list(train_patients&val_patients)
    leakage_train_test=list(train_patients&test_patients)
    leakage_val_test=list(val_patients&test_patients)
    if leakage_train_val or leakage_train_test or leakage_val_test:
        raise ValueError(f"Data leakage detectado: train_val={leakage_train_val}, train_test={leakage_train_test}, val_test={leakage_val_test}")

    return splits,stats


def split_by_image(all_images,all_labels,slides,label_map,train_size=0.8,val_size=0.1,test_size=0.1,random_state=42):
    if abs(train_size+val_size+test_size-1.0)>=1e-6:
        raise ValueError("Las proporciones deben sumar 1.0")

    all_images=np.array(all_images)
    all_labels=np.array(all_labels)
    slides=np.array(slides)
    img_train,img_temp,y_train,y_temp,slides_train,slides_temp=train_test_split(all_images,all_labels
                                                                                ,slides,train_size=train_size
                                                                                ,stratify=all_labels
                                                                                ,random_state=random_state)
    val_ratio=val_size/(val_size+test_size)
    img_val,img_test,y_val,y_test,slides_val,slides_test=train_test_split(img_temp,y_temp,slides_temp
                                                                          ,train_size=val_ratio,stratify=y_temp
                                                                          ,random_state=random_state)

    splits={"train":{"images":img_train.tolist(),"labels":y_train.tolist(),"slides":slides_train.tolist()}
            ,"val":{"images":img_val.tolist(),"labels":y_val.tolist(),"slides":slides_val.tolist()}
            ,"test":{"images":img_test.tolist(),"labels":y_test.tolist(),"slides":slides_test.tolist()}}
    
    num_classes=int(all_labels.max())+1
    inv_map={}
    for k,v in label_map.items(): # mismo que en split_patient
        inv_map[v]=k    
    stats={}
    for split_name,split_data in splits.items():
        labels=split_data["labels"]
        slides_split=split_data["slides"]
        counts=[0]*num_classes
        for y in labels:
            counts[int(y)]+=1
        class_counts_named={}
        for i in range(num_classes):
            name=inv_map.get(i)
            class_counts_named[name]=counts[i]
        total_imgs=len(labels)
        class_ratios_named={}
        for name,count in class_counts_named.items():
            if total_imgs:
                class_ratios_named[name]=count/total_imgs
            else:
                class_ratios_named[name]=0.0
        stats[split_name]={"num_images":total_imgs,"num_patients":len(set(slides_split)),"slides":list(slides_split),
                           "class_counts":counts,"class_counts_named":class_counts_named,"class_ratios_named":class_ratios_named}

    train_patients=set(splits["train"]["slides"])
    val_patients=set(splits["val"]["slides"])
    test_patients=set(splits["test"]["slides"])
    leakage_train_val=list(train_patients&val_patients)
    leakage_train_test=list(train_patients&test_patients)
    leakage_val_test=list(val_patients&test_patients)
    if leakage_train_val or leakage_train_test or leakage_val_test:
        print(f"Data leakage detectado (esperado en split por imagen): train_val={len(leakage_train_val)}, train_test={len(leakage_train_test)}, val_test={len(leakage_val_test)}")
    else:
        print("No se detectó data leakage en split por imagen (inusual; revisa los parámetros).")
    return splits,stats

# Solo se comenarán diferencias respecto a Binary
def plot_split_distributions(stats):
    # Extraemos métricas precalculadas de forma directa
    splits=list(stats.keys()) 
    num_classes=len(stats[splits[0]]["class_counts"])   # aqui no podemos obtenerlo como stats[split_name]['num_benign'])
                                                        # porque no sería robusto para cada subclase, así que lo debemos 
                                                        # obtener por bucle
    if class_names is None:
        class_names=[]
        for i in range(num_classes):
            class_names.append(f"c{i}")
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
    plt.show(block=True)
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
    plt.show(block=True)


def parse_arguments():
    parser=argparse.ArgumentParser(description="Split del dataset BreakHis multiclase.")
    parser.add_argument("--base-path",default=DEFAULT_BASE_PATH,help="Ruta raíz del dataset BreakHis.")
    parser.add_argument("--output-dir",default="splits",help="Directorio de salida para los JSON y las gráficas.")
    parser.add_argument("--train-size",type=float,default=0.8,help="Proporción de entrenamiento (default 0.8).")
    parser.add_argument("--val-size",type=float,default=0.1,help="Proporción de validación (default 0.1).")
    parser.add_argument("--test-size",type=float,default=0.1,help="Proporción de test (default 0.1).")
    parser.add_argument("--random-state",type=int,default=42,help="Reproducibilidad (default 42).")
    parser.add_argument("--split-mode",default="patient",choices=["patient","image"],help="patient (sin fuga) o image (estratificado por clase, con fuga).")
    parser.add_argument("--no-plot",action="store_true",help="No mostrar gráficas.")
    return parser.parse_args()

if __name__=="__main__" and __package__ is None:
    project_root=Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0,str(project_root))

if __name__=="__main__":
    args=parse_arguments()
    os.makedirs(args.output_dir,exist_ok=True)
    data,all_images,all_labels,label_map,slides=read_multiclass_breakhis_data(base_path=args.base_path,verbose=False)
    if args.split_mode=="patient":
        splits,stats=split_by_patient(all_images,all_labels,slides,label_map,train_size=args.train_size
                                      ,val_size=args.val_size,test_size=args.test_size,random_state=args.random_state)
    else:
        splits,stats=split_by_image(all_images,all_labels,slides,label_map,train_size=args.train_size
                                    ,val_size=args.val_size,test_size=args.test_size,random_state=args.random_state)

    print("\nESTADÍSTICAS DEL SPLIT:")
    inv_map={}
    for k,v in label_map.items(): # Igual que en split_patient
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

    for split_name,split_data in splits.items():
        path=os.path.join(args.output_dir,f"{split_name}.json")
        with open(path,"w",encoding="utf-8") as f:
            json.dump(split_data,f,indent=2)

    print(f"\nArchivos guardados en: {args.output_dir}")
    plot_split_distributions(stats)
