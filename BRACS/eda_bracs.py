import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

# Base por defecto: BRACS/BRACS_RoI/latest_version
DEFAULT_BASE=(Path(__file__).resolve().parent/"BRACS"/"BRACS_RoI"/"latest_version_resized")


def parse_args()->argparse.Namespace:
    parser=argparse.ArgumentParser("EDA sencillo para BRACS (clasificación multiclase).")
    parser.add_argument("--base-dir",type=str,default=str(DEFAULT_BASE),help="Raíz de BRACS_RoI/latest_version (contiene train/val/test).")
    parser.add_argument("--no-plots",action="store_true",help="No mostrar gráficas.")
    return parser.parse_args()


def main():
    args=parse_args()
    base_dir=Path(args.base_dir)

    splits=["train","val","test"]
    records=[]
    class_dirs=[]

    # Leer dataset y sacar paciente del nombre (sin funciones, directo)
    for split in splits:
        split_dir=base_dir/split
        if not split_dir.is_dir():
            continue
        for cls in sorted([d for d in os.listdir(split_dir) if (split_dir/d).is_dir()]):
            class_dirs.append(cls)
            cls_dir=split_dir/cls
            for fname in os.listdir(cls_dir):
                if not fname.lower().endswith((".png",".jpg",".jpeg")):
                    continue
                stem=Path(fname).stem
                parts=stem.split("_")
                if len(parts)>=3:
                    patient_id=parts[1]
                else:
                    patient_id=stem
                records.append({"filepath":str(cls_dir/fname),"filename":fname,"split":split,"class_name":cls,"patient_id":patient_id})

    if not records:
        raise ValueError(f"No se encontraron imágenes en {base_dir}")

    df=pd.DataFrame(records)
    classes=sorted(list(set(class_dirs)))
    label_map={}
    i=0
    while i<len(classes):
        label_map[classes[i]]=i
        i+=1
    df["label"]=df["class_name"].map(label_map)

    # Resumen rápido para saber qué demonios estamos cargando.
    print("Resumen general")
    print(f"Base:{base_dir}")
    print(f"Total imágenes:{len(df)}")
    print(f"Pacientes únicos:{df['patient_id'].nunique()}")
    print(f"Clases:{label_map}")
    print(f"Splits:{df['split'].value_counts().to_dict()}")

    # Distribución por clase y split
    dist_global=df["class_name"].value_counts().sort_index()
    print("\nDistribución global por clase:")
    for cls,cnt in dist_global.items():
        print(f"{cls}:{cnt}")

    print("\nDistribución por split y clase:")
    dist_split=df.groupby(["split","class_name"]).size().reset_index(name="count")
    for _,row in dist_split.iterrows():
        print(f"{row['split']}:{row['class_name']}->{row['count']}")

    # Pacientes por clase
    print("\nPacientes únicos por clase (global):")
    patients_per_class=df.groupby("class_name")["patient_id"].nunique()
    for cls,cnt in patients_per_class.items():
        print(f"{cls}:{cnt}")

    # Pacientes mixtos 
    patient_class_counts=df.groupby("patient_id")["class_name"].nunique()
    mixed=(patient_class_counts>1).sum()
    total_patients=len(patient_class_counts)
    mixed_ratio=0.0
    if total_patients>0:
        mixed_ratio=mixed/total_patients*100.0
    print(f"\nPacientes con múltiples clases:{mixed}/{total_patients}({mixed_ratio:.1f}%)")

    # Estadísticos imágenes por paciente
    imgs_per_patient=df.groupby("patient_id").size()
    stats={"mean":imgs_per_patient.mean(),"median":imgs_per_patient.median(),"std":imgs_per_patient.std()
           ,"min":imgs_per_patient.min(),"max":imgs_per_patient.max(),"q1":imgs_per_patient.quantile(0.25),"q3":imgs_per_patient.quantile(0.75)}
    print("\nImágenes por paciente (stats):")
    print(f"Promedio:{stats['mean']:.2f}")
    print(f"Mediana:{stats['median']:.2f}")
    print(f"Desviación estándar:{stats['std']:.2f}")
    print(f"Rango:min={stats['min']:.0f},Q1={stats['q1']:.0f},Q3={stats['q3']:.0f},max={stats['max']:.0f}")

    # Calidad de datos (nulos/duplicados)
    print("\nCalidad de datos (nulos/duplicados)")
    nulls=df.isnull().sum()
    for col,cnt in nulls.items():
        print(f"{col}:{cnt}")

    dup_paths_rows=df.duplicated(subset="filepath",keep=False).sum()
    dup_names_rows=df.duplicated(subset="filename",keep=False).sum()
    repeated_names_multi_path=(df.groupby("filename")["filepath"].nunique()>1).sum()
    print(f"Rutas duplicadas(filas afectadas):{dup_paths_rows}")
    print(f"Nombres de archivo duplicados(filas afectadas):{dup_names_rows}")
    print(f"Nombres repetidos en rutas distintas:{repeated_names_multi_path}")

    patient_split_counts=df.groupby("patient_id")["split"].nunique()
    leak_patients=patient_split_counts[patient_split_counts>1]
    print(f"Pacientes en múltiples splits(posible fuga):{len(leak_patients)}")
    if len(leak_patients)>0:
        ejemplo=", ".join(leak_patients.index[:10])
        if ejemplo:
            print(f"Ejemplo:{ejemplo}")

    # Integridad de imágenes y resoluciones
    img_meta=[]
    corrupted=[]
    for _,row in df.iterrows():
        path=row["filepath"]
        if not Path(path).exists():
            corrupted.append({"filepath":path,"error":"No existe"})
            continue
        try:
            file_size_kb=Path(path).stat().st_size/1024
            with Image.open(path) as img:
                width,height=img.size
                img.load()
            img_meta.append({"filepath":path,"width":width,"height":height,"file_size_kb":file_size_kb,"split":row["split"],"class_name":row["class_name"]})
        except Exception as exc:
            corrupted.append({"filepath":path,"error":str(exc)})

    meta_df=pd.DataFrame(img_meta)
    print("\nIntegridad y resoluciones de imágenes")
    print(f"Imágenes analizadas:{len(meta_df)}")
    print(f"Imágenes corruptas/no legibles:{len(corrupted)}")
    if corrupted:
        for item in corrupted[:10]:
            print(f"-{item['filepath']}:{item['error']}")
        if len(corrupted)>10:
            print(f"...y{len(corrupted)-10} más")

    if meta_df.empty:
        print("No se pudieron extraer resoluciones.")
    else:
        width_stats=meta_df["width"].describe()
        height_stats=meta_df["height"].describe()
        size_stats=meta_df["file_size_kb"].describe()

        print("Ancho(px)->min/mediana/promedio/max:",f"{width_stats['min']:.0f}/{width_stats['50%']:.0f}/{width_stats['mean']:.2f}/{width_stats['max']:.0f}")
        print("Alto(px)->min/mediana/promedio/max:",f"{height_stats['min']:.0f}/{height_stats['50%']:.0f}/{height_stats['mean']:.2f}/{height_stats['max']:.0f}")
        print("Tamaño archivo(KB)->min/mediana/promedio/max:",f"{size_stats['min']:.1f}/{size_stats['50%']:.1f}/{size_stats['mean']:.1f}/{size_stats['max']:.1f}")

        res_counts=meta_df.groupby(["width","height"]).size().reset_index(name="count")
        res_counts=res_counts.sort_values("count",ascending=False)
        top_res=res_counts.head(5)
        print(f"Resoluciones únicas:{len(res_counts)}")
        print("Top resoluciones más frecuentes:")
        for _,row in top_res.iterrows():
            print(f"{int(row['width'])}x{int(row['height'])}:{int(row['count'])} imágenes")

        per_split=meta_df.groupby(["split","width","height"]).size().reset_index(name="count")
        for split in meta_df["split"].unique():
            top_split=per_split[per_split["split"]==split]
            top_split=top_split.sort_values("count",ascending=False).head(3)
            if top_split.empty:
                continue
            print(f"Top resoluciones en {split}:")
            for _,row in top_split.iterrows():
                print(f"{int(row['width'])}x{int(row['height'])}:{int(row['count'])} imágenes")

    # Graficas
    if not args.no_plots:
        plt.style.use("ggplot")
        fig,ax=plt.subplots(1,2,figsize=(12,5))
        counts=df["class_name"].value_counts().sort_index()
        sns.barplot(x=counts.index,y=counts.values,ax=ax[0],color="steelblue",edgecolor="black")
        ax[0].set_title("Distribución por clase (global)")
        ax[0].set_xlabel("Clase")
        ax[0].set_ylabel("Imágenes")
        ax[0].tick_params(axis="x",rotation=45)
        ax[1].pie(counts.values,labels=counts.index,autopct="%1.1f%%",startangle=90)
        ax[1].set_title("Porcentaje por clase (global)")
        plt.tight_layout()
        plt.show(block=True)

        fig2,ax2=plt.subplots(figsize=(10,6))
        split_cls=df.groupby(["split","class_name"]).size().reset_index(name="count")
        sns.barplot(data=split_cls,x="split",y="count",hue="class_name",ax=ax2)
        ax2.set_title("Distribución por split y clase")
        ax2.set_ylabel("Imágenes")
        plt.tight_layout()
        plt.show(block=True)

        fig3,ax3=plt.subplots(figsize=(8,5))
        imgs_per_patient=df.groupby("patient_id").size()
        ax3.hist(imgs_per_patient,bins=30,color="skyblue",edgecolor="black")
        ax3.set_title("Distribución de imágenes por paciente")
        ax3.set_xlabel("Imágenes por paciente")
        ax3.set_ylabel("Pacientes")
        plt.tight_layout()
        plt.show(block=True)


if __name__=="__main__":
    main()
