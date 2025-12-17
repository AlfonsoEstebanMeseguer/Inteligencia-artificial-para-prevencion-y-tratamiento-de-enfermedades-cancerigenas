import os
import json
import numpy as np
from collections import defaultdict
from sklearn.model_selection import GroupShuffleSplit
import matplotlib.pyplot as plt
from readDataset import read_binary_breakhis_data


def split_by_patient(
    all_images,
    all_labels,
    slides,
    train_size=0.8,
    val_size=0.1,
    test_size=0.1,
    random_state=42,
):
    """
    Split train/val/test por paciente (group = patient_id).
    No hay fuga de datos.

    Returns:
        dict con splits y estadísticas
    """

    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, \
        "Las proporciones deben sumar 1.0"

    all_images = np.array(all_images)
    all_labels = np.array(all_labels)
    slides = np.array(slides)

    # --------------------------------------------------
    # 1. Train vs (Val+Test)
    # --------------------------------------------------
    gss_1 = GroupShuffleSplit(
        n_splits=1,
        train_size=train_size,
        random_state=random_state
    )

    train_idx, temp_idx = next(
        gss_1.split(all_images, all_labels, groups=slides)
    )

    # --------------------------------------------------
    # 2. Val vs Test (desde temp)
    # --------------------------------------------------
    val_ratio = val_size / (val_size + test_size)

    gss_2 = GroupShuffleSplit(
        n_splits=1,
        train_size=val_ratio,
        random_state=random_state
    )

    val_idx_rel, test_idx_rel = next(
        gss_2.split(
            all_images[temp_idx],
            all_labels[temp_idx],
            groups=slides[temp_idx]
        )
    )

    val_idx = temp_idx[val_idx_rel]
    test_idx = temp_idx[test_idx_rel]

    # --------------------------------------------------
    # 3. Construir splits
    # --------------------------------------------------
    splits = {
        "train": {
            "images": all_images[train_idx].tolist(),
            "labels": all_labels[train_idx].tolist(),
            "slides": slides[train_idx].tolist()
        },
        "val": {
            "images": all_images[val_idx].tolist(),
            "labels": all_labels[val_idx].tolist(),
            "slides": slides[val_idx].tolist()
        },
        "test": {
            "images": all_images[test_idx].tolist(),
            "labels": all_labels[test_idx].tolist(),
            "slides": slides[test_idx].tolist()
        }
    }

    # --------------------------------------------------
    # 4. Estadísticas
    # --------------------------------------------------
    stats = {}
    for split_name, split_data in splits.items():
        stats[split_name] = {
            "num_images": len(split_data["images"]),
            "num_patients": len(set(split_data["slides"])),
            "num_benign": split_data["labels"].count(0),
            "num_malignant": split_data["labels"].count(1),
            "ratio_malig_benign": (
                split_data["labels"].count(1) /
                max(split_data["labels"].count(0), 1)
            )
        }

    # --------------------------------------------------
    # 5. Sanity check (CRÍTICO)
    # --------------------------------------------------
    train_patients = set(splits["train"]["slides"])
    val_patients = set(splits["val"]["slides"])
    test_patients = set(splits["test"]["slides"])

    leakage_check = {
        "train_val": list(train_patients & val_patients),
        "train_test": list(train_patients & test_patients),
        "val_test": list(val_patients & test_patients)
    }

    assert not any(leakage_check.values()), \
        f"❌ Data leakage detectado: {leakage_check}"

    return splits, stats

def plot_split_distributions(stats, output_dir):
    """
    Plotea la distribución benigno/maligno por split
    """

    splits = list(stats.keys())
    benign_counts = [stats[s]['num_benign'] for s in splits]
    malignant_counts = [stats[s]['num_malignant'] for s in splits]

    x = range(len(splits))
    width = 0.35

    # -------------------------------
    # 1. Barras apiladas
    # -------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(x, benign_counts, width, label='Benign', color='tab:blue')
    ax.bar(x, malignant_counts, width, bottom=benign_counts,
           label='Malignant', color='tab:orange')

    ax.set_xlabel('Split')
    ax.set_ylabel('Número de imágenes')
    ax.set_title('Distribución de clases por split (imágenes)')
    ax.set_xticks(x)
    ax.set_xticklabels([s.upper() for s in splits])
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    for i, (b, m) in enumerate(zip(benign_counts, malignant_counts)):
        total = b + m
        ax.text(i, total + 50, f'{total}', ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'split_distribution_images.png'), dpi=150)
    plt.close()

    # -------------------------------
    # 2. Ratio maligno/benigno
    # -------------------------------
    ratios = [stats[s]['ratio_malig_benign'] for s in splits]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(splits, ratios, color='steelblue', edgecolor='black')

    ax.axhline(2.19, color='red', linestyle='--',
               label='Ratio global (2.19:1)')

    ax.set_title('Ratio maligno / benigno por split')
    ax.set_ylabel('Ratio')
    ax.set_xlabel('Split')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    for bar, r in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.1,
                f'{r:.2f}:1',
                ha='center',
                fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'split_ratio.png'), dpi=150)
    plt.close()

# ======================================================
# EJECUCIÓN PRINCIPAL
# ======================================================
if __name__ == "__main__":

    BASE_PATH = r"D:\TFG_BREAKHIST\BreakHist\data\BreakHis - Breast Cancer Histopathological Database\dataset_cancer_v1\dataset_cancer_v1\classificacao_binaria"
    OUTPUT_DIR = "splits"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("🔍 Cargando dataset...")
    _, all_images, all_labels, label_map, slides = read_binary_breakhis_data(
        base_path=BASE_PATH,
        verbose=False
    )

    print("✂️  Realizando split por paciente (80/10/10)...")
    splits, stats = split_by_patient(
        all_images,
        all_labels,
        slides,
        train_size=0.8,
        val_size=0.1,
        test_size=0.1,
        random_state=42
    )

    # --------------------------------------------------
    # Mostrar estadísticas
    # --------------------------------------------------
    print("\n📊 ESTADÍSTICAS DEL SPLIT:")
    for split, s in stats.items():
        print(f"\n🔹 {split.upper()}")
        print(f"   • Imágenes: {s['num_images']}")
        print(f"   • Pacientes: {s['num_patients']}")
        print(f"   • Benignas: {s['num_benign']}")
        print(f"   • Malignas: {s['num_malignant']}")
        print(f"   • Ratio maligno/benigno: {s['ratio_malig_benign']:.2f}:1")

    # --------------------------------------------------
    # Guardar splits
    # --------------------------------------------------
    for split_name, split_data in splits.items():
        path = os.path.join(OUTPUT_DIR, f"{split_name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(split_data, f, indent=2)

    print("\n✅ Split completado correctamente")
    print(f"📁 Archivos guardados en: {OUTPUT_DIR}")

    print("\n📊 Generando visualizaciones del split...")
    plot_split_distributions(stats, OUTPUT_DIR)
    print("📈 Gráficas guardadas en:", OUTPUT_DIR)


    """
    Para VAL
    
    El split es por paciente

    Hay muy pocos pacientes benignos (24 en total)

    Con solo 8 pacientes en validación, basta con que:

    entren 1–2 pacientes benignos

    y varios malignos “grandes”

    → el ratio se dispara
    """