# TFG_BREAKHIST

Repositorio para experimentos de clasificación binaria (benign/malignant) en **BreakHis** con TensorFlow/Keras.

## Estructura del proyecto

- `src/config/`: lectura de dataset, split por paciente, EDA y configuración del pipeline.
  - `src/config/readDataset.py`: utilidades para leer BreakHis (binario).
  - `src/config/split_dataset.py`: split **por paciente** (evita data leakage) y guardado en JSON.
  - `src/config/config_dataset.py`: `DatasetConfig` + `BreakHisDataLoader` (tf.data, augmentations, normalización).
  - `src/config/eda.py`: análisis exploratorio.
- `src/train/`: scripts de entrenamiento (CNN desde cero y transfer learning).
- `src/data/`: directorio recomendado para artefactos generados (p.ej. `splits/`).
- `BreakHist/`: datos locales (si los tienes descargados aquí).

## Requisitos

- Python 3.9+ (recomendado 3.10).
- TensorFlow (y opcionalmente `tensorflow-addons` para `cnn_transfer_2.py`).
- Paquetes típicos: `numpy`, `scikit-learn`, `matplotlib`, `Pillow` (y otros si usas `eda.py`).

Si ya tienes tu entorno montado, puedes ignorar esta sección.

## Nota importante sobre imports

El código está modularizado bajo `src/` como paquete. Se recomienda ejecutar **desde la raíz del proyecto**:

- Como script: `python src/train/cnn_transfer_1.py ...`
- O como módulo (más "limpio"): `python -m src.train.cnn_transfer_1 ...`

Los scripts incluyen un pequeño “bootstrap” para que funcionen también si los ejecutas como archivo suelto.

## Dataset (BreakHis binario)

Los scripts esperan una ruta `--base-path` que apunte al dataset binario con una estructura similar a:

```
<base-path>/
  40X/
    benign/
    malignant/
  100X/
    benign/
    malignant/
  200X/
    benign/
    malignant/
  400X/
    benign/
    malignant/
```

En tu proyecto, la ruta por defecto suele ser algo como:
`D:\TFG_BREAKHIST\BreakHist\data\BreakHis - Breast Cancer Histopathological Database\dataset_cancer_v1\dataset_cancer_v1\classificacao_binaria`

## 1) Generar splits (recomendado)

Genera `train.json`, `val.json` y `test.json` con split **por paciente**:

```bash
python src/config/split_dataset.py ^
  --base-path "D:\TFG_BREAKHIST\BreakHist\data\BreakHis - Breast Cancer Histopathological Database\dataset_cancer_v1\dataset_cancer_v1\classificacao_binaria" ^
  --output-dir "src/data/splits" ^
  --train-size 0.8 --val-size 0.1 --test-size 0.1 ^
  --no-verbose
```

Esto también guarda gráficas de distribución en el mismo directorio de salida.

## 2) Entrenamiento (scripts)

Todos los scripts crean los splits automáticamente si no existen (pero es mejor generarlos explícitamente como arriba).

### CNN “light” (desde cero)

**CNN #1**
```bash
python src/train/cnn_light_1.py ^
  --base-path "D:\...\classificacao_binaria" ^
  --splits-dir "src/data/splits" ^
  --epochs 40 --batch-size 32 ^
  --img-size 224 224 ^
  --augmentation-level medium ^
  --normalization-mode imagenet
```

**CNN #2**
```bash
python src/train/cnn_light_2.py --base-path "D:\...\classificacao_binaria" --splits-dir "src/data/splits"
```

**CNN #3**
```bash
python src/train/cnn_light_3.py --base-path "D:\...\classificacao_binaria" --splits-dir "src/data/splits"
```

**CNN #4**
```bash
python src/train/cnn_light_4.py --base-path "D:\...\classificacao_binaria" --splits-dir "src/data/splits"
```

**CNN #5 (DenseNet-SE + focal loss)**
```bash
python src/train/cnn_light_5.py --base-path "D:\...\classificacao_binaria" --splits-dir "src/data/splits"
```

### Transfer learning

**Transfer #1 (ResNet50)**
```bash
python src/train/cnn_transfer_1.py ^
  --base-path "D:\...\classificacao_binaria" ^
  --splits-dir "src/data/splits" ^
  --epochs-head 15 --epochs-ft 40 ^
  --batch-size 32 ^
  --augmentation-level advanced ^
  --normalization-mode resnet
```

**Transfer #2 (EfficientNetB0 + AdamW de TFA)**

Este script actualmente lleva `BASE_PATH` y `SPLITS_DIR` hardcodeados dentro de `main()`.
Si quieres usar otra ruta, ajusta esas variables en `src/train/cnn_transfer_2.py`.

## 3) EDA

El EDA es pesado y tiene dependencias extra. Ejecútalo sólo si tienes el entorno preparado:

```bash
python src/config/eda.py
```

## Problemas comunes

- **`ModuleNotFoundError: No module named 'src'`**: ejecuta desde la raíz del repo (`D:\TFG_BREAKHIST`) o usa `python -m ...`.
- **Splits en carpeta distinta**: asegúrate de pasar `--splits-dir "src/data/splits"` (o usa `splits/` en la raíz, pero sé consistente).
- **`tensorflow_addons` no instalado**: afecta a `cnn_transfer_2.py`.
