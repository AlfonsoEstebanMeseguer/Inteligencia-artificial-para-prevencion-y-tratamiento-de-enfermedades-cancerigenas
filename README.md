[Generado por IA]
[Faltan Bases de datos, habría que colocarlas en su carpeta correspondiente]

# TFG_BREAKHIST 
 
Repositorio principal del trabajo de TFG para clasificacion histopatologica con TensorFlow y Keras. 
Incluye tres lineas de trabajo: BreakHis binario, BreakHis multiclase y BRACS multiclase. 
 
## 1. Estructura general 
 
- BreakHist_Binary/: pipeline completo para clasificacion binaria. 
- BreakHist_Multiclass/: pipeline completo para clasificacion multiclase. 
- BRACS/: pipeline multiclase para dataset BRACS. 
- tf_gpu_210/: entorno virtual local Python 3.10 + TensorFlow 2.10. 
- models/: salida de modelos cuando se ejecuta desde la raiz (segun script). 
 
## 2. Requisitos 
 
Librerias usadas en el proyecto: tensorflow, numpy, matplotlib, scikit-learn, pandas, seaborn y pillow. 
 
Activacion de entorno recomendada: 
```bat 
D:\TFG_BREAKHIST\tf_gpu_210\Scripts\activate 
python --version 
``` 
 
## 3. Rutas esperadas de dataset 
 
- BreakHis binario (default): `BreakHist_Binary\BreakHist\data\BreakHis - Breast Cancer Histopathological Database\dataset_cancer_v1\dataset_cancer_v1\classificacao_binaria` 
- BreakHis multiclase (default): `BreakHist_Multiclass\BreakHist\data\BreakHis - Breast Cancer Histopathological Database\dataset_cancer_v1\dataset_cancer_v1\classificacao_multiclasse` 
- BRACS (default entrenamiento): `BRACS\BRACS\BRACS_RoI\latest_version_resized` 
 
Variables de entorno soportadas: `BREAKHIS_BASE` y `BREAKHIS_BASE_MULTICLASS`. 
 
## 4. Flujo comun 
 
1. Lectura y validacion del dataset. 
2. Generacion/carga de splits train/val/test. 
3. Construccion de tf.data con preprocess y augmentacion. 
4. Entrenamiento del modelo. 
5. Evaluacion y guardado de artefactos. 
 
Recomendacion: usar `split_mode=patient` para evitar fuga de datos por paciente.
 
## 5. BreakHist_Binary 
 
Estructura principal: 
- src\config: lectura, split, dataset, augmentaciones y normalizacion. 
- src\models: definicion de arquitecturas. 
- src\train: scripts de entrenamiento. 
- src\utils\utils.py: evaluacion, metricas y Grad-CAM. 
- splits\split_patient y splits\split_imagen: JSON de splits. 
 
Scripts disponibles: 
- src\train\cnn_light_1.py 
- src\train\cnn_light_2.py 
- src\train\cnn_transfer_2.py 
 
Ejecucion recomendada (desde BreakHist_Binary): 
```bat 
cd /d D:\TFG_BREAKHIST\BreakHist_Binary 
python src\train\cnn_light_1.py --help 
python src\train\cnn_light_2.py --split-mode patient 
python src\train\cnn_transfer_2.py --split-mode patient 
``` 
 
Artefactos generados de ejemplo: 
- BreakHist_Binary\models\cnn_transfer_2\cnn7_effnetb0_transfer_final.weights.h5 
- BreakHist_Binary\splits\gradcam_fixed.json
 
## 6. BreakHist_Multiclass 
 
Estructura principal: 
- config\readDataset.py y config\create_dataset.py 
- models\models_definitions.py 
- utils\utils.py 
- train\cnn_light_1.py, train\cnn_light_2.py, train\cnn_transfer_1.py 
- splits\split_patient y splits\split_imagen 
 
Ejecucion recomendada (desde la raiz del repo): 
```bat 
cd /d D:\TFG_BREAKHIST 
python BreakHist_Multiclass\train\cnn_light_1.py --help 
python BreakHist_Multiclass\train\cnn_light_2.py --split-mode patient 
python BreakHist_Multiclass\train\cnn_transfer_1.py --split-mode patient 
``` 
 
Nota: varios scripts guardan resultados en models por nombre de script, relativo al directorio de ejecucion. 
 
## 7. BRACS 
 
Archivos clave: 
- config_bracs.py: lectura, split por paciente, tf.data, metricas y graficas. 
- eda_bracs.py: analisis exploratorio de clases, pacientes y calidad de imagen. 
- bracs_models.py: arquitectura de red para BRACS. 
- train\bracs_cnn_transfer_1.py: entrenamiento de transfer learning end-to-end. 
- splits\split_train.json, split_val.json y split_test.json. 
 
Ejecucion recomendada: 
```bat 
cd /d D:\TFG_BREAKHIST 
python BRACS\eda_bracs.py --help 
python BRACS\train\bracs_cnn_transfer_1.py --help 
python BRACS\train\bracs_cnn_transfer_1.py --overwrite-splits 
``` 
 
Salidas tipicas en BRACS: 
- BRACS\models\bracs_cnn_transfer_1\best_head.h5 
- BRACS\models\bracs_cnn_transfer_1\best_finetune.h5 
- BRACS\models\bracs_cnn_transfer_1\final.h5 
- BRACS\models\bracs_cnn_transfer_1\report_test.json 
 
## 8. Comandos rapidos 
 
```bat 
REM 1) Activar entorno 
D:\TFG_BREAKHIST\tf_gpu_210\Scripts\activate 
 
REM 2) Entrenar binario (ejemplo) 
cd /d D:\TFG_BREAKHIST\BreakHist_Binary 
python src\train\cnn_transfer_2.py --split-mode patient 
 
REM 3) Entrenar multiclase BreakHis (ejemplo) 
cd /d D:\TFG_BREAKHIST 
python BreakHist_Multiclass\train\cnn_transfer_1.py --split-mode patient 
 
REM 4) Entrenar BRACS (ejemplo) 
python BRACS\train\bracs_cnn_transfer_1.py --overwrite-splits 
``` 
 
## 9. Notas practicas 
 
- Ejecuta siempre --help para revisar argumentos antes de correr entrenamientos largos. 
- Para resultados comparables, fija seed y usa split_mode patient. 
- Si cambias ruta de dataset, usa --base-path o variables de entorno. 
- Si los splits no corresponden al dataset actual, regenera JSON de splits. 
 
Documento creado para dejar trazabilidad tecnica global del trabajo en D:\TFG_BREAKHIST.
