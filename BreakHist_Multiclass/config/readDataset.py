import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

# Ruta por defecto apuntando a la carpeta multiclase
DEFAULT_BASE_PATH=os.environ.get("BREAKHIS_BASE_MULTICLASS",os.path.join(Path(__file__).resolve().parents[2],"BreakHist"
                                ,"data","BreakHis - Breast Cancer Histopathological Database"
                                ,"dataset_cancer_v1","dataset_cancer_v1","classificacao_multiclasse"))

# Reutilizamos la función binaria para evitar duplicar lógica
BINARY_CONFIG_DIR=Path(__file__).resolve().parents[2]/"BreakHist_Binary"/"src"/"config"
if str(BINARY_CONFIG_DIR) not in sys.path:
    sys.path.insert(0,str(BINARY_CONFIG_DIR))
try:
    from readDataset import read_binary_breakhis_data
except ImportError as e:
    raise ImportError(f"No se pudo importar read_binary_breakhis_data desde {BINARY_CONFIG_DIR}") from e

def parse_arguments():
    parser=argparse.ArgumentParser(description="Lectura del dataset BreakHis en clasificación multiclase.")
    parser.add_argument("--base-path",default=DEFAULT_BASE_PATH,help="Ruta raíz del dataset BreakHis (multiclase).")
    parser.add_argument("--verbose",action="store_true",default=False,help="Imprimir resumen.")
    return parser.parse_args()

def main():
    args=parse_arguments()
    data,all_images,all_labels,label_map,slides = read_binary_breakhis_data(argparse.base_path,argparse.verbose)
    if len(label_map)<=2:
        raise ValueError(f"Se esperaban >2 clases para multiclase en {argparse.base_path}, se encontraron {len(label_map)}: {list(label_map.keys())}")
    return data,all_images,all_labels,label_map,slides

if __name__=="__main__":
    main()
