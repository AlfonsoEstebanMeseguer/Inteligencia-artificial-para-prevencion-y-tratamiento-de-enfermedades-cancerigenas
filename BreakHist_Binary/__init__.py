"""
Inicializa BreakHist_Binary como paquete de primer nivel y expone el código que vive en src/.
"""
import sys
import pathlib

_root = pathlib.Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
