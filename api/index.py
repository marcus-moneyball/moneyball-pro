import sys
import os

# Força o diretório 'api' a ser a prioridade máxima de busca do Python
dir_path = os.path.dirname(os.path.abspath(__file__))
if dir_path not in sys.path:
    sys.path.insert(0, dir_path)

from main import app
