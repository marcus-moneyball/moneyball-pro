import os
import sys

# Pega o caminho absoluto da pasta api e da raiz do projeto
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Adiciona ambas ao sys.path do Python
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Importa o app FastAPI/Flask
try:
    from main import app
except ModuleNotFoundError:
    from api.main import app
