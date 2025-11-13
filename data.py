import os
import numpy as np
from sklearn.model_selection import train_test_split

# Configurações e constantes
RANDOM_STATE = 42        # RNG do GA e operadores
TREE_RANDOM_STATE = 1    # Semente randômica do Decision Tree
MAX_TRAIN_ROWS = 10000   # Amostra para tornar a execução mais rápida

# Carregamento dos CSVs
def load_mnist_csv(path, max_rows=None):
    data = None
    try:
        import pandas as pd
        df = pd.read_csv(path)
        data = df.values
    except Exception:
        try:
            data = np.loadtxt(path, delimiter=',', skiprows=1)
        except Exception:
            data = np.genfromtxt(path, delimiter=',', skip_header=1)
    if max_rows is not None:
        data = data[:max_rows]
    X = data[:, 1:].astype(np.float32)
    y = data[:, 0].astype(int)
    return X, y
 
dataset_dir = os.path.join(os.path.dirname(__file__), 'dataset')
train_csv = os.path.join(dataset_dir, 'mnist_train.csv')
test_csv = os.path.join(dataset_dir, 'mnist_test.csv')

# Carregar dados
#X_train_full, y_train_full = load_mnist_csv(train_csv, max_rows=MAX_TRAIN_ROWS)
X_train_full, y_train_full = load_mnist_csv(train_csv)
X_test, y_test = load_mnist_csv(test_csv)

# Split para busca/validação interna
X_train_search, X_val_search, y_train_search, y_val_search = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=RANDOM_STATE, stratify=y_train_full
)

n_features = X_train_search.shape[1]