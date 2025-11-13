import time
import logging
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from data import X_train_full, y_train_full, X_test, y_test, n_features, TREE_RANDOM_STATE
 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_baseline():
    logging.info("Iniciando Baseline...")
    t0 = time.perf_counter()
    search_time = time.perf_counter() - t0
    clf = DecisionTreeClassifier(random_state=TREE_RANDOM_STATE)
    clf.fit(X_train_full, y_train_full)
    train_time = time.perf_counter() - t0
    acc_test = accuracy_score(y_test, clf.predict(X_test))
   
    print(f"""Algoritmo: Baseline
        Acurácia (Dados Teste): {float(acc_test) * 100:.2f}%
        Porcentagem de features: 1.0
        Tempo para busca das features: {search_time:.4f} segundos
        Tempo de treinamento: {float(train_time):.4f} segundos
        Quantidade de Features Selecionadas: {int(n_features)}"""
    )