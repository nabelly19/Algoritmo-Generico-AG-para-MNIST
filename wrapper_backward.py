import time
import numpy as np
import logging
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from data import X_train_search, y_train_search, X_val_search, y_val_search, X_train_full, y_train_full, X_test, y_test, n_features, TREE_RANDOM_STATE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def backward_selection(X_tr, y_tr, X_val, y_val, random_state=1):
    features = list(range(X_tr.shape[1]))
    clf = DecisionTreeClassifier(random_state=random_state)
    clf.fit(X_tr[:, features], y_tr)
    best_acc = accuracy_score(X_val, clf.predict(X_val[:, features])) if False else accuracy_score(y_val, clf.predict(X_val[:, features]))
    start = time.perf_counter()
    changed = True
    while changed and len(features) > 1:
        changed = False
        best_removal = None
        best_removal_acc = best_acc
        for f in features:
            cand = [x for x in features if x != f]
            c = DecisionTreeClassifier(random_state=random_state)
            c.fit(X_tr[:, cand], y_tr)
            acc = accuracy_score(y_val, c.predict(X_val[:, cand]))
            if acc >= best_removal_acc:
                best_removal_acc = acc
                best_removal = f
        if best_removal is not None and best_removal_acc >= best_acc:
            features.remove(best_removal)
            best_acc = best_removal_acc
            changed = True
            logging.info(f"Removida feature {best_removal}, acc val = {best_acc:.4f}, features={len(features)}")
    search_time = time.perf_counter() - start
    return np.array(features, dtype=int), best_acc, search_time

def run_wrapper_backward():
    logging.info("Iniciando Wrapper Backward...")
    selected, val_acc, search_time = backward_selection(X_train_search, y_train_search, X_val_search, y_val_search, random_state=TREE_RANDOM_STATE)
    logging.info(f"Backward: selecionadas {len(selected)} features em {search_time:.2f}s; acc val = {val_acc:.4f}")

    t0 = time.perf_counter()
    final = DecisionTreeClassifier(random_state=TREE_RANDOM_STATE)
    final.fit(X_train_full[:, selected], y_train_full)
    train_time = time.perf_counter() - t0
    acc_test = accuracy_score(y_test, final.predict(X_test[:, selected]))
    return {
        'method': 'Wrapper-Backward',
        'acc_test': float(acc_test),
        'perc_features': float(len(selected) / n_features),
        'search_time': float(search_time),
        'train_time': float(train_time),
        'n_features_selected': int(len(selected))
    }

if __name__ == "__main__":
    selected, val_acc, search_time = backward_selection(X_train_search, y_train_search, X_val_search, y_val_search, random_state=TREE_RANDOM_STATE)
    logging.info(f"Backward: selecionadas {len(selected)} features em {search_time:.2f}s; acc val = {val_acc:.4f}")

    t0 = time.perf_counter()
    final = DecisionTreeClassifier(random_state=TREE_RANDOM_STATE)
    final.fit(X_train_full[:, selected], y_train_full)
    train_time = time.perf_counter() - t0
    acc_test = accuracy_score(y_test, final.predict(X_test[:, selected]))
    print({
        'method': 'Wrapper-Backward',
        'acc_test': float(acc_test),
        'perc_features': float(len(selected) / n_features),
        'search_time': float(search_time),
        'train_time': float(train_time),
        'n_features_selected': int(len(selected))
    })