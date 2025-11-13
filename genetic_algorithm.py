import numpy as np
import time
import logging
from joblib import Parallel, delayed
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from data import X_train_search, y_train_search, X_val_search, y_val_search, X_train_full, y_train_full, X_test, y_test, n_features, RANDOM_STATE, TREE_RANDOM_STATE
 
# logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Hiperparâmetros do GA
POP_SIZE = 60
N_GEN = 30
STAGNANT_LIMIT = 8
P_INIT = 0.3
P_CROSS = 0.8
P_MUT = 1 / n_features
ELITISM = 2
ALPHA, BETA = 0.9, 0.1

# Paralelismo
N_JOBS = -1

np.random.seed(RANDOM_STATE)

def chrom_key(chrom):
    return np.packbits(chrom.astype(np.uint8)).tobytes()

def eval_chrom(chrom):
    key = chrom_key(chrom)
    selected = np.where(chrom == 1)[0]
    if len(selected) == 0:
        return key, 0.0
    clf = DecisionTreeClassifier(random_state=TREE_RANDOM_STATE)
    clf.fit(X_train_search[:, selected], y_train_search)
    y_pred = clf.predict(X_val_search[:, selected])
    acc = accuracy_score(y_val_search, y_pred)
    perc_features = len(selected) / n_features
    fitness = ALPHA * acc - BETA * perc_features
    return key, fitness

def init_population():
    population = []
    for _ in range(POP_SIZE):
        chrom = (np.random.rand(n_features) < P_INIT).astype(np.uint8)
        if chrom.sum() == 0:
            chrom[np.random.randint(0, n_features)] = 1
        population.append({'chrom': chrom, 'fitness': None})
    return population

def tournament_selection(pop, k=3):
    idxs = np.random.choice(len(pop), k, replace=False)
    best_idx = max(idxs, key=lambda i: pop[i]['fitness'])
    return pop[best_idx]

def crossover(parent1, parent2):
    if np.random.rand() < P_CROSS:
        mask = np.random.rand(n_features) < 0.5
        child1 = np.where(mask, parent1['chrom'], parent2['chrom']).astype(np.uint8)
        child2 = np.where(mask, parent2['chrom'], parent1['chrom']).astype(np.uint8)
    else:
        child1, child2 = parent1['chrom'].copy(), parent2['chrom'].copy()
    return {'chrom': child1, 'fitness': None}, {'chrom': child2, 'fitness': None}

def mutate(ind):
    chrom = ind['chrom']
    flip_mask = (np.random.rand(n_features) < P_MUT)
    if flip_mask.any():
        chrom = (chrom ^ flip_mask.astype(np.uint8))
    if chrom.sum() == 0:
        chrom[np.random.randint(0, n_features)] = 1
    ind['chrom'] = chrom
    ind['fitness'] = None

def evaluate_population_candidates(pop, cache):
    to_eval = {}
    for ind in pop:
        if ind['fitness'] is None:
            k = chrom_key(ind['chrom'])
            if k not in cache and k not in to_eval:
                to_eval[k] = ind['chrom'].copy()
    if to_eval:
        chroms = list(to_eval.values())
        results = Parallel(n_jobs=N_JOBS)(delayed(eval_chrom)(chrom) for chrom in chroms)
        for key, fitness in results:
            cache[key] = fitness
    for ind in pop:
        if ind['fitness'] is None:
            ind['fitness'] = cache.get(chrom_key(ind['chrom']), 0.0)

def run_ga():
    logging.info("Iniciando GA...")
    start_time = time.perf_counter()
    cache = {}
    population = init_population()
    evaluate_population_candidates(population, cache)

    best = max(population, key=lambda i: i['fitness'])
    best_fitness = best['fitness']
    no_improve = 0

    for gen in range(1, N_GEN + 1):
        new_pop = sorted(population, key=lambda x: x['fitness'], reverse=True)[:ELITISM]
        while len(new_pop) < POP_SIZE:
            p1 = tournament_selection(population)
            p2 = tournament_selection(population)
            c1, c2 = crossover(p1, p2)
            mutate(c1); mutate(c2)
            new_pop.extend([c1, c2])
        population = new_pop[:POP_SIZE]
        evaluate_population_candidates(population, cache)
        current_best = max(population, key=lambda i: i['fitness'])
        if current_best['fitness'] > best_fitness:
            best = current_best
            best_fitness = current_best['fitness']
            no_improve = 0
        else:
            no_improve += 1
        logging.info(f"Geração {gen}: Melhor fitness = {best_fitness:.4f} | Features = {best['chrom'].sum()}")
        if no_improve >= STAGNANT_LIMIT:
            logging.warning("Parada por estagnação.")
            break

    search_time = time.perf_counter() - start_time
    selected_features = np.where(best['chrom'] == 1)[0]
    train_start = time.perf_counter()
    final_model = DecisionTreeClassifier(random_state=TREE_RANDOM_STATE)
    final_model.fit(X_train_full[:, selected_features], y_train_full)
    train_time = time.perf_counter() - train_start
    y_pred_test = final_model.predict(X_test[:, selected_features])
    acc_test = accuracy_score(y_test, y_pred_test)

    print(f"""Algoritmo: Genético
        Acurácia (Dados Teste): {float(acc_test)},
        Porcentagem de features: {float(len(selected_features) / n_features)},
        Tempo para busca das features: {float(search_time)},
        Tempo de treinamento: {float(train_time)},
        Quantidade de Features Selecionadas: {int(len(selected_features))},
        best_chrom: {best['chrom'].copy()}
    """)