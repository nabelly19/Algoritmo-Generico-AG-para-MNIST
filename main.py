import numpy as np
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score

# ==============================
# 1. Carregar e preparar o dataset
# ==============================
print("Carregando MNIST...")
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
X = X.astype(np.float32)
y = y.astype(int)

# Usar amostra reduzida (10.000 exemplos) para tornar o GA viável
X_train_full, y_train_full = X[:10000], y[:10000]
X_test, y_test = X[60000:], y[60000:]

# Dividir treino da busca (para o GA) e validação interna (para fitness)
X_train_search, X_val_search, y_train_search, y_val_search = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42
)

n_features = X_train_search.shape[1]

# ==============================
# 2. Hiperparâmetros do GA
# ==============================
POP_SIZE = 60
N_GEN = 30
STAGNANT_LIMIT = 8
P_INIT = 0.3
P_CROSS = 0.8
P_MUT = 1 / n_features
ELITISM = 2
ALPHA, BETA = 0.9, 0.1
RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)

# ==============================
# 3. Funções utilitárias
# ==============================
def init_population():
    population = []
    for _ in range(POP_SIZE):
        chrom = (np.random.rand(n_features) < P_INIT).astype(int)
        if chrom.sum() == 0:  # garantir que ao menos uma feature seja usada
            chrom[np.random.randint(0, n_features)] = 1
        population.append({'chrom': chrom, 'fitness': None})
    return population


def evaluate_individual(ind, cache):
    """Calcula o fitness de um indivíduo usando cache"""
    chrom = ind['chrom']
    key = tuple(chrom.tolist())
    if key in cache:
        return cache[key]
    
    selected = np.where(chrom == 1)[0]
    if len(selected) == 0:
        return 0.0  # indivíduo inválido
    
    # Treinar modelo Decision Tree
    clf = DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE)
    clf.fit(X_train_search[:, selected], y_train_search)
    y_pred = clf.predict(X_val_search[:, selected])
    
    acc = accuracy_score(y_val_search, y_pred)
    perc_features = len(selected) / n_features
    fitness = ALPHA * acc - BETA * perc_features
    
    cache[key] = fitness
    return fitness


def tournament_selection(pop, k=3):
    """Seleciona um indivíduo por torneio"""
    candidates = np.random.choice(pop, k)
    return max(candidates, key=lambda ind: ind['fitness'])


def crossover(parent1, parent2):
    """Crossover uniforme"""
    if np.random.rand() < P_CROSS:
        mask = np.random.rand(n_features) < 0.5
        child1 = np.where(mask, parent1['chrom'], parent2['chrom']).astype(int)
        child2 = np.where(mask, parent2['chrom'], parent1['chrom']).astype(int)
    else:
        child1, child2 = parent1['chrom'].copy(), parent2['chrom'].copy()
    return {'chrom': child1, 'fitness': None}, {'chrom': child2, 'fitness': None}


def mutate(ind):
    """Mutação bit-flip"""
    chrom = ind['chrom']
    for i in range(n_features):
        if np.random.rand() < P_MUT:
            chrom[i] = 1 - chrom[i]
    if chrom.sum() == 0:
        chrom[np.random.randint(0, n_features)] = 1
    ind['chrom'] = chrom
    ind['fitness'] = None


# ==============================
# 4. Execução do GA
# ==============================
print("\nIniciando busca com GA...")
start_time = time.perf_counter()
cache = {}
population = init_population()

# Avaliar população inicial
for ind in population:
    ind['fitness'] = evaluate_individual(ind, cache)

best = max(population, key=lambda i: i['fitness'])
best_fitness = best['fitness']
no_improve = 0

history = [(0, best_fitness, best['chrom'].sum())]

for gen in range(1, N_GEN + 1):
    new_pop = sorted(population, key=lambda x: x['fitness'], reverse=True)[:ELITISM]
    
    while len(new_pop) < POP_SIZE:
        p1 = tournament_selection(population)
        p2 = tournament_selection(population)
        c1, c2 = crossover(p1, p2)
        mutate(c1)
        mutate(c2)
        new_pop.extend([c1, c2])
    
    population = new_pop[:POP_SIZE]
    
    for ind in population:
        if ind['fitness'] is None:
            ind['fitness'] = evaluate_individual(ind, cache)
    
    current_best = max(population, key=lambda i: i['fitness'])
    
    if current_best['fitness'] > best_fitness:
        best = current_best
        best_fitness = current_best['fitness']
        no_improve = 0
    else:
        no_improve += 1
    
    history.append((gen, best_fitness, best['chrom'].sum()))
    
    print(f"Geração {gen}: Melhor fitness = {best_fitness:.4f} | Features = {best['chrom'].sum()}")
    
    if no_improve >= STAGNANT_LIMIT:
        print("Critério de parada por estagnação atingido.")
        break

search_time = time.perf_counter() - start_time

# ==============================
# 5. Avaliação final no conjunto de teste
# ==============================
selected_features = np.where(best['chrom'] == 1)[0]
print(f"\nNúmero de features selecionadas: {len(selected_features)} ({len(selected_features)/n_features*100:.2f}%)")

train_start = time.perf_counter()
final_model = DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE)
final_model.fit(X_train_full[:, selected_features], y_train_full)
train_time = time.perf_counter() - train_start

y_pred_test = final_model.predict(X_test[:, selected_features])
acc_test = accuracy_score(y_test, y_pred_test)

# ==============================
# 6. Resultados finais
# ==============================
print("\n===== Resultados GA =====")
print(f"Acurácia no teste: {acc_test:.4f}")
print(f"Porcentagem de features selecionadas: {len(selected_features)/n_features*100:.2f}%")
print(f"Tempo de busca de features: {search_time:.2f} s")
print(f"Tempo de treinamento final: {train_time:.2f} s")
