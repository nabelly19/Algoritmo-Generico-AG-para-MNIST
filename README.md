## 🛠️ Decisões de Projeto do Algoritmo Genético (GA) para Seleção de Features

Este projeto utiliza um Algoritmo Genético (GA) para otimizar a seleção de *features* (pixels) do *dataset* MNIST ($784$ *features* originais) com o objetivo de treinar um classificador **Decision Tree**. As decisões de projeto do GA buscam balancear a performance do modelo (acurácia) com a complexidade (número de *features* selecionadas).

---

### 1. Representação (Cromossomo)

* **Decisão:** **Vetor binário de 784 posições**. Cada posição (gene) é uma *feature*: $1$ (selecionada) ou $0$ (excluída).
* **Justificativa:** Codificação canônica para o problema de seleção de *features* (incluir/excluir), permitindo a exploração de subconjuntos de **dimensões variáveis** de forma flexível.

---

### 2. População Inicial

* **Decisão:** Tamanho de população moderado ($P \approx 50$ a $100$ indivíduos). Inicialização **aleatória** com probabilidade de inclusão de *feature* $p_{init} \approx 0.3$.
* **Justificativa:** Equilíbrio entre **diversidade** e **tempo de avaliação**. O valor $p_{init} < 0.5$ favorece subconjuntos menores desde o início.

---

### 3. Função de Fitness

* **Decisão:** Função de objetivo único que combina **acurácia** e **penalização de complexidade** (número de *features*).

    $$
    \text{fitness}(S) = \alpha \times \text{Accuracy}_{val}(S) - \beta \times \frac{\#\text{features}(S)}{784}
    $$

    (Ex: $\alpha=0.9, \beta=0.1$)
* **Justificativa:** Essencial para modelar o **trade-off** entre **desempenho** (max. acurácia) e **complexidade** (min. *features*), evitando a seleção de todas as *features*. A acurácia ($\text{Accuracy}_{val}(S)$) é calculada em um conjunto de validação interno.

---

### 4. Operadores Genéticos

| Componente | Decisão Proposta | Parâmetros Sugeridos | Justificativa |
| :--- | :--- | :--- | :--- |
| **Seleção de Pais** | **Seleção por Torneio** | Tamanho do Torneio $k=3$ ou $k=5$. **Elitismo** com $e=2$ melhores indivíduos. | Garante estabilidade e progresso contínuo, crucial devido ao custo de avaliação por indivíduo. |
| **Crossover** | **Crossover Uniforme** | Taxa de Crossover $p_c \approx 0.8$. | Promove a **mistura completa** dos genes, explorando melhor as combinações de *features* no espaço de alta dimensão. |
| **Mutação** | Inversão de Bit | Taxa de Mutação $p_m = \frac{1}{784} \approx 0.0013$. | Mantém a **diversidade** e evita a convergência prematura. A taxa $1/m$ garante, em média, uma mutação por cromossomo por geração. |

---

### 5. Critério de Parada

* **Decisão:** Parada quando:
    1.  O **número máximo de gerações** for alcançado ($G_{\text{max}} \approx 30$ a $50$).
    2.  Ou **não houver melhora** no melhor indivíduo após $g_{\text{stagnant}}$ gerações (Ex: $g_{\text{stagnant}} = 10$).
* **Justificativa:** O critério de estagnação ($g_{\text{stagnant}}$) otimiza o tempo de execução, interrompendo a busca quando o algoritmo atinge uma convergência.

---

### 6. Considerações Práticas e Logística Experimental

* **Amostragem:** Será utilizada uma **amostra reduzida** do treinamento (Ex: $10.000$ exemplos) na fase de busca (cálculo do *fitness*) para garantir a **viabilidade computacional**.
* **Consistência:** Todas as abordagens (GA, *Wrapper* Sequencial, *Baseline*) usarão o mesmo modelo (**Decision Tree**) e o mesmo **Conjunto de Teste** ($10.000$ exemplos) para avaliação final.
* **Integridade:** O conjunto de teste **não** será usado em nenhuma parte da busca de *features* (cálculo do *fitness*).
* 
### 7. Pseudocódigo:

``` csharp

Input: 
    X_train_full (n_train × 784) – conjunto de treinamento completo usado para busca
    y_train_full (n_train) – rótulos correspondentes
    X_test (n_test × 784), y_test (n_test) – conjunto de teste (não usado na busca)
Parameters:
    P = tamanho da população (ex: 80)
    G_max = número máximo de gerações (ex: 40)
    g_stagnant = número máximo de gerações sem melhoria (ex: 10)
    p_init = probabilidade inicial de seleção de cada feature (ex: 0.3)
    p_crossover = taxa de crossover (ex: 0.8)
    p_mutation = taxa de mutação por gene (ex: 1/784 ≈ 0.0013)
    elitism_size = número de melhores indivíduos que passam direto (ex: 2)
    α, β = pesos da função de fitness (ex: α=0.9, β=0.1)

Procedure:
1. Dividir X_train_full, y_train_full em:
       – X_train_search, y_train_search (por exemplo 80% dos dados) – para treinar modelo durante GA
       – X_val_search, y_val_search (por exemplo 20%) – para validar cada indivíduo e calcular acurácia

2. Inicializar população Pop = []  
   For i in 1 … P:
       individual.chromosome = vetor de comprimento 784 com cada gene = 1 com probabilidade p_init, ou 0 com probabilidade (1-p_init)
       Pop.append(individual)

3. Avaliar fitness de cada indivíduo em Pop:
   For cada individual ind in Pop:
       seletor = indices onde ind.chromosome == 1  
       Treinar modelo DecisionTreeClassifier() em X_train_search[:, seletor], y_train_search  
       Fazer predição em X_val_search[:, seletor] → acurácia = acc  
       pct = (#features_selected) / 784  
       fitness = α * acc - β * pct  
       ind.fitness = fitness  
   Registrar melhor_indivíduo = indivíduo com maior fitness  
   melhor_fitness_historico = melhor_indivíduo.fitness  
   generacao = 0  
   geracoes_sem_melhoria = 0

4. Enquanto (generacao < G_max) e (geracoes_sem_melhoria < g_stagnant):
       generacao += 1
       
       ## 4.1 – Seleção:
       NovaPop = []
       Copiar os elitism_size melhores indivíduos de Pop para NovaPop (elitismo)
       
       Enquanto (|NovaPop| < P):
           Selecionar “pai1” mediante torneio de tamanho k (ex: 3) da Pop  
           Selecionar “pai2” da mesma forma  
           
           ## 4.2 – Crossover:
           Gerar dois filhos filho1, filho2:
               Com probabilidade p_crossover:  
                   Para cada gene j = 1 … 784:
                       se rand() < 0.5 então filho1.gene[j] = pai1.gene[j], filho2.gene[j] = pai2.gene[j]
                       else filho1.gene[j] = pai2.gene[j], filho2.gene[j] = pai1.gene[j]
               Caso contrário (sem crossover) filho1 = pai1.copy(), filho2 = pai2.copy()
           
           ## 4.3 – Mutação:
           Para cada filho in {filho1, filho2}:
               Para each gene j = 1 … 784:
                   if rand() < p_mutation then filho.gene[j] = 1 - filho.gene[j]
               Garantir que pelo menos 1 gene = 1 (opcional: se todos zeros, setar um gene aleatório para 1)
           
           Adicionar filho1, filho2 em NovaPop (até preencher P)
       
       Pop = NovaPop  (nova geração)
       
       ## 4.4 – Avaliar fitness da nova população:
       Para cada indivíduo ind in Pop que ainda não tenha fitness calculado:
           seletor = indices onde ind.chromosome == 1  
           Treinar DecisionTreeClassifier em X_train_search[:, seletor], y_train_search  
           Validar em X_val_search[:, seletor] → acc  
           pct = (#features_selected) / 784  
           fitness = α * acc - β * pct  
           ind.fitness = fitness  
       
       Encontrar melhor indivíduo da geração: if melhor_indivíduo_gen.fitness > melhor_fitness_historico:
           melhor_indivíduo = melhor_indivíduo_gen  
           melhor_fitness_historico = melhor_indivíduo.fitness  
           geracoes_sem_melhoria = 0
       else:
           geracoes_sem_melhoria += 1

5. Ao fim do loop: melhor_indivíduo.chromosome define o **subconjunto de features final** selecionadas pelo GA.

6. **Treinamento final do modelo**:  
   seletor_final = indices onde melhor_indivíduo.chromosome == 1  
   Treinar DecisionTreeClassifier em *todo* X_train_full[:, seletor_final] e y_train_full  
   Fazer previsão em X_test[:, seletor_final], y_test → obter acurácia_final

7. **Relatório**:  
   – Acurácia no teste = acurácia_final  
   – Porcentagem de features selecionadas = (#seletor_final) / 784 × 100%  
   – Tempo de busca = tempo gasto na fase 1-5  
   – Tempo de treinamento = tempo gasto na fase 6  

Output: melhor_indivíduo, acurácia_final, porcentagem_features, tempos.

End.
