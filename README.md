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