# 🤖 Modelo Preditivo para Otimização Agrícola

Este projeto de *Machine Learning (ML)* foi desenvolvido para a startup fictícia FarmTech Solutions. Seu objetivo é diagnosticar as condições de solo e clima de uma área e prever qual é a cultura agrícola ideal a ser plantada, focando na *otimização de rendimento* e tomada de decisão baseada em dados. O modelo final alcança mais de *99% de acurácia* na classificação.

***

## 🛠 1. Foco Técnico: Bibliotecas e Ferramentas

O projeto utiliza o ecossistema Python, destacando a proficiência nas seguintes ferramentas e bibliotecas:

### A. Análise e Visualização (EDA)

| Biblioteca | Uso Principal |
| :--- | :--- |
| **pandas** & **numpy** | Manipulação de dados (DataFrames). Usado para carregar o CSV, calcular estatísticas e agrupar dados por cultura. |
| **matplotlib** & **seaborn** | Criação dos 5 gráficos obrigatórios, incluindo o *Boxplot* (para Perfil Ideal) e o *Heatmap* (para Correlação entre N, P, K). |

### B. Machine Learning (Modelagem Preditiva)

| Ferramenta | Uso Principal |
| :--- | :--- |
| **scikit-learn** | Framework central de ML para todos os algoritmos. |
| **StandardScaler** | *Pré-processamento crítico* que padroniza as features (N, P, K, etc.), garantindo a alta performance de modelos sensíveis à escala (SVM, Regressão Logística). |
| **train_test_split** | Divisão dos dados em treino (80%) e teste (20%) com a função stratify para garantir o balanceamento. |

***

## 🏆 2. Resultados Finais e Performance do Modelo

O projeto comparou a performance de 5 algoritmos de classificação para identificar o mais eficaz na recomendação de culturas.

| Modelo | Acurácia (Exemplo) | Análise |
| :--- | :--- | :--- |
| *Random Forest* | *~0.9954 (99.54%)* | *Vencedor:* Algoritmo de classificação robusto, ideal para lidar com a natureza não-linear dos dados de solo e clima. |
| *SVM (SVC)* | ~0.9041 | Melhorou drasticamente a performance após o uso do StandardScaler. |
| *Regressão Logística* | ~0.9727 | Demonstrou que o problema possui um alto grau de separabilidade linear. |

### Conclusão

O algoritmo *Random Forest* foi selecionado como o modelo de produção, pois sua alta acurácia garante que a recomendação da cultura ideal será feita com *confiança máxima*, otimizando a decisão do agricultor.

***

## 📁 3. Estrutura do Repositório

* **analise.ipynb**: Contém todo o relatório (EDA, gráficos, treinamento e conclusões).
* **produtos_agricolas.csv**: Base de dados.
* **src/** (Futuro): Será o módulo de código de produção para carregar o modelo treinado.
* **notebooks/** (Futuro): Pasta opcional para rascunhos de análise.


## ➡ 4. Próximos Passos e Funcionalidades Futuras

O objetivo inicial de modelagem e prova de conceito foi concluído. O próximo estágio do projeto focará em transformar o modelo treinado em uma *ferramenta funcional* para uso prático.

### Funcionalidades a Serem Desenvolvidas:

1.  *Persistência do Modelo:*
    * O modelo vencedor (Random Forest) será serializado (salvo) usando **joblib** ou **pickle** para evitar o re-treinamento desnecessário a cada uso.
    * O StandardScaler também será salvo, garantindo que os novos dados de entrada sejam pré-processados corretamente.

2.  **Modularização e Produção (src/):**
    * O código de treinamento e predição será refatorado e movido para módulos Python (.py) dentro da pasta src/, seguindo padrões de engenharia de software.
    * Será criada uma função principal (recomendar_cultura) para receber as 7 variáveis de solo e clima e retornar a predição.

3.  *Desenvolvimento de Interface:*
    * Criação de uma aplicação que ainda sera pensada