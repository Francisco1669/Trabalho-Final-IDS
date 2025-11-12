# Sistema de Detecção de Intrusões (IDS) com XAI

**Trabalho Final - Engenharia de Sistemas de Detecção de Intrusões**
**Professor:** Silvio Quincozes
**Dataset:** NSL-KDD
**Modelo:** Random Forest
**Técnicas XAI:** SHAP e LIME

---

## 📋 Descrição do Projeto

Este projeto implementa um Sistema de Detecção de Intrusões (IDS) baseado em Machine Learning com foco em **Inteligência Artificial Explicável (XAI)**. O objetivo é não apenas detectar ataques de rede, mas também **explicar por que** o modelo toma suas decisões.

### Objetivo Principal

Comparar duas técnicas de explicabilidade:
- **SHAP** (SHapley Additive exPlanations) - explicabilidade global e local
- **LIME** (Local Interpretable Model-agnostic Explanations) - explicabilidade local

---

## 🎯 Funcionalidades Implementadas

### ✅ Machine Learning
- [x] Pré-processamento completo do NSL-KDD
- [x] Encoding de variáveis categóricas
- [x] Treinamento de Random Forest (100 estimadores)
- [x] Classificação binária: Normal (0) vs Ataque (1)

### ✅ Avaliação do Modelo
- [x] Métricas completas (Accuracy, Precision, Recall, F1-Score, ROC-AUC)
- [x] Relatório de classificação detalhado
- [x] Matriz de confusão
- [x] Curva ROC
- [x] Feature importance nativa do Random Forest

### ✅ Explicabilidade com SHAP
- [x] Summary plot (importância global)
- [x] Bar plot (importância média)
- [x] Waterfall plot (explicação individual)
- [x] Dependence plot (relação feature mais importante)

### ✅ Explicabilidade com LIME
- [x] Explicações locais para amostras específicas
- [x] Análise de casos: TP, TN, FP, FN
- [x] Visualizações individuais

### ✅ Comparação SHAP vs LIME
- [x] Comparação visual lado a lado
- [x] Análise de concordância/divergência
- [x] Discussão teórica das diferenças

---

## 📁 Estrutura do Projeto

```
Trabalho-Final-IDS/
├── main.py                # Código principal (completo e refatorado)
├── main_old.py           # Backup do código original
├── requirements.txt      # Dependências do projeto
├── README.md            # Este arquivo
├── KDDTest+.txt         # Dataset de treino (22.543 amostras)
├── KDDTest-21.txt       # Dataset de teste (11.849 amostras)
└── output/              # Diretório com visualizações (gerado automaticamente)
    ├── 01_confusion_matrix.png
    ├── 02_roc_curve.png
    ├── 03_rf_feature_importance.png
    ├── 04_shap_summary_plot.png
    ├── 05_shap_bar_plot.png
    ├── 06_shap_waterfall_attack.png
    ├── 07_shap_dependence_plot.png
    ├── 08_lime_example_1_TP.png
    ├── 08_lime_example_2_TN.png
    ├── 08_lime_example_3_FP.png
    ├── 08_lime_example_4_FN.png
    ├── 08_lime_example_5_TP.png
    ├── 09_comparison_1_TP.png
    ├── 09_comparison_2_TN.png
    ├── 09_comparison_3_FP.png
    ├── 09_comparison_4_FN.png
    └── 09_comparison_5_TP.png
```

---

## 🚀 Como Executar

### 1. Instalar Dependências

```bash
pip install -r requirements.txt
```

**Dependências:**
- pandas >= 2.0.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- shap >= 0.42.0
- lime >= 0.2.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0

### 2. Executar o Projeto

```bash
python main.py
```

### 3. Resultados

Todas as visualizações serão salvas no diretório `output/` automaticamente.

---

## 📊 Fluxo de Execução

O código executa automaticamente as seguintes etapas:

### **ETAPA 1: Carregamento e Pré-processamento**
- Leitura dos datasets NSL-KDD
- Encoding de variáveis categóricas (`protocol_type`, `service`, `flag`)
- Conversão de labels para binário (0=normal, 1=ataque)
- Remoção da coluna `difficulty`

### **ETAPA 2: Treinamento do Modelo**
- Random Forest com 100 estimadores
- Treinamento usando todas as cores do processador (`n_jobs=-1`)

### **ETAPA 3: Avaliação do Modelo**
- Cálculo de métricas: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Geração de matriz de confusão
- Geração de curva ROC
- Feature importance do Random Forest

### **ETAPA 4: Explicabilidade com SHAP**
- Cálculo de SHAP values para 500 amostras
- Summary plot (importância global)
- Bar plot (importância média)
- Waterfall plot (exemplo de ataque)
- Dependence plot (feature mais importante)

### **ETAPA 5: Explicabilidade com LIME**
- Criação do LimeTabularExplainer
- Explicação de 5 exemplos variados:
  - TP (True Positive) - Ataque corretamente detectado
  - TN (True Negative) - Normal corretamente classificado
  - FP (False Positive) - Falso alarme
  - FN (False Negative) - Ataque não detectado

### **ETAPA 6: Comparação SHAP vs LIME**
- Comparação visual lado a lado
- Análise de concordância entre as técnicas
- Discussão teórica das diferenças

---

## 📈 Visualizações Geradas

### 1. Avaliação do Modelo
- **Matriz de Confusão:** Distribuição de TP, TN, FP, FN
- **Curva ROC:** Desempenho do classificador
- **Feature Importance:** Top 15 features mais importantes (Random Forest)

### 2. SHAP (Explicabilidade Global e Local)
- **Summary Plot:** Visão global da importância e impacto das features
- **Bar Plot:** Importância média absoluta das features
- **Waterfall Plot:** Como cada feature contribuiu para uma predição específica
- **Dependence Plot:** Relação entre valor da feature e SHAP value

### 3. LIME (Explicabilidade Local)
- **Explicações individuais:** Por que o modelo classificou cada amostra específica
- **Casos variados:** TP, TN, FP, FN

### 4. Comparação SHAP vs LIME
- **Comparações lado a lado:** Mesmas amostras explicadas por ambas técnicas
- **Análise de concordância:** Features importantes em ambas

---

## 🔍 Diferenças SHAP vs LIME

### SHAP (SHapley Additive exPlanations)
**Características:**
- ✅ Baseado em teoria dos jogos (valores de Shapley)
- ✅ Garante propriedades matemáticas desejáveis
- ✅ Explicação global E local
- ✅ Mais eficiente para modelos baseados em árvores
- ✅ Valores representam contribuição exata para a predição
- ✅ Deterministico e consistente

**Quando usar:**
- Modelos baseados em árvores (Random Forest, XGBoost, etc.)
- Quando precisar de explicações globais
- Quando consistência é crítica

### LIME (Local Interpretable Model-agnostic Explanations)
**Características:**
- ✅ Model-agnostic (funciona com qualquer modelo)
- ✅ Aproximação linear local
- ✅ Explicação puramente local
- ⚠️ Estocástico (pode variar entre execuções)
- ✅ Mais intuitivo para não-especialistas
- ✅ Captura bem não-linearidades locais

**Quando usar:**
- Qualquer tipo de modelo (inclusive caixas-pretas)
- Quando precisar explicar predições específicas
- Quando interpretabilidade é mais importante que exatidão matemática

### Concordância Esperada
- Ambos concordam nas features mais importantes
- Diferenças indicam comportamentos complexos do modelo
- SHAP tende a ser mais estável
- LIME pode capturar nuances locais que SHAP não vê

---

## ⚠️ Nota sobre os Datasets

**Importante:** O arquivo `KDDTrain+.txt` original do NSL-KDD não estava disponível no repositório.

**Solução adotada:**
- Utilizamos `KDDTest+.txt` (22.543 amostras) como conjunto de treino
- Utilizamos `KDDTest-21.txt` (11.849 amostras) como conjunto de teste

**Para uso em produção:**
- Recomenda-se obter o `KDDTrain+.txt` original (~125.000 amostras)
- Disponível em: https://www.unb.ca/cic/datasets/nsl.html
- Basta substituir no código: `load_and_preprocess_data(train_file='KDDTrain+.txt')`

---

## 🎓 Para o Relatório Final

### Tópicos para Discussão

1. **Desempenho do Modelo**
   - Analise as métricas obtidas
   - Discuta precision vs recall no contexto de IDS
   - Explique o trade-off entre falsos positivos e falsos negativos

2. **Explicabilidade com SHAP**
   - Quais features foram mais importantes globalmente?
   - O que o summary plot revela sobre o dataset?
   - Como interpretar o waterfall plot?

3. **Explicabilidade com LIME**
   - Como as explicações locais ajudam a entender predições específicas?
   - Por que o modelo errou nos casos FP e FN?
   - Quais features foram determinantes em cada caso?

4. **Comparação SHAP vs LIME**
   - Houve concordância entre as técnicas?
   - Onde divergiram? Por quê?
   - Qual técnica foi mais útil para seu caso de uso?
   - Vantagens e desvantagens de cada abordagem

5. **Implicações para IDS**
   - Como explicabilidade ajuda em sistemas de detecção de intrusões?
   - Como usar XAI para reduzir falsos positivos?
   - Como explicar alertas para analistas de segurança?

---

## 🛠️ Personalização

### Alterar Número de Amostras SHAP
```python
shap_explainer, shap_values, X_test_sample = explain_with_shap(
    model, X_train, X_test, feature_names, n_samples=1000  # Altere aqui
)
```

### Alterar Número de Exemplos LIME
```python
lime_explainer, lime_explanations, examples, labels_desc = explain_with_lime(
    model, X_train, X_test, y_test, feature_names, n_examples=10  # Altere aqui
)
```

### Alterar Número de Árvores do Random Forest
```python
model = train_model(X_train, y_train, n_estimators=200)  # Altere aqui
```

---

## 📚 Referências

- **NSL-KDD Dataset:** https://www.unb.ca/cic/datasets/nsl.html
- **SHAP:** Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions.
- **LIME:** Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?" Explaining the predictions of any classifier.
- **Random Forest:** Breiman, L. (2001). Random forests. Machine learning, 45(1), 5-32.

---

## ✅ Status do Projeto

- [x] Leitura e pré-processamento de dados
- [x] Treinamento do modelo Random Forest
- [x] Avaliação completa (métricas, confusion matrix, ROC)
- [x] Implementação SHAP (global e local)
- [x] Implementação LIME (local)
- [x] Comparação SHAP vs LIME
- [x] Visualizações completas
- [x] Documentação

**Projeto pronto para entrega!** 🎉

---

## 👨‍💻 Autor

Trabalho desenvolvido para a disciplina de Engenharia de Sistemas de Detecção de Intrusões.

Professor: Silvio Quincozes
