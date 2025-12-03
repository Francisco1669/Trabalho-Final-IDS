# Trabalho Final - Engenharia de Sistemas de Detecção de Intrusões - 2o Semestre de 2025

- **Grupo 8 - XAI**
- **Professor:** Silvio Quincozes
- **Dataset:** NSL-KDD
- **Ferramentas de XAI:** SHAP e LIME

## Descrição do Trabalho

O objetivo central deste trabalho é desenvolver um classificador para detecção de intrusões de rede e aplicar técnicas de Inteligência Artificial Explicável (XAI) para compreender como o modelo toma suas decisões.

O desenvolvimento foi dividido nas seguintes etapas:

1. **Pré-processamento de Dados:**
    * Tratamento do dataset NSL-KDD, incluindo a nomeação correta das 41 colunas de tráfego de rede.
    * Remoção de artefatos de metadados (coluna `difficulty`) para evitar vazamento de dados (*data leakage*).
    * Codificação de variáveis categóricas (*Label Encoding*) e binarização do alvo (Normal vs. Ataque).

2.  **Modelagem Preditiva:**
    * Treinamento de um classificador **XGBoost** (*Extreme Gradient Boosting*).
    * Aplicação de técnicas de balanceamento de classes utilizando o hiperparâmetro `scale_pos_weight`, resultando em um modelo com Recall de 0.99 para detecção de ataques.

3.  **Explainable AI (XAI):**
    * Implementação do **SHAP** (*SHapley Additive exPlanations*) para análise de consistência global e atribuição exata de importância das *features*.
    * Implementação do **LIME** (*Local Interpretable Model-agnostic Explanations*) para geração de regras de decisão locais.
    * Análise comparativa entre as duas abordagens para validar a confiabilidade das detecções e identificar redundâncias (correlações) nos dados.

## Organização do Repositório

A estrutura de diretórios deste projeto está organizada da seguinte forma:

* **code/**: Contém o *Jupyter Notebook* (`.ipynb`) com todo o código fonte desenvolvido, desde a carga de dados até a geração dos gráficos de explicabilidade.
* **data/**: Armazena os arquivos brutos do dataset NSL-KDD em formato `.txt` (ex: `KDDTest-21.txt`).
* **output/**: Contém os artefatos gerados pela execução do XAI, incluindo os arquivos HTML interativos das explicações locais do SHAP e LIME.

## Instalação e Execução

Siga os passos abaixo para configurar o ambiente de desenvolvimento e executar o projeto.

### 1. Criação do Ambiente Virtual

É recomendado utilizar um ambiente virtual para isolar as dependências do projeto. Abra o terminal na raiz do projeto e execute:

**Windows:**
```bash
python -m venv venv
.\venv\Scripts\activate
```

Após criar o ambiente virtual e ativa-lo, instale as dependências:
```bash
pip intall pandas numpy scikit-learn xgboost plotly shap lime 
```

---
