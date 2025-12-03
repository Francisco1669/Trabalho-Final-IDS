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


## Resultados e Discussão

Os experimentos validaram tanto a eficácia do classificador quanto a necessidade de múltiplas ferramentas de auditoria para garantir a confiabilidade do sistema.

### 1. Desempenho do Classificador (XGBoost)
Utilizando o hiperparâmetro `scale_pos_weight` para tratar o desbalanceamento nativo do NSL-KDD, o modelo priorizou a sensibilidade da detecção, vital para cenários de segurança crítica.

* **Recall (Classe Ataque):** **0.99** (O modelo falhou em detectar apenas 18 intrusões em um universo de 2.909 tentativas).
* **F1-Score Geral:** **0.99**.
* **Conclusão:** O classificador demonstrou robustez excepcional, mantendo uma taxa de Falsos Positivos extremamente baixa (11 ocorrências).

### 2. Comparativo SHAP vs. LIME
O principal objetivo deste projeto foi a análise das diferenças entre os métodos de explicabilidade de IA. A alta correlação entre as métricas de rede (identificada na matriz de correlação) impactou as ferramentas de forma distinta:

| Critério | SHAP (Consistência Global) | LIME (Interpretação Local) |
| :--- | :--- | :--- |
| **Foco da Explicação** | **Causa Raiz Técnica:** Apontou a taxa de erros de sincronização (`serror_rate`) como fator determinante (assinatura de *SYN Flood*). | **Regras de Fronteira:** Focou em sintomas secundários, como a ausência de resposta (`dst_bytes = 0`) e protocolo. |
| **Robustez à Correlação** | **Alta:** Lidou bem com a multicolinearidade, atribuindo peso à variável representativa e ignorando redundâncias. | **Média/Baixa:** Apresentou instabilidade ao gerar perturbações em cenários de alta dependência entre variáveis. |
| **Aplicação Recomendada** | Auditoria Forense e validação de engenharia do modelo. | Triagem rápida de alertas para operadores de Nível 1. |

Portanto, o projeto validou um classificador XGBoost binário (com Recall de 0.99) para detecção de intrusões, evidenciando que a confiabilidade do sistema depende da ferramenta de auditoria escolhida. A análise comparativa demonstrou que o SHAP oferece maior robustez ao lidar consistentemente com a alta correlação dos dados e identificar a causa raiz técnica, superando a instabilidade do LIME frente à redundância estatística. Enquanto o LIME lida melhor em um cenário detriagem rápida com regras simples, o SHAP é indispensável para a validação mais detalhada.

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


## Link do Artigo Final
Para o acessar o artigo basta entrar no [link](https://pt.overleaf.com/6355131817sgrcggxyyxnw#b66ec9).