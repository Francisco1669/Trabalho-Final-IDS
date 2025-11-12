# Guia de Instalação

## Instalação das Dependências

### Opção 1: Instalação Completa (Recomendada)

```bash
pip install pandas numpy scikit-learn shap lime matplotlib seaborn
```

### Opção 2: Se LIME Falhar

O LIME pode ter problemas de compilação em alguns ambientes. Se você encontrar erros ao instalar LIME:

#### Solução A: Instalar Dependências de Build

No Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install python3-dev build-essential
pip install lime
```

No Fedora/RHEL:
```bash
sudo dnf install python3-devel gcc
pip install lime
```

No macOS:
```bash
xcode-select --install
pip install lime
```

#### Solução B: Usar Conda (Mais Confiável)

```bash
conda create -n ids-xai python=3.10
conda activate ids-xai
conda install -c conda-forge pandas numpy scikit-learn matplotlib seaborn
pip install shap lime
```

#### Solução C: Versão Específica do LIME

```bash
pip install lime==0.2.0.1
```

#### Solução D: Instalar do GitHub

```bash
pip install git+https://github.com/marcotcr/lime.git
```

### Opção 3: Executar Apenas com SHAP

Se LIME não instalar, você pode executar uma versão do código apenas com SHAP. Veja `main_shap_only.py`.

## Verificação da Instalação

Execute este comando para verificar se todas as bibliotecas foram instaladas:

```bash
python3 -c "import pandas, numpy, sklearn, shap, lime, matplotlib, seaborn; print('✓ Todas as dependências instaladas com sucesso!')"
```

## Requisitos de Sistema

- **Python:** 3.8 ou superior
- **RAM:** Mínimo 4GB (recomendado 8GB)
- **Espaço em Disco:** ~500MB para dependências
- **Processador:** Qualquer (multi-core recomendado para treino mais rápido)

## Problemas Comuns

### 1. Erro "No module named 'sklearn'"

```bash
pip install scikit-learn
```

### 2. Erro com matplotlib no macOS

```bash
pip install matplotlib --upgrade
```

### 3. Erro "Failed building wheel for lime"

Veja "Solução A" acima para instalar dependências de compilação.

### 4. Erro de memória durante execução

Reduza o número de amostras SHAP no código:
```python
explain_with_shap(..., n_samples=100)  # ao invés de 500
```

## Instalação em Ambiente Virtual (Recomendado)

```bash
# Criar ambiente virtual
python3 -m venv venv

# Ativar ambiente virtual
# No Linux/macOS:
source venv/bin/activate
# No Windows:
venv\Scripts\activate

# Instalar dependências
pip install -r requirements.txt
```

## Testando a Instalação

Após instalar tudo, teste com:

```bash
python3 main.py
```

Se tudo estiver correto, você verá:
```
================================================================================
 SISTEMA DE DETECÇÃO DE INTRUSÕES COM EXPLICABILIDADE (XAI)
 Dataset: NSL-KDD | Modelo: Random Forest | XAI: SHAP + LIME
================================================================================
```

## Suporte

Se encontrar problemas de instalação:

1. Verifique se está usando Python 3.8+
2. Tente usar um ambiente virtual
3. Considere usar conda ao invés de pip
4. Use a versão apenas com SHAP se LIME não funcionar
