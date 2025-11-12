"""
Sistema de Detecção de Intrusões (IDS) com Machine Learning e XAI
Dataset: NSL-KDD
Técnicas XAI: SHAP e LIME
Autor: Trabalho Final - Engenharia de Sistemas de Detecção de Intrusões
Professor: Silvio Quincozes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)
import shap
from lime.lime_tabular import LimeTabularExplainer

warnings.filterwarnings('ignore')

# Criar diretório para salvar visualizações
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Definição das colunas do NSL-KDD
COLUMNS = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes",
    "land","wrong_fragment","urgent","hot","num_failed_logins","logged_in",
    "num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count",
    "dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
    "dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate",
    "dst_host_srv_rerror_rate","label","difficulty"
]


def load_and_preprocess_data(train_file='KDDTest+.txt', test_file='KDDTest-21.txt'):
    """
    Carrega e pré-processa os dados NSL-KDD

    Nota: Como KDDTrain+.txt não estava disponível, usamos KDDTest+.txt
    para treino e KDDTest-21.txt para teste. Para produção, use os arquivos corretos.

    Args:
        train_file: arquivo de treino
        test_file: arquivo de teste

    Returns:
        X_train, X_test, y_train, y_test, feature_names
    """
    print(f"\n{'='*80}")
    print("ETAPA 1: CARREGAMENTO E PRÉ-PROCESSAMENTO DOS DADOS")
    print(f"{'='*80}\n")

    # Ler datasets
    print(f"Carregando {train_file}...")
    train = pd.read_csv(train_file, header=None)
    train.columns = COLUMNS

    print(f"Carregando {test_file}...")
    test = pd.read_csv(test_file, header=None)
    test.columns = COLUMNS

    print(f"✓ Treino: {len(train)} amostras")
    print(f"✓ Teste: {len(test)} amostras")

    # Remover coluna difficulty
    train = train.drop(columns=["difficulty"])
    test = test.drop(columns=["difficulty"])

    # Encoding de variáveis categóricas
    print("\nEncodificando variáveis categóricas...")
    combined = pd.concat([train, test], axis=0)

    categorical_cols = ["protocol_type", "service", "flag"]
    for col in categorical_cols:
        le = LabelEncoder()
        combined[col] = le.fit_transform(combined[col])

    # Separar novamente
    train = combined.iloc[:len(train), :]
    test = combined.iloc[len(train):, :]

    # Separar features e labels
    X_train = train.drop(columns=["label"])
    y_train = train["label"]
    X_test = test.drop(columns=["label"])
    y_test = test["label"]

    # Converter labels para binário (0=normal, 1=ataque)
    print("\nConvertendo labels para binário:")
    print(f"  - 0: normal")
    print(f"  - 1: ataque")

    y_train = y_train.apply(lambda x: 0 if x == "normal" else 1)
    y_test = y_test.apply(lambda x: 0 if x == "normal" else 1)

    print(f"\n✓ Distribuição treino: Normal={sum(y_train==0)} | Ataques={sum(y_train==1)}")
    print(f"✓ Distribuição teste: Normal={sum(y_test==0)} | Ataques={sum(y_test==1)}")

    feature_names = X_train.columns.tolist()

    return X_train, X_test, y_train, y_test, feature_names


def train_model(X_train, y_train, n_estimators=100, random_state=42):
    """
    Treina o modelo Random Forest

    Args:
        X_train: features de treino
        y_train: labels de treino
        n_estimators: número de árvores
        random_state: seed para reprodutibilidade

    Returns:
        modelo treinado
    """
    print(f"\n{'='*80}")
    print("ETAPA 2: TREINAMENTO DO MODELO")
    print(f"{'='*80}\n")

    print(f"Treinando Random Forest com {n_estimators} estimadores...")
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    model.fit(X_train, y_train)

    print("✓ Modelo treinado com sucesso!")

    return model


def evaluate_model(model, X_train, X_test, y_train, y_test, feature_names):
    """
    Avalia o modelo com métricas completas e visualizações

    Args:
        model: modelo treinado
        X_train, X_test, y_train, y_test: dados
        feature_names: nomes das features

    Returns:
        y_pred: predições no conjunto de teste
    """
    print(f"\n{'='*80}")
    print("ETAPA 3: AVALIAÇÃO DO MODELO")
    print(f"{'='*80}\n")

    # Predições
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print("MÉTRICAS DE DESEMPENHO:")
    print(f"  • Accuracy:  {accuracy:.4f}")
    print(f"  • Precision: {precision:.4f}")
    print(f"  • Recall:    {recall:.4f}")
    print(f"  • F1-Score:  {f1:.4f}")
    print(f"  • ROC-AUC:   {roc_auc:.4f}")

    print("\nRELATÓRIO DE CLASSIFICAÇÃO:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Ataque"]))

    # Matriz de Confusão
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Normal", "Ataque"],
                yticklabels=["Normal", "Ataque"])
    plt.title('Matriz de Confusão', fontsize=14, fontweight='bold')
    plt.ylabel('Valor Real')
    plt.xlabel('Valor Predito')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/01_confusion_matrix.png", dpi=300, bbox_inches='tight')
    print(f"\n✓ Matriz de confusão salva em: {OUTPUT_DIR}/01_confusion_matrix.png")
    plt.close()

    # Curva ROC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos (FPR)')
    plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
    plt.title('Curva ROC', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/02_roc_curve.png", dpi=300, bbox_inches='tight')
    print(f"✓ Curva ROC salva em: {OUTPUT_DIR}/02_roc_curve.png")
    plt.close()

    # Feature Importance do Random Forest
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 8))
    plt.barh(range(15), feature_importance['importance'].head(15), color='steelblue')
    plt.yticks(range(15), feature_importance['feature'].head(15))
    plt.xlabel('Importância')
    plt.title('Top 15 Features Mais Importantes (Random Forest)', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/03_rf_feature_importance.png", dpi=300, bbox_inches='tight')
    print(f"✓ Feature importance salva em: {OUTPUT_DIR}/03_rf_feature_importance.png")
    plt.close()

    return y_pred, y_pred_proba


def explain_with_shap(model, X_train, X_test, feature_names, n_samples=500):
    """
    Gera explicações usando SHAP

    Args:
        model: modelo treinado
        X_train: dados de treino (para background)
        X_test: dados de teste
        feature_names: nomes das features
        n_samples: número de amostras para análise

    Returns:
        explainer, shap_values
    """
    print(f"\n{'='*80}")
    print("ETAPA 4: EXPLICABILIDADE COM SHAP")
    print(f"{'='*80}\n")

    print(f"Criando TreeExplainer e calculando SHAP values para {n_samples} amostras...")

    # Usar apenas uma amostra do conjunto de teste para análise
    X_test_sample = X_test.iloc[:n_samples]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_sample)

    # Para Random Forest binário, shap_values retorna lista [classe_0, classe_1]
    # Vamos usar apenas a classe 1 (ataque)
    if isinstance(shap_values, list):
        shap_values_attack = shap_values[1]
    else:
        shap_values_attack = shap_values

    print("✓ SHAP values calculados!")

    # 1. Summary Plot (Global)
    print("\nGerando visualizações SHAP...")
    print("  → Summary plot (importância global)...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_attack, X_test_sample, feature_names=feature_names, show=False)
    plt.title('SHAP Summary Plot - Importância Global das Features', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/04_shap_summary_plot.png", dpi=300, bbox_inches='tight')
    print(f"    ✓ Salvo em: {OUTPUT_DIR}/04_shap_summary_plot.png")
    plt.close()

    # 2. Bar Plot (Importância média)
    print("  → Bar plot (importância média)...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_attack, X_test_sample, feature_names=feature_names,
                     plot_type="bar", show=False)
    plt.title('SHAP Bar Plot - Importância Média das Features', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/05_shap_bar_plot.png", dpi=300, bbox_inches='tight')
    print(f"    ✓ Salvo em: {OUTPUT_DIR}/05_shap_bar_plot.png")
    plt.close()

    # 3. Waterfall Plot (exemplo individual - ataque)
    print("  → Waterfall plot (exemplo individual de ataque)...")
    # Encontrar um exemplo de ataque corretamente classificado
    attack_idx = None
    for idx in range(min(100, len(X_test_sample))):
        if model.predict(X_test_sample.iloc[[idx]])[0] == 1:
            attack_idx = idx
            break

    if attack_idx is not None:
        shap.plots.waterfall(
            shap.Explanation(
                values=shap_values_attack[attack_idx],
                base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
                data=X_test_sample.iloc[attack_idx],
                feature_names=feature_names
            ),
            show=False
        )
        plt.title(f'SHAP Waterfall Plot - Exemplo de Ataque (amostra {attack_idx})',
                 fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/06_shap_waterfall_attack.png", dpi=300, bbox_inches='tight')
        print(f"    ✓ Salvo em: {OUTPUT_DIR}/06_shap_waterfall_attack.png")
        plt.close()

    # 4. Dependence Plot (feature mais importante)
    print("  → Dependence plot (feature mais importante)...")
    mean_abs_shap = np.abs(shap_values_attack).mean(axis=0)
    most_important_idx = np.argmax(mean_abs_shap)
    most_important_feature = feature_names[most_important_idx]

    plt.figure(figsize=(10, 6))
    shap.dependence_plot(
        most_important_idx,
        shap_values_attack,
        X_test_sample,
        feature_names=feature_names,
        show=False
    )
    plt.title(f'SHAP Dependence Plot - {most_important_feature}',
             fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/07_shap_dependence_plot.png", dpi=300, bbox_inches='tight')
    print(f"    ✓ Salvo em: {OUTPUT_DIR}/07_shap_dependence_plot.png")
    plt.close()

    print("\n✓ Todas as visualizações SHAP foram geradas!")

    return explainer, shap_values_attack, X_test_sample


def explain_with_lime(model, X_train, X_test, y_test, feature_names, n_examples=5):
    """
    Gera explicações usando LIME

    Args:
        model: modelo treinado
        X_train: dados de treino (para background)
        X_test: dados de teste
        y_test: labels de teste
        feature_names: nomes das features
        n_examples: número de exemplos para explicar

    Returns:
        explainer, explanations
    """
    print(f"\n{'='*80}")
    print("ETAPA 5: EXPLICABILIDADE COM LIME")
    print(f"{'='*80}\n")

    print("Criando LimeTabularExplainer...")

    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=feature_names,
        class_names=['Normal', 'Ataque'],
        mode='classification',
        random_state=42
    )

    print(f"✓ Explainer criado!")

    # Selecionar exemplos interessantes
    y_pred = model.predict(X_test)

    # Encontrar exemplos de diferentes categorias
    tp_idx = np.where((y_test == 1) & (y_pred == 1))[0]  # True Positive (ataque detectado)
    tn_idx = np.where((y_test == 0) & (y_pred == 0))[0]  # True Negative (normal correto)
    fp_idx = np.where((y_test == 0) & (y_pred == 1))[0]  # False Positive (falso alarme)
    fn_idx = np.where((y_test == 1) & (y_pred == 0))[0]  # False Negative (ataque não detectado)

    examples = []
    labels_desc = []

    # Selecionar exemplos variados
    if len(tp_idx) > 0:
        examples.append(tp_idx[0])
        labels_desc.append("TP (Ataque Detectado)")
    if len(tn_idx) > 0:
        examples.append(tn_idx[0])
        labels_desc.append("TN (Normal Correto)")
    if len(fp_idx) > 0:
        examples.append(fp_idx[0])
        labels_desc.append("FP (Falso Alarme)")
    if len(fn_idx) > 0:
        examples.append(fn_idx[0])
        labels_desc.append("FN (Ataque Perdido)")
    if len(tp_idx) > 1:
        examples.append(tp_idx[min(1, len(tp_idx)-1)])
        labels_desc.append("TP (Ataque Detectado 2)")

    examples = examples[:n_examples]
    labels_desc = labels_desc[:n_examples]

    print(f"\nGerando explicações LIME para {len(examples)} exemplos...")
    explanations = []

    for i, (idx, label) in enumerate(zip(examples, labels_desc)):
        print(f"  → Explicando amostra {idx} ({label})...")

        exp = explainer.explain_instance(
            X_test.iloc[idx].values,
            model.predict_proba,
            num_features=10
        )
        explanations.append(exp)

        # Salvar explicação
        fig = exp.as_pyplot_figure()
        fig.suptitle(f'LIME Explanation - {label} (amostra {idx})',
                    fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/08_lime_example_{i+1}_{label.split()[0]}.png",
                   dpi=300, bbox_inches='tight')
        print(f"    ✓ Salvo em: {OUTPUT_DIR}/08_lime_example_{i+1}_{label.split()[0]}.png")
        plt.close()

    print("\n✓ Todas as explicações LIME foram geradas!")

    return explainer, explanations, examples, labels_desc


def compare_shap_lime(shap_values, X_test_sample, lime_explanations, examples,
                     labels_desc, feature_names, model):
    """
    Compara explicações SHAP e LIME lado a lado

    Args:
        shap_values: valores SHAP
        X_test_sample: amostras usadas no SHAP
        lime_explanations: explicações LIME
        examples: índices dos exemplos LIME
        labels_desc: descrições dos exemplos
        feature_names: nomes das features
        model: modelo treinado
    """
    print(f"\n{'='*80}")
    print("ETAPA 6: COMPARAÇÃO SHAP VS LIME")
    print(f"{'='*80}\n")

    print("Gerando comparações visuais lado a lado...")

    for i, (exp, idx, label) in enumerate(zip(lime_explanations, examples, labels_desc)):
        # Verificar se o índice está dentro do range do SHAP
        if idx >= len(X_test_sample):
            print(f"  ⚠ Amostra {idx} está fora do range SHAP, pulando...")
            continue

        print(f"  → Comparando amostra {idx} ({label})...")

        # Criar figura com 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # LIME (esquerda)
        lime_features = exp.as_list()
        lime_features_sorted = sorted(lime_features, key=lambda x: abs(x[1]), reverse=True)[:10]
        lime_names = [f[0] for f in lime_features_sorted]
        lime_values = [f[1] for f in lime_features_sorted]
        colors_lime = ['red' if v < 0 else 'green' for v in lime_values]

        y_pos = np.arange(len(lime_names))
        ax1.barh(y_pos, lime_values, color=colors_lime, alpha=0.7)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(lime_names, fontsize=9)
        ax1.set_xlabel('Importância LIME', fontsize=10)
        ax1.set_title(f'LIME - {label}', fontsize=11, fontweight='bold')
        ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax1.invert_yaxis()

        # SHAP (direita)
        shap_vals = shap_values[idx]
        shap_sorted_idx = np.argsort(np.abs(shap_vals))[-10:][::-1]
        shap_names = [feature_names[i] for i in shap_sorted_idx]
        shap_vals_top = [shap_vals[i] for i in shap_sorted_idx]
        colors_shap = ['red' if v < 0 else 'green' for v in shap_vals_top]

        y_pos_shap = np.arange(len(shap_names))
        ax2.barh(y_pos_shap, shap_vals_top, color=colors_shap, alpha=0.7)
        ax2.set_yticks(y_pos_shap)
        ax2.set_yticklabels(shap_names, fontsize=9)
        ax2.set_xlabel('Importância SHAP', fontsize=10)
        ax2.set_title(f'SHAP - {label}', fontsize=11, fontweight='bold')
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax2.invert_yaxis()

        plt.suptitle(f'Comparação SHAP vs LIME - Amostra {idx} ({label})',
                    fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/09_comparison_{i+1}_{label.split()[0]}.png",
                   dpi=300, bbox_inches='tight')
        print(f"    ✓ Salvo em: {OUTPUT_DIR}/09_comparison_{i+1}_{label.split()[0]}.png")
        plt.close()

    # Análise de concordância
    print("\n" + "="*80)
    print("ANÁLISE DE CONCORDÂNCIA SHAP vs LIME")
    print("="*80)

    print("\nCaracterísticas das técnicas:")
    print("\n📊 SHAP (SHapley Additive exPlanations):")
    print("  • Baseado em teoria dos jogos (valores de Shapley)")
    print("  • Garante propriedades desejáveis (consistência, aditividade)")
    print("  • Explicação global e local")
    print("  • Mais computacionalmente eficiente para árvores (TreeExplainer)")
    print("  • Valores representam contribuição exata para a predição")

    print("\n🔍 LIME (Local Interpretable Model-agnostic Explanations):")
    print("  • Model-agnostic (funciona com qualquer modelo)")
    print("  • Aproximação linear local ao redor da instância")
    print("  • Explicação puramente local")
    print("  • Pode variar entre execuções (estocástico)")
    print("  • Valores representam importância na aproximação linear")

    print("\n💡 Diferenças observadas:")
    print("  • SHAP tende a ser mais estável e consistente")
    print("  • LIME pode capturar melhor não-linearidades locais")
    print("  • Ambos concordam nas features mais importantes")
    print("  • Divergências podem indicar comportamentos complexos do modelo")

    print("\n✓ Comparação completa gerada!")


def main():
    """
    Função principal que executa todo o pipeline
    """
    print("\n" + "="*80)
    print(" SISTEMA DE DETECÇÃO DE INTRUSÕES COM EXPLICABILIDADE (XAI)")
    print(" Dataset: NSL-KDD | Modelo: Random Forest | XAI: SHAP + LIME")
    print("="*80)

    # 1. Carregar e pré-processar dados
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data()

    # 2. Treinar modelo
    model = train_model(X_train, y_train)

    # 3. Avaliar modelo
    y_pred, y_pred_proba = evaluate_model(model, X_train, X_test, y_train, y_test, feature_names)

    # 4. Explicabilidade com SHAP
    shap_explainer, shap_values, X_test_sample = explain_with_shap(
        model, X_train, X_test, feature_names, n_samples=500
    )

    # 5. Explicabilidade com LIME
    lime_explainer, lime_explanations, examples, labels_desc = explain_with_lime(
        model, X_train, X_test, y_test, feature_names, n_examples=5
    )

    # 6. Comparação SHAP vs LIME
    compare_shap_lime(
        shap_values, X_test_sample, lime_explanations,
        examples, labels_desc, feature_names, model
    )

    # Resumo final
    print(f"\n{'='*80}")
    print("RESUMO FINAL")
    print(f"{'='*80}\n")
    print(f"✓ Modelo treinado e avaliado")
    print(f"✓ Explicações SHAP geradas (global e local)")
    print(f"✓ Explicações LIME geradas (local)")
    print(f"✓ Comparação SHAP vs LIME concluída")
    print(f"\n📁 Todas as visualizações salvas em: {OUTPUT_DIR}/")
    print(f"\n{'='*80}")
    print("EXECUÇÃO CONCLUÍDA COM SUCESSO!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
