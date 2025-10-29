import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import shap


# Ler os datasets
train = pd.read_csv('KDDTest-21.txt', header=None)
test = pd.read_csv('KDDTest+.txt', header=None)

cols = [
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

train.columns = cols
test.columns = cols

train = train.drop(columns=["difficulty"])
test = test.drop(columns=["difficulty"])

# 🔧 Juntar treino e teste só para codificar categorias sem erro
combined = pd.concat([train, test], axis=0)

for col in ["protocol_type", "service", "flag"]:
    le = LabelEncoder()
    combined[col] = le.fit_transform(combined[col])

# Separar de novo
train = combined.iloc[:len(train), :]
test = combined.iloc[len(train):, :]

# Separar features e rótulos
X_train = train.drop(columns=["label"])
y_train = train["label"]
X_test = test.drop(columns=["label"])
y_test = test["label"]

# Converter rótulos para 0/1
y_train = y_train.apply(lambda x: 0 if x == "normal" else 1)
y_test = y_test.apply(lambda x: 0 if x == "normal" else 1)

# Treinar modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Testar modelo
y_pred = model.predict(X_test)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test[:100])

print("\nGerando grafico de explicabilidade...")

# Verifica se o SHAP retornou uma lista (binário) ou matriz única (único output)
if isinstance(shap_values, list):
    shap.summary_plot(shap_values[1], X_test[:100])  # classe 1 = ataque
else:
    shap.summary_plot(shap_values, X_test[:100])
