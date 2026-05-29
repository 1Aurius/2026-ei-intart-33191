import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# 1. Carregar os dados
data = load_breast_cancer()
X, y = data.data, data.target

df = pd.DataFrame(X, columns=data.feature_names)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

df['target'] = y

# 2. Exploração de Dados no Terminal
print("--- Basic Statistics ---")
print(df.describe())
print("\n--- Class Distribution ---")
print(df['target'].value_counts())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)


print("\nA gerar gráficos de análise exploratória...")

# Gráfico 1: Heatmap de Correlação
plt.figure(figsize=(14, 12))
matriz_correlacao = df.corr()
sns.heatmap(matriz_correlacao, cmap='coolwarm', annot=False, fmt=".2f", linewidths=0.5)
plt.title('Matriz de Correlação - Breast Cancer Dataset', fontsize=16)
plt.tight_layout()
plt.savefig('heatmap_correlacao.png', dpi=300)
plt.close()

# Gráfico 2: Boxplot (Caos vs Uniformidade)
plt.figure(figsize=(8, 6))
sns.boxplot(x='target', y='worst area', hue='target', data=df, palette={0: '#ff9999', 1: '#99ff99'}, legend=False)
plt.title('Distribuição de "Worst Area": Maligno (0) vs Benigno (1)', fontsize=14)
plt.xlabel('Diagnóstico (0 = Maligno, 1 = Benigno)')
plt.ylabel('Worst Area')
plt.tight_layout()
plt.savefig('boxplot_worst_area.png', dpi=300)
plt.close()

print("Sucesso: Imagens guardadas na diretoria atual.")

y_pred = model.predict(X_test_scaled)
print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=['Malignant (0)', 'Benign (1)']))