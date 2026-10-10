"""
Prototipo Secuencial Inicial (Fase 1 - Pre-Refactorización).
Propósito: Demostrar el estado inicial del código antes de aplicar
la arquitectura modular y los patrones de Programación Orientada a Objetos (POO).
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Carga lineal de datos
url_or_path = "data/raw/heart_failure_clinical_records_dataset.csv"
print("Cargando dataset...")
df = pd.read_csv(url_or_path)

print(f"Dimensiones del dataset: {df.shape}")
print(df.head())

# 2. Separación de variables (acoplamiento directo)
X = df.drop(columns=["DEATH_EVENT"])
y = df["DEATH_EVENT"]

# 3. Partición de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Entrenamiento de modelos simples (sin pipeline ni escalado)
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)
print(f"Accuracy Regresión Logística: {accuracy_score(y_test, lr_preds):.4f}")

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
print(f"Accuracy Random Forest: {accuracy_score(y_test, rf_preds):.4f}")

print("\nReporte de Clasificación Random Forest:")
print(classification_report(y_test, rf_preds))

