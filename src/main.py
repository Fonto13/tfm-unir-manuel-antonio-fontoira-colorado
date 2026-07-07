"""
TFM - Preprocesamiento, análisis exploratorio y modelado sobre PLAsTiCC
Autor: Manuel Antonio Fontoira Colorado

Descripción:
Este script implementa el flujo completo de trabajo para la clasificación automática de
supernovas a partir del dataset PLAsTiCC. Incluye:

1. Carga de curvas de luz y metadatos.
2. Ingeniería de variables agregadas, temporales, morfológicas y por banda fotométrica.
3. Creación de dataset_final.csv.
4. Análisis exploratorio de datos.
5. Entrenamiento y evaluación de Random Forest multiclase.
6. Reformulación binaria para detección de supernovas Ia.
7. Entrenamiento y evaluación de Random Forest, XGBoost y MLP.
8. Validación cruzada estratificada y ajuste de hiperparámetros para RF y MLP.
9. Comparación final de modelos mediante accuracy, AUC y F2-score.
"""

# ============================================================
# 0. IMPORTACIÓN DE LIBRERÍAS
# ============================================================

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    fbeta_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    learning_curve,
    train_test_split,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier


# ============================================================
# 1. CONFIGURACIÓN GENERAL
# ============================================================

LIGHTCURVES_PATH = "lightcurves.csv"
METADATA_PATH = "metadata.csv"
OUTPUT_DATASET_PATH = "dataset_final.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2
RF_N_ESTIMATORS = 100
THRESHOLD_BINARIO = 0.4
CLASE_IA = 90

# Variables predictoras base. A esta lista se añadirán automáticamente
# las variables generadas por banda fotométrica.
BASE_FEATURES = [
    "flux_mean",
    "flux_median",
    "flux_std",
    "flux_max",
    "flux_min",
    "flux_iqr",
    "flux_skew",
    "flux_kurtosis",
    "flux_amplitude",
    "flux_err_mean",
    "n_detected",
    "n_observations",
    "mjd_duration",
    "time_to_peak",
    "hostgal_photoz",
    "hostgal_photoz_err",
    "distmod",
    "mwebv",
]


# ============================================================
# 2. FUNCIONES AUXILIARES
# ============================================================

def evaluar_modelo_binario(nombre_modelo, y_true, y_proba, threshold=0.4):
    """
    Evalúa un modelo binario a partir de sus probabilidades de clase positiva.
    Devuelve las predicciones, accuracy, AUC y F2-score.
    """
    y_pred = (y_proba >= threshold).astype(int)
    accuracy = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)
    f2 = fbeta_score(y_true, y_pred, beta=2)

    print(f"\n===== {nombre_modelo} | threshold={threshold} =====")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC ROC: {auc:.4f}")
    print(f"F2-score: {f2:.4f}")
    print("\nClassification report:")
    print(classification_report(y_true, y_pred))
    print("Matriz de confusión:")
    print(confusion_matrix(y_true, y_pred))

    return y_pred, accuracy, auc, f2


def dibujar_curva_roc(nombre_modelo, y_true, y_proba):
    """Dibuja la curva ROC de un modelo binario."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"{nombre_modelo} (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Clasificador aleatorio")
    plt.title(f"Curva ROC - {nombre_modelo}")
    plt.xlabel("Tasa de falsos positivos")
    plt.ylabel("Tasa de verdaderos positivos")
    plt.legend()
    plt.tight_layout()
    plt.show()


def dibujar_curva_precision_recall(nombre_modelo, y_true, y_proba):
    """Dibuja la curva Precision-Recall de un modelo binario."""
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    avg_precision = average_precision_score(y_true, y_proba)

    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, label=f"{nombre_modelo} (AP = {avg_precision:.3f})")
    plt.title(f"Curva Precision-Recall - {nombre_modelo}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tight_layout()
    plt.show()


def dibujar_curva_aprendizaje(nombre_modelo, estimator, X_train, y_train, cv):
    """Dibuja la curva de aprendizaje usando F1-score como métrica."""
    train_sizes, train_scores, test_scores = learning_curve(
        estimator,
        X_train,
        y_train,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5),
    )

    plt.figure(figsize=(7, 6))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), label="Entrenamiento")
    plt.plot(train_sizes, np.mean(test_scores, axis=1), label="Validación")
    plt.title(f"Curva de aprendizaje - {nombre_modelo}")
    plt.xlabel("Número de muestras de entrenamiento")
    plt.ylabel("F1-score")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# 3. CARGA DE ARCHIVOS ORIGINALES
# ============================================================

print("Cargando archivos...")
lightcurves = pd.read_csv(LIGHTCURVES_PATH)
metadata = pd.read_csv(METADATA_PATH)
print("Lightcurves cargado con éxito.")
print("Metadata cargado con éxito.")


# ============================================================
# 4. PREPROCESAMIENTO E INGENIERÍA DE VARIABLES
# ============================================================

print("\nPreprocesando archivos y generando variables...")

# ------------------------------------------------------------
# 4.1. Variables agregadas globales por objeto
# ------------------------------------------------------------

features = lightcurves.groupby("object_id").agg(
    flux_mean=("flux", "mean"),
    flux_median=("flux", "median"),
    flux_std=("flux", "std"),
    flux_max=("flux", "max"),
    flux_min=("flux", "min"),
    flux_iqr=("flux", lambda x: x.quantile(0.75) - x.quantile(0.25)),
    flux_skew=("flux", "skew"),
    flux_kurtosis=("flux", "kurt"),
    flux_err_mean=("flux_err", "mean"),
    n_detected=("detected_bool", "sum"),
    n_observations=("detected_bool", "count"),
    mjd_min=("mjd", "min"),
    mjd_max=("mjd", "max"),
).reset_index()

features["flux_amplitude"] = features["flux_max"] - features["flux_min"]
features["mjd_duration"] = features["mjd_max"] - features["mjd_min"]
features = features.drop(columns=["mjd_min", "mjd_max"])

# ------------------------------------------------------------
# 4.2. Variables temporales: tiempo desde primera observación hasta máximo flujo
# ------------------------------------------------------------

idx_peak = lightcurves.groupby("object_id")["flux"].idxmax()
peak_info = lightcurves.loc[idx_peak, ["object_id", "mjd"]].rename(
    columns={"mjd": "mjd_peak"}
)

mjd_first = lightcurves.groupby("object_id")["mjd"].min().reset_index()
mjd_first = mjd_first.rename(columns={"mjd": "mjd_first"})

peak_info = peak_info.merge(mjd_first, on="object_id", how="left")
peak_info["time_to_peak"] = peak_info["mjd_peak"] - peak_info["mjd_first"]

features = features.merge(
    peak_info[["object_id", "time_to_peak"]],
    on="object_id",
    how="left",
)

# ------------------------------------------------------------
# 4.3. Variables por banda fotométrica
# ------------------------------------------------------------

passband_features = lightcurves.groupby(["object_id", "passband"]).agg(
    flux_mean_pb=("flux", "mean"),
    flux_std_pb=("flux", "std"),
    flux_max_pb=("flux", "max"),
    flux_min_pb=("flux", "min"),
).reset_index()

passband_features = passband_features.pivot(
    index="object_id",
    columns="passband",
)

passband_features.columns = [
    f"{stat}_{pb}" for stat, pb in passband_features.columns
]
passband_features = passband_features.reset_index()

features = features.merge(passband_features, on="object_id", how="left")

# Ahora sí se incorporan las variables por banda a la lista de predictores.
passband_columns = [
    col for col in features.columns
    if col.startswith("flux_mean_pb_")
    or col.startswith("flux_std_pb_")
    or col.startswith("flux_max_pb_")
    or col.startswith("flux_min_pb_")
]

SELECTED_FEATURES = BASE_FEATURES + passband_columns

print(f"Número de variables predictoras generadas: {len(SELECTED_FEATURES)}")


# ============================================================
# 5. UNIÓN CON METADATOS Y CREACIÓN DEL DATASET FINAL
# ============================================================

data_final = features.merge(metadata, on="object_id", how="inner")

selected_columns = SELECTED_FEATURES + ["target"]
data_selected = data_final[selected_columns].copy()

# Imputación sencilla de valores nulos usando la mediana de cada variable numérica.
data_selected = data_selected.fillna(data_selected.median(numeric_only=True))

data_selected.to_csv(OUTPUT_DATASET_PATH, index=False)

print("\nPreprocesamiento completo. Dataset final creado con éxito.")
print(f"Archivo guardado como: {OUTPUT_DATASET_PATH}")
print(f"Dimensiones dataset final: {data_selected.shape}")


# ============================================================
# 6. CARGA DEL DATASET FINAL Y SEPARACIÓN X / y
# ============================================================

dataset = pd.read_csv(OUTPUT_DATASET_PATH)

X = dataset[SELECTED_FEATURES]
y = dataset["target"]

dataset["is_Ia"] = (dataset["target"] == CLASE_IA).astype(int)

print("\nPrimeras filas del dataset final:")
print(dataset.head())
print("\nVariables predictoras seleccionadas:")
print(SELECTED_FEATURES)


# ============================================================
# 7. ANÁLISIS EXPLORATORIO DE DATOS
# ============================================================

# ------------------------------------------------------------
# 7.1. Distribución multiclase
# ------------------------------------------------------------

target_counts = dataset["target"].value_counts().sort_index()

plt.figure(figsize=(10, 5))
target_counts.plot(kind="bar")
plt.title("Distribución de clases en el dataset final")
plt.xlabel("Clase target")
plt.ylabel("Número de objetos")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 7.2. Distribución binaria
# ------------------------------------------------------------

binary_counts = dataset["is_Ia"].value_counts().sort_index()
binary_counts.index = ["No Ia", "Ia"]

plt.figure(figsize=(6, 5))
binary_counts.plot(kind="bar")
plt.title("Distribución binaria: supernovas Ia frente al resto")
plt.xlabel("Clase")
plt.ylabel("Número de objetos")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 7.3. Histogramas de variables principales
# ------------------------------------------------------------

variables_hist = [
    "flux_mean",
    "flux_std",
    "flux_amplitude",
    "mjd_duration",
    "time_to_peak",
    "distmod",
    "hostgal_photoz",
]

for var in variables_hist:
    plt.figure(figsize=(7, 5))

    upper_limit = dataset[var].quantile(0.99)
    lower_limit = dataset[var].quantile(0.01)

    ia_values = dataset[
        (dataset["is_Ia"] == 1)
        & (dataset[var] >= lower_limit)
        & (dataset[var] <= upper_limit)
    ][var]

    no_ia_values = dataset[
        (dataset["is_Ia"] == 0)
        & (dataset[var] >= lower_limit)
        & (dataset[var] <= upper_limit)
    ][var]

    plt.hist(no_ia_values, bins=40, alpha=0.6, label="No Ia")
    plt.hist(ia_values, bins=40, alpha=0.6, label="Ia")
    plt.title(f"Distribución de {var}")
    plt.xlabel(var)
    plt.ylabel("Frecuencia")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------
# 7.4. Matriz de correlación
# ------------------------------------------------------------

corr_vars = [
    "flux_mean",
    "flux_median",
    "flux_std",
    "flux_max",
    "flux_min",
    "flux_iqr",
    "flux_skew",
    "flux_kurtosis",
    "flux_amplitude",
    "flux_err_mean",
    "n_detected",
    "n_observations",
    "mjd_duration",
    "time_to_peak",
    "hostgal_photoz",
    "hostgal_photoz_err",
    "distmod",
    "mwebv"
]

corr_matrix = dataset[corr_vars].corr()

plt.figure(figsize=(10,8))

plt.imshow(corr_matrix, aspect="auto")

plt.colorbar(label="Correlación")

plt.xticks(
    range(len(corr_vars)),
    corr_vars,
    rotation=90,
    fontsize=9
)

plt.yticks(
    range(len(corr_vars)),
    corr_vars,
    fontsize=9
)

plt.title("Matriz de correlación de variables globales y temporales")

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 7.5. Boxplots de variables principales por clase
# ------------------------------------------------------------

variables_box = [
    "flux_mean",
    "flux_std",
    "flux_amplitude",
    "mjd_duration",
    "time_to_peak",
    "distmod",
    "hostgal_photoz",
]

for var in variables_box:
    plt.figure(figsize=(7, 5))
    dataset.boxplot(column=var, by="is_Ia")
    plt.title(f"Boxplot de {var} por clase")
    plt.suptitle("")
    plt.xlabel("Clase (0 = No Ia, 1 = Ia)")
    plt.ylabel(var)
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------
# 7.6. Mapa de valores nulos
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))
plt.imshow(dataset.isnull(), aspect="auto", cmap="viridis")
plt.colorbar(label="Valor nulo")
plt.title("Mapa de calor de valores nulos en dataset_final")
plt.xlabel("Columnas")
plt.ylabel("Filas")
plt.xticks(range(len(dataset.columns)), dataset.columns, rotation=90)
plt.tight_layout()
plt.show()


# ============================================================
# 8. MODELO MULTICLASE: RANDOM FOREST
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

print("\nTrain multiclase:", X_train.shape)
print("Test multiclase:", X_test.shape)

rf_multiclass = RandomForestClassifier(
    n_estimators=RF_N_ESTIMATORS,
    random_state=RANDOM_STATE,
)

rf_multiclass.fit(X_train, y_train)
y_pred_multiclass = rf_multiclass.predict(X_test)

print("\n===== Random Forest multiclase =====")
print("Accuracy:", accuracy_score(y_test, y_pred_multiclass))
print("\nClassification report:")
print(classification_report(y_test, y_pred_multiclass))


# ============================================================
# 9. REFORMULACIÓN BINARIA: SUPERNOVAS IA VS RESTO
# ============================================================

y_binary = dataset["is_Ia"]
X_binary = dataset[SELECTED_FEATURES]

print("\nDistribución de clases en el problema binario:")
print(y_binary.value_counts())
print("\nProporciones:")
print(y_binary.value_counts(normalize=True))

X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
    X_binary,
    y_binary,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_binary,
)

print("\nTrain binario:", X_train_bin.shape)
print("Test binario:", X_test_bin.shape)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


# ============================================================
# 10. MODELO BINARIO: RANDOM FOREST BÁSICO
# ============================================================

rf_binary = RandomForestClassifier(
    n_estimators=RF_N_ESTIMATORS,
    random_state=RANDOM_STATE,
)

time_start = time.time()
rf_binary.fit(X_train_bin, y_train_bin)
time_rf_basic = time.time() - time_start

print(f"\nTiempo de entrenamiento Random Forest binario: {time_rf_basic:.2f} segundos")

y_proba_rf_basic = rf_binary.predict_proba(X_test_bin)[:, 1]
y_pred_rf_basic, acc_rf_basic, auc_rf_basic, f2_rf_basic = evaluar_modelo_binario(
    "Random Forest básico",
    y_test_bin,
    y_proba_rf_basic,
    THRESHOLD_BINARIO,
)

dibujar_curva_roc("Random Forest básico", y_test_bin, y_proba_rf_basic)
dibujar_curva_precision_recall("Random Forest básico", y_test_bin, y_proba_rf_basic)

# Importancia de variables del Random Forest básico.
feature_importance = pd.DataFrame({
    "feature": SELECTED_FEATURES,
    "importance": rf_binary.feature_importances_,
}).sort_values(by="importance", ascending=True)

plt.figure(figsize=(9, 8))
plt.barh(feature_importance["feature"], feature_importance["importance"])
plt.title("Importancia de variables - Random Forest binario")
plt.xlabel("Importancia")
plt.ylabel("Variable")
plt.tight_layout()
plt.show()


# ============================================================
# 11. MODELO BINARIO: RANDOM FOREST OPTIMIZADO
# ============================================================

rf_model = RandomForestClassifier(
    random_state=RANDOM_STATE,
    class_weight="balanced",
)

param_grid_rf = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
}

grid_rf = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid_rf,
    scoring="f1",
    cv=cv,
    n_jobs=-1,
    verbose=1,
)

time_start = time.time()
grid_rf.fit(X_train_bin, y_train_bin)
time_rf_opt = time.time() - time_start

best_rf = grid_rf.best_estimator_
cv_scores_rf = grid_rf.cv_results_["mean_test_score"]

print(f"\nTiempo de ejecución GridSearchCV RF binario: {time_rf_opt:.2f} segundos")
print("Mejores parámetros RF:", grid_rf.best_params_)
print(f"Mejor F1 medio CV RF: {grid_rf.best_score_:.4f}")

# Media y desviación del mejor modelo por folds.
best_index_rf = grid_rf.best_index_
print(
    "F1 CV RF optimizado: "
    f"{grid_rf.cv_results_['mean_test_score'][best_index_rf]:.4f} ± "
    f"{grid_rf.cv_results_['std_test_score'][best_index_rf]:.4f}"
)

y_proba_rf_opt = best_rf.predict_proba(X_test_bin)[:, 1]
y_pred_rf_opt, acc_rf_opt, auc_rf_opt, f2_rf_opt = evaluar_modelo_binario(
    "Random Forest optimizado",
    y_test_bin,
    y_proba_rf_opt,
    THRESHOLD_BINARIO,
)

dibujar_curva_roc("Random Forest optimizado", y_test_bin, y_proba_rf_opt)
dibujar_curva_precision_recall("Random Forest optimizado", y_test_bin, y_proba_rf_opt)
dibujar_curva_aprendizaje("Random Forest optimizado", best_rf, X_train_bin, y_train_bin, cv)


# ============================================================
# 12. MODELO BINARIO: XGBOOST
# ============================================================

xgb_binary = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=RANDOM_STATE,
)

time_start = time.time()
xgb_binary.fit(X_train_bin, y_train_bin)
time_xgb = time.time() - time_start

print(f"\nTiempo de entrenamiento XGBoost binario: {time_xgb:.2f} segundos")

y_proba_xgb = xgb_binary.predict_proba(X_test_bin)[:, 1]
y_pred_xgb, acc_xgb, auc_xgb, f2_xgb = evaluar_modelo_binario(
    "XGBoost binario",
    y_test_bin,
    y_proba_xgb,
    THRESHOLD_BINARIO,
)

dibujar_curva_roc("XGBoost binario", y_test_bin, y_proba_xgb)
dibujar_curva_precision_recall("XGBoost binario", y_test_bin, y_proba_xgb)

# Importancia de variables de XGBoost.
xgb_importance = pd.DataFrame({
    "feature": SELECTED_FEATURES,
    "importance": xgb_binary.feature_importances_,
}).sort_values(by="importance", ascending=True)

plt.figure(figsize=(9, 8))
plt.barh(xgb_importance["feature"], xgb_importance["importance"])
plt.title("Importancia de variables - XGBoost binario")
plt.xlabel("Importancia")
plt.ylabel("Variable")
plt.tight_layout()
plt.show()


# ============================================================
# 13. MODELO BINARIO: RED NEURONAL MLP
# ============================================================

# ------------------------------------------------------------
# 13.1. Escalado de variables
# ------------------------------------------------------------

scaler = StandardScaler()
X_train_bin_scaled = scaler.fit_transform(X_train_bin)
X_test_bin_scaled = scaler.transform(X_test_bin)

print("\nEscalado completado para MLP.")
print("Shape X_train_bin_scaled:", X_train_bin_scaled.shape)
print("Shape X_test_bin_scaled:", X_test_bin_scaled.shape)

# ------------------------------------------------------------
# 13.2. MLP básico
# ------------------------------------------------------------

mlp_basic = MLPClassifier(
    hidden_layer_sizes=(50,),
    activation="relu",
    solver="adam",
    max_iter=500,
    random_state=RANDOM_STATE,
)

time_start = time.time()
mlp_basic.fit(X_train_bin_scaled, y_train_bin)
time_mlp_basic = time.time() - time_start

print(f"\nTiempo de entrenamiento MLP básico: {time_mlp_basic:.2f} segundos")

y_proba_mlp_basic = mlp_basic.predict_proba(X_test_bin_scaled)[:, 1]

# Evaluación con threshold estándar 0,5.
y_pred_mlp_05, acc_mlp_05, auc_mlp_basic, f2_mlp_05 = evaluar_modelo_binario(
    "MLP básico",
    y_test_bin,
    y_proba_mlp_basic,
    threshold=0.5,
)

# Evaluación con threshold 0,4.
y_pred_mlp_04, acc_mlp_04, auc_mlp_04, f2_mlp_04 = evaluar_modelo_binario(
    "MLP básico",
    y_test_bin,
    y_proba_mlp_basic,
    threshold=THRESHOLD_BINARIO,
)

dibujar_curva_roc("MLP básico", y_test_bin, y_proba_mlp_basic)
dibujar_curva_precision_recall("MLP básico", y_test_bin, y_proba_mlp_basic)

# ------------------------------------------------------------
# 13.3. MLP optimizado
# ------------------------------------------------------------

mlp_model = MLPClassifier(
    random_state=RANDOM_STATE,
    max_iter=1000,
    early_stopping=True,
)

param_grid_mlp = {
    "hidden_layer_sizes": [(50,), (100,), (50, 50)],
    "activation": ["relu"],
    "solver": ["adam"],
    "alpha": [0.0001, 0.001],
    "learning_rate_init": [0.001, 0.01],
}

grid_mlp = GridSearchCV(
    estimator=mlp_model,
    param_grid=param_grid_mlp,
    scoring="f1",
    cv=cv,
    n_jobs=-1,
    verbose=1,
)

time_start = time.time()
grid_mlp.fit(X_train_bin_scaled, y_train_bin)
time_mlp_opt = time.time() - time_start

best_mlp = grid_mlp.best_estimator_

print(f"\nTiempo de ejecución GridSearchCV MLP binario: {time_mlp_opt:.2f} segundos")
print("Mejores parámetros MLP:", grid_mlp.best_params_)
print(f"Mejor F1 medio CV MLP: {grid_mlp.best_score_:.4f}")

best_index_mlp = grid_mlp.best_index_
print(
    "F1 CV MLP optimizado: "
    f"{grid_mlp.cv_results_['mean_test_score'][best_index_mlp]:.4f} ± "
    f"{grid_mlp.cv_results_['std_test_score'][best_index_mlp]:.4f}"
)

y_proba_mlp_opt = best_mlp.predict_proba(X_test_bin_scaled)[:, 1]
y_pred_mlp_opt, acc_mlp_opt, auc_mlp_opt, f2_mlp_opt = evaluar_modelo_binario(
    "MLP optimizado",
    y_test_bin,
    y_proba_mlp_opt,
    THRESHOLD_BINARIO,
)

dibujar_curva_roc("MLP optimizado", y_test_bin, y_proba_mlp_opt)
dibujar_curva_precision_recall("MLP optimizado", y_test_bin, y_proba_mlp_opt)
dibujar_curva_aprendizaje("MLP optimizado", best_mlp, X_train_bin_scaled, y_train_bin, cv)


# ============================================================
# 14. COMPARACIÓN FINAL DE MODELOS
# ============================================================

comparativa_modelos = pd.DataFrame({
    "Modelo": [
        "Random Forest básico",
        "Random Forest optimizado",
        "XGBoost binario",
        "MLP básico threshold 0.5",
        "MLP básico threshold 0.4",
        "MLP optimizado",
    ],
    "Accuracy": [
        acc_rf_basic,
        acc_rf_opt,
        acc_xgb,
        acc_mlp_05,
        acc_mlp_04,
        acc_mlp_opt,
    ],
    "AUC ROC": [
        auc_rf_basic,
        auc_rf_opt,
        auc_xgb,
        auc_mlp_basic,
        auc_mlp_04,
        auc_mlp_opt,
    ],
    "F2-score": [
        f2_rf_basic,
        f2_rf_opt,
        f2_xgb,
        f2_mlp_05,
        f2_mlp_04,
        f2_mlp_opt,
    ],
    "Tiempo entrenamiento / ajuste (s)": [
        time_rf_basic,
        time_rf_opt,
        time_xgb,
        time_mlp_basic,
        time_mlp_basic,
        time_mlp_opt,
    ],
})

print("\n===== Comparativa final de modelos =====")
print(comparativa_modelos)

comparativa_modelos.to_csv("comparativa_modelos.csv", index=False)
print("\nComparativa guardada como: comparativa_modelos.csv")

# Gráfico comparativo de métricas principales.
metricas_plot = comparativa_modelos.set_index("Modelo")[["Accuracy", "AUC ROC", "F2-score"]]
metricas_plot.plot(kind="bar", figsize=(12, 6))
plt.title("Comparación de métricas principales entre modelos binarios")
plt.ylabel("Valor")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

print("\nEjecución finalizada correctamente.")
