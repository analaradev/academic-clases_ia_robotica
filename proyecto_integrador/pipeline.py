import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# parametros generales
TEST_SIZE = 0.20
SEED = 42


# 1. Carga del dataset
datos = fetch_california_housing(as_frame=True)
df = datos.frame
df.to_csv("california_housing.csv", index=False)


# 2. Identificacion de X e y
X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]


# 3. Division del conjunto de datos (shuffle + split manual)
rng = np.random.default_rng(seed=SEED)
indices = rng.permutation(len(df))
X = X.iloc[indices].reset_index(drop=True)
y = y.iloc[indices].reset_index(drop=True)

n_total = len(df)
n_test = int(np.floor(n_total * TEST_SIZE))
n_train = n_total - n_test

X_train = X.iloc[:n_train].reset_index(drop=True)
X_test = X.iloc[n_train:].reset_index(drop=True)
y_train = y.iloc[:n_train].reset_index(drop=True)
y_test = y.iloc[n_train:].reset_index(drop=True)


# 4. Preprocesamiento
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 5. Entrenamiento del modelo
modelo = RandomForestRegressor(n_estimators=100, random_state=SEED)
modelo.fit(X_train_scaled, y_train)


# 6. Evaluacion con metricas adecuadas
pred = modelo.predict(X_test_scaled)

mae = mean_absolute_error(y_test, pred)
mse = mean_squared_error(y_test, pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, pred)

print("=" * 40)
print(f"Total de registros: {n_total}")
print(f"Registros en entrenamiento: {n_train}")
print(f"Registros en prueba: {n_test}")
print("-" * 40)
print(f"MAE:  {mae:.4f}")
print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2:   {r2:.4f}")
print("=" * 40)


# 7. Graficas y resultados
# grafica 1: reales vs predicciones
plt.figure(figsize=(6, 6))
plt.scatter(y_test, pred, alpha=0.4, edgecolors="k", linewidths=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
plt.xlabel("Valor real (MedHouseVal)")
plt.ylabel("Prediccion")
plt.title("Reales vs Predicciones")
plt.tight_layout()
plt.savefig("grafica_reales_vs_pred.png")
plt.close()

# grafica 2: histograma de errores
errores = y_test - pred
plt.figure(figsize=(6, 4))
plt.hist(errores, bins=50, edgecolor="black")
plt.xlabel("Error (real - prediccion)")
plt.ylabel("Frecuencia")
plt.title("Distribucion de errores")
plt.tight_layout()
plt.savefig("grafica_errores.png")
plt.close()

# guardar archivos de salida
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

resultados = pd.DataFrame({
    "real": y_test,
    "prediccion": pred,
    "error": errores
})
resultados.to_csv("resultados_prediccion.csv", index=False)

print("Archivos generados correctamente.")
