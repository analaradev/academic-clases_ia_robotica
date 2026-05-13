import pandas as pd
import numpy as np

# leer datos
datos = pd.read_csv("regression_students_v1.csv")

# separar X e y
X = datos.drop(columns=["FinalExamScore"])
y = datos["FinalExamScore"]

# mezclar todo aleatoriamente
np.random.seed(42)
indices = np.random.permutation(len(datos))
X = X.iloc[indices].reset_index(drop=True)
y = y.iloc[indices].reset_index(drop=True)

# definir porcentaje de prueba (por ejemplo 0.20 = 20%)
test_size = 0.20

corte = int(len(datos) * (1 - test_size))

X_train = X.iloc[:corte]
X_test = X.iloc[corte:]
y_train = y.iloc[:corte]
y_test = y.iloc[corte:]

# mostrar resultados
print("Total de registros:", len(datos))
print("Registros en entrenamiento:", len(X_train))
print("Registros en prueba:", len(X_test))

# guardar en csv
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("Archivos guardados: X_train.csv, X_test.csv, y_train.csv, y_test.csv")
