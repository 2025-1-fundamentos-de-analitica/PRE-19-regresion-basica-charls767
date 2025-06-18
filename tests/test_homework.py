"""Autograding script (versión mejorada y correcta)."""

import pickle
import pandas as pd
from sklearn.metrics import mean_squared_error

def test_01():
    """Test que valida la predicción de MPG en datos de prueba no vistos."""

    # Cargar dataset
    dataset = pd.read_csv("files/input/auto_mpg.csv")
    dataset = dataset.dropna()

    # Mapear columna 'Origin' y generar variables dummy
    dataset["Origin"] = dataset["Origin"].map({1: "USA", 2: "Europe", 3: "Japan"})
    dataset = pd.get_dummies(dataset, columns=["Origin"], prefix="", prefix_sep="")

    # Separar en train y test (mismo random_state que notebook)
    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    # Separar variables predictoras y target
    y_test = test_dataset.pop("MPG")

    # Cargar modelo entrenado y scaler
    try:
        with open("files/mlp.pickle", "rb") as file:
            mlp = pickle.load(file)
        with open("files/features_scaler.pickle", "rb") as file:
            features_scaler = pickle.load(file)
    except FileNotFoundError as e:
        raise AssertionError(f"Archivo faltante: {e.filename}")

    # Validar que las columnas coincidan
    if test_dataset.shape[1] != features_scaler.mean_.shape[0]:
        raise AssertionError("Columnas incompatibles con el scaler.")

    # Estandarizar y predecir
    X_test_scaled = features_scaler.transform(test_dataset)
    y_pred = mlp.predict(X_test_scaled)

    # Evaluar desempeño
    mse = mean_squared_error(y_true=y_test, y_pred=y_pred)
    print(f"Mean Squared Error (MSE) en test: {mse:.4f}")

    # Validar umbral
    assert mse < 7.745, f"El MSE es muy alto: {mse:.4f}"


# Sera ?
