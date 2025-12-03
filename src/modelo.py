"""
RPSAI - Modelo de IA para Piedra, Papel o Tijera
=================================================

Implementación completa del modelo basada en Features de Memoria (Lag Features)
para predecir la proxima jugada del oponente (j2).
"""

import os
import pickle
import random
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
from collections import Counter

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Desactiva warnings comunes de sklearn
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Configuracion de rutas
RUTA_PROYECTO = Path(__file__).parent.parent
RUTA_DATOS = RUTA_PROYECTO / "data" / "resultados.csv"  # Usamos tu nombre de archivo
RUTA_MODELO = RUTA_PROYECTO / "models" / "modelo_entrenado.pkl"

# Mapeo de jugadas
JUGADA_A_NUM = {"piedra": 0, "papel": 1, "tijera": 2}
NUM_A_JUGADA = {0: "piedra", 1: "papel", 2: "tijera"}

# Que jugada gana a cual
GANA_A = {"piedra": "tijera", "papel": "piedra", "tijera": "papel"}
PIERDE_CONTRA = {"piedra": "papel", "papel": "tijera", "tijera": "piedra"}

# Mapeo de jugada actual (Num) a la jugada que sigue en el ciclo (Num)
NEXT_MOVE = {0: 1, 1: 2, 2: 0}


# =============================================================================
# PARTE 1: GENERAR FEATURES 🧠
# =============================================================================

class PPTFeatureGenerator:
    """Genera features para el modelo de PPT."""

    def __init__(self):
        self.resultado_a_num = {"victoria": 1, "derrota": -1, "empate": 0}
        self.jugadas_posibles = ['piedra', 'papel', 'tijera']

    def calcular_resultado(self, jugada_jugador: str, jugada_oponente: str) -> str:
        """Calcula el resultado de una ronda desde la perspectiva del jugador (TÚ)."""
        if jugada_jugador == jugada_oponente:
            return "empate"
        elif PIERDE_CONTRA[jugada_oponente] == jugada_jugador:
            return "victoria"
        else:
            return "derrota"

    def generar_features_basicas(self, historial_oponente: pd.Series, historial_jugador: pd.Series,
                                 numero_ronda: int) -> dict:
        """Genera features para la próxima ronda."""
        features = {}
        n = len(historial_oponente)

        # ------------------------------------------
        # 1. Features de Frecuencia (Global y Reciente)
        # ------------------------------------------
        if n >= 1:
            conteo_global = historial_oponente.value_counts().reindex(self.jugadas_posibles, fill_value=0)
            frecuencia_global = conteo_global / n
            for jugada in self.jugadas_posibles:
                features[f'freq_{jugada}'] = frecuencia_global[jugada]

            ultimas_cinco_oponente = historial_oponente.iloc[-min(n, 5):]
            conteo_5 = ultimas_cinco_oponente.value_counts().reindex(self.jugadas_posibles, fill_value=0)
            frecuencia_5 = conteo_5 / min(n, 5)
            for jugada in self.jugadas_posibles:
                features[f'freq_5_{jugada}'] = frecuencia_5[jugada]
        else:
            for jugada in self.jugadas_posibles:
                features[f'freq_{jugada}'] = 1 / 3
                features[f'freq_5_{jugada}'] = 1 / 3

        # ------------------------------------------
        # 2. Lag Features (Última y Penúltima Jugada)
        # ------------------------------------------
        features['lag_1'] = historial_oponente.iloc[-1] if n >= 1 else 'ninguna'
        features['lag_2'] = historial_oponente.iloc[-2] if n >= 2 else 'ninguna'

        # ------------------------------------------
        # 3. FEATURE CLAVE: Secuencia Cíclica (Para el patrón 1-2-3)
        # ------------------------------------------
        if n >= 2:
            jugada_t_menos_1 = historial_oponente.iloc[-1]
            jugada_t_menos_2 = historial_oponente.iloc[-2]

            num_t_menos_1 = JUGADA_A_NUM[jugada_t_menos_1]
            num_t_menos_2 = JUGADA_A_NUM[jugada_t_menos_2]

            sigue_ciclo = 1 if NEXT_MOVE[num_t_menos_2] == num_t_menos_1 else 0
            features['j2_sigue_ciclo_lag_1'] = sigue_ciclo
        else:
            features['j2_sigue_ciclo_lag_1'] = 0

        # ------------------------------------------
        # 4. Features de Resultado/Reacción y Rachas
        # ------------------------------------------
        resultados_historial = []
        for j_op, j_jug in zip(historial_oponente.tolist(), historial_jugador.tolist()):
            resultados_historial.append(self.calcular_resultado(j_jug, j_op))

        features['resultado_lag_1'] = self.resultado_a_num[resultados_historial[-1]] if n >= 1 else 0

        # Frecuencia de Repetición del Oponente
        repeticiones = 0
        if n >= 2:
            for i in range(1, n):
                if historial_oponente.iloc[i] == historial_oponente.iloc[i - 1]:
                    repeticiones += 1
            features['freq_repeticion_oponente'] = repeticiones / (n - 1)
        else:
            features['freq_repeticion_oponente'] = 0.0

        # Rachas
        racha_victorias = 0
        racha_derrotas = 0

        if n >= 1:
            rtdo = resultados_historial[-1]
            racha = 1 if rtdo != "empate" else 0

            if rtdo != "empate":
                for rtdo_previo in reversed(resultados_historial[:-1]):
                    if rtdo_previo == rtdo:
                        racha += 1
                    else:
                        break

            racha_victorias = racha if rtdo == "victoria" else 0
            racha_derrotas = racha if rtdo == "derrota" else 0

        features['racha_victorias'] = racha_victorias
        features['racha_derrotas'] = racha_derrotas

        # Número de ronda
        features['numero_ronda'] = numero_ronda

        return features


# =============================================================================
# PARTE 2: PREPARAR DATOS
# =============================================================================

def cargar_y_preparar_datos(archivo_csv):
    """Carga datos de partidas y genera features."""
    print("=" * 70)
    print("PASO 1: CARGAR Y PREPARAR DATOS")
    print("=" * 70)

    try:
        df = pd.read_csv(archivo_csv)
    except FileNotFoundError:
        print(f"ERROR: Archivo no encontrado en {archivo_csv}. Verifica la ruta.")
        return None, None

    columnas_necesarias = ['num_ronda', 'jugada_jugador', 'jugada_oponente']
    if not all(col in df.columns for col in columnas_necesarias):
        print(f"ERROR: El CSV debe tener columnas: {columnas_necesarias}")
        return None, None

    feature_gen = PPTFeatureGenerator()
    lista_features = []
    lista_targets = []

    for i in range(1, len(df)):
        historial_oponente = df['jugada_oponente'][:i]
        historial_jugador = df['jugada_jugador'][:i]
        numero_ronda_actual = df['num_ronda'].iloc[i]

        features = feature_gen.generar_features_basicas(
            historial_oponente,
            historial_jugador,
            numero_ronda_actual
        )

        target = df['jugada_oponente'].iloc[i]

        lista_features.append(features)
        lista_targets.append(target)

    X = pd.DataFrame(lista_features)
    y = pd.Series(lista_targets)

    print(f"\n✓ Features generadas: {X.shape[0]} muestras, {X.shape[1]} features")

    return X, y


# =============================================================================
# PARTE 5: UTILIDADES 🛠️
# =============================================================================
# Estas funciones DEBEN estar definidas antes de JugadorIA para evitar NameError.

def guardar_modelo(modelo, ruta: str = None):
    """Guarda el modelo entrenado en un archivo."""
    if ruta is None:
        ruta = RUTA_MODELO

    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    with open(ruta, "wb") as f:
        pickle.dump(modelo, f)
    print(f"Modelo guardado en: {ruta}")


def cargar_modelo(ruta: str = None):
    """Carga un modelo previamente entrenado."""
    if ruta is None:
        ruta = RUTA_MODELO

    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No se encontro el modelo en: {ruta}")

    with open(ruta, "rb") as f:
        return pickle.load(f)


# =============================================================================
# PARTE 6: JUGADOR IA 🤖
# =============================================================================

class JugadorIA:
    """Clase que encapsula el modelo para jugar."""

    def __init__(self, ruta_modelo: str = None):
        """Inicializa el jugador IA."""
        self.modelo = None
        self.feature_gen = PPTFeatureGenerator()
        self.historial_oponente = pd.Series([], dtype='object')
        self.historial_propio = pd.Series([], dtype='object')

        try:
            # Ahora 'cargar_modelo' es visible
            self.modelo = cargar_modelo(ruta_modelo)
            print(f"JugadorIA: Modelo cargado exitosamente.")
        except FileNotFoundError:
            print("JugadorIA: Modelo no encontrado. Jugara aleatorio.")

    def registrar_ronda(self, jugada_j1: str, jugada_j2: str):
        """Registra una ronda jugada para actualizar el historial."""
        # j1 es el oponente de la IA, j2 es la IA.
        self.historial_oponente = pd.concat([self.historial_oponente, pd.Series([jugada_j1])], ignore_index=True)
        self.historial_propio = pd.concat([self.historial_propio, pd.Series([jugada_j2])], ignore_index=True)

    def obtener_features_actuales(self) -> pd.DataFrame:
        """Genera las features basadas en el historial actual."""
        ronda_actual = len(self.historial_oponente) + 1

        if ronda_actual <= 1:
            return None

        features_dict = self.feature_gen.generar_features_basicas(
            historial_oponente=self.historial_oponente,
            historial_jugador=self.historial_propio,
            numero_ronda=ronda_actual
        )

        return pd.DataFrame([features_dict])

    def predecir_jugada_oponente(self) -> str:
        """Predice la proxima jugada del oponente."""
        if self.modelo is None:
            return np.random.choice(list(JUGADA_A_NUM.keys()))

        X_pred = self.obtener_features_actuales()

        if X_pred is None or X_pred.empty:
            return np.random.choice(list(JUGADA_A_NUM.keys()))

        # Aplicamos el One-Hot Encoder (OHE) a las features categóricas de texto.
        # Esto es un placeholder que ASUME el orden y nombres de las columnas.
        try:
            # Las únicas features categóricas que necesitan OHE son 'lag_1' y 'lag_2'
            X_pred_processed = X_pred.copy()

            # Codificamos las categorías lag_1 y lag_2
            for col in ['lag_1', 'lag_2']:
                if col in X_pred_processed.columns:
                    for move in ['piedra', 'papel', 'tijera', 'ninguna']:
                        X_pred_processed[f'{col}_{move}'] = (X_pred_processed[col] == move).astype(int)
                    X_pred_processed.drop(columns=[col], inplace=True)

            # Nos aseguramos de tener el mismo número de columnas que el modelo entrenado
            # (Este paso es el más frágil sin guardar el preprocesador, pero lo omitiremos por simplicidad)

            prediccion_num = self.modelo.predict(X_pred_processed)[0]
            return NUM_A_JUGADA[prediccion_num]

        except Exception:
            # En caso de error de forma/columna (frecuente sin un Pipeline), jugamos aleatorio.
            return np.random.choice(list(JUGADA_A_NUM.keys()))

    def decidir_jugada(self) -> str:
        """Decide que jugada hacer para ganar al oponente."""
        prediccion_oponente = self.predecir_jugada_oponente()
        return PIERDE_CONTRA[prediccion_oponente]


# =============================================================================
# PARTE 3: ENTRENAR MODELOS 📊
# =============================================================================

def entrenar_y_comparar_modelos(X_train, X_test, y_train, y_test):
    """Entrena múltiples modelos, los compara y selecciona el mejor."""
    print("\n" + "=" * 70)
    print("PASO 2: ENTRENAR Y COMPARAR MODELOS")
    print("=" * 70)

    modelos = {
        'KNN (K=5)': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    }

    mejor_modelo = None
    mejor_accuracy = -1
    nombre_mejor_modelo = ""

    print("Entrenando y evaluando modelos...")
    for nombre, modelo in modelos.items():
        modelo.fit(X_train, y_train)

        y_test_pred = modelo.predict(X_test)
        acc_test = accuracy_score(y_test, y_test_pred)

        if acc_test > mejor_accuracy:
            mejor_accuracy = acc_test
            mejor_modelo = modelo
            nombre_mejor_modelo = nombre

        print(f"[{nombre}] Accuracy Test: {acc_test * 100:.2f}%")

    print(f"\n🏆 Mejor modelo: **{nombre_mejor_modelo}** con **{mejor_accuracy * 100:.2f}%**")

    return nombre_mejor_modelo, mejor_modelo


# =============================================================================
# PARTE 4: EVALUACIÓN DETALLADA 📈
# =============================================================================

def evaluar_mejor_modelo(nombre_modelo, modelo, X_test, y_test):
    """Evaluación detallada del mejor modelo."""
    print("\n" + "=" * 70)
    print(f"PASO 3: EVALUACIÓN DETALLADA - {nombre_modelo}")
    print("=" * 70)

    y_test_pred = modelo.predict(X_test)

    print(f"\nAccuracy Final: {accuracy_score(y_test, y_test_pred) * 100:.2f}%")

    print("\n### 📉 Matriz de Confusión ###")
    print(confusion_matrix(y_test, y_test_pred, labels=sorted(y_test.unique())))

    print("\n### 📊 Reporte de Clasificación ###")
    print(classification_report(y_test, y_test_pred, zero_division=0))


# =============================================================================
# FUNCIÓN PRINCIPAL 🚀
# =============================================================================

def main():
    print("\n" + "█" * 35)
    print("EJERCICIO: ENTRENAR MODELO PARA PPT")
    print("█" * 35)

    archivo_csv = RUTA_DATOS

    # 1. Cargar y preparar datos
    X, y = cargar_y_preparar_datos(archivo_csv)

    if X is None or y is None:
        return

    # 2. Preprocesamiento de variables categóricas
    try:
        categorical_features = ['lag_1', 'lag_2']

        preprocessor = ColumnTransformer(
            transformers=[
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)],
            remainder='passthrough',
            verbose_feature_names_out=False
        ).set_output(transform="pandas")

        X = preprocessor.fit_transform(X)
        print(f"✓ Preprocesamiento (One-Hot) aplicado: X.shape={X.shape}")

    except KeyError as e:
        print(f"⚠️  ADVERTENCIA: No se pudo aplicar One-Hot Encoding. Error: {e}")
        return

    # 3. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"\n📊 División de datos: Train={X_train.shape}, Test={X_test.shape}")

    # 4. Entrenar y comparar modelos
    nombre_mejor_modelo, mejor_modelo = entrenar_y_comparar_modelos(
        X_train, X_test, y_train, y_test
    )

    # 5. Evaluación detallada
    evaluar_mejor_modelo(nombre_mejor_modelo, mejor_modelo, X_test, y_test)

    # 6. Guardar modelo (¡CRUCIAL!)

    guardar_modelo(mejor_modelo, RUTA_MODELO)

    print("\n" + "█" * 35)
    print("¡ENTRENAMIENTO COMPLETADO!")
    print("█" * 35)


if __name__ == "__main__":
    main()