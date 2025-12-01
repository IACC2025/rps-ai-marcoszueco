"""
EJERCICIO PARA ALUMNOS: Entrenar Modelo para Piedra, Papel o Tijera
====================================================================

OBJETIVO:
Aplicar TODO lo aprendido en esta clase para crear un modelo que prediga
la próxima jugada del oponente en PPT usando tus DATOS REALES de partidas.

REQUISITOS PREVIOS:
- Debes tener un CSV con datos de tus partidas contra compañeros
- El CSV debe tener al menos estas columnas:
  * jugada_jugador (tu jugada: piedra/papel/tijera)
  * jugada_oponente (jugada del oponente: piedra/papel/tijera)
  * numero_ronda (número de ronda: 1, 2, 3, ...)

TAREAS:
1. Usar tus datos reales de partidas (mínimo 100 rondas)
2. Crear función para generar features básicas
3. Preparar datos con train/test split
4. Entrenar múltiples modelos (KNN, Decision Tree, Random Forest)
5. Comparar resultados
6. Implementar un jugador IA que use el mejor modelo

OBJETIVO:
- Lograr >50% accuracy (mejor que aleatorio 33%)
- Comparar al menos 3 modelos diferentes
- Mostrar confusion matrix
- (Bonus) Implementar clase JugadorIA

¡BUENA SUERTE!
"""
import os
import pickle
import random
from pathlib import Path
from tokenize import String

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



# Configuracion de rutas
RUTA_PROYECTO = Path(__file__).parent.parent
RUTA_DATOS = RUTA_PROYECTO / "data" / "partidas.csv"
RUTA_MODELO = RUTA_PROYECTO / "models" / "modelo_entrenado.pkl"

# Mapeo de jugadas a numeros (para el modelo)
JUGADA_A_NUM = {"piedra": 0, "papel": 1, "tijera": 2}
NUM_A_JUGADA = {0: "piedra", 1: "papel", 2: "tijera"}

# Que jugada gana a cual
GANA_A = {"piedra": "tijera", "papel": "piedra", "tijera": "papel"}
PIERDE_CONTRA = {"piedra": "papel", "papel": "tijera", "tijera": "piedra"}

# =============================================================================
# PARTE 1: GENERAR FEATURES
# =============================================================================

class PPTFeatureGenerator:
    """
    Genera features para el modelo de PPT

    TODO: Implementa los métodos para generar features básicas
    """

    def __init__(self):
        self.jugadas_counter = {
            "piedra": "papel",
            "papel": "tijera",
            "tijera": "piedra"
        }

    def calcular_resultado(self, jugada_jugador, jugada_oponente):


        """
        Calcula el resultado de una ronda

        TODO: Implementa esta función
        Debe retornar: "victoria", "derrota", o "empate"
        """
        # TU CÓDIGO AQUÍ
        if (jugada_jugador == "piedra" and jugada_oponente == "tijera") or (jugada_jugador == "tijera" and jugada_oponente == "papel") or (jugada_jugador == "papel" and jugada_oponente == "piedra"):
            return "derrota" #del oponente
        elif (jugada_jugador == "piedra" and jugada_oponente == "papel") or (jugada_jugador == "papel" and jugada_oponente == "tijera") or (jugada_jugador == "tijera" and jugada_oponente == "piedra"):
            return "victoria" # del oponente
        else:
            return "empate"

    def generar_features_basicas(self, historial_oponente, historial_jugador, numero_ronda):
        """
        Genera features básicas para una ronda

        Args:
            historial_oponente: lista de jugadas del oponente hasta ahora
            historial_jugador: lista de jugadas del jugador hasta ahora
            numero_ronda: número de ronda actual

        Returns:
            dict con features

        TODO: Genera al menos estas features:
        - freq_piedra, freq_papel, freq_tijera (frecuencia global)
        - freq_5_piedra, freq_5_papel, freq_5_tijera (últimas 5 jugadas)
        - lag_1_piedra, lag_1_papel, lag_1_tijera (última jugada, one-hot)
        - lag_2_* (penúltima jugada)
        - racha_victorias, racha_derrotas
        - numero_ronda
        - fase_inicio, fase_medio, fase_final (one-hot)

        PISTA: Revisa los ejercicios de Feature Engineering (Clase 06)
        """
        features = {}

        # TODO: Implementa features de frecuencia global
        if not historial_oponente.empty:
            total = len(historial_oponente)
            # TU CÓDIGO AQUÍ: Calcula freq_piedra, freq_papel, freq_tijera
            conteo=historial_oponente.value_counts()
            conteo = conteo.reindex(
                ['piedra', 'papel', 'tijera'],
                fill_value=0
            )
            frecuencia=conteo/total
            features['freq_piedra'] = frecuencia['piedra']
            features['freq_papel'] = frecuencia['papel']
            features['freq_tijera'] = frecuencia['tijera']
        else:
            features['freq_piedra'] = 0.33
            features['freq_papel'] = 0.33
            features['freq_tijera'] = 0.33

        # TODO: Implementa features de frecuencia reciente (últimas 5)
        # TU CÓDIGO AQUÍ
        ultimas_cinco_oponente = historial_oponente[-5:]
        cinco_conteo = ultimas_cinco_oponente.value_counts()
        cinco_conteo = cinco_conteo.reindex(
            ['piedra', 'papel', 'tijera'],
            fill_value=0
        )
        frecuencia_5 = cinco_conteo / 5
        features['freq_5_piedra'] = frecuencia_5['piedra']
        features['freq_5_papel'] = frecuencia_5['papel']
        features['freq_5_tijera'] = frecuencia_5['tijera']
        # TODO: Implementa lag features (última y penúltima jugada)
        # TU CÓDIGO AQUÍ
        features['lag_1']=historial_oponente.iloc[-1]
        if len(historial_oponente) >= 2:
            features['lag_2'] = historial_oponente.iloc[-2]
        else:
            # Asignar un valor por defecto si no hay penúltima jugada
            features['lag_2'] = 'ninguna'
        ###
        """
        # =============================================================================
        # NUEVA FEATURE: Frecuencia de Empate después de Empate (E|E)
        # =============================================================================

        resultados_historial = []

        # 1. Generar la secuencia de resultados a partir del historial
        # Nota: Usar .tolist() para manejar los índices de Series correctamente
        for j_op, j_jug in zip(historial_oponente.tolist(), historial_jugador.tolist()):
            resultados_historial.append(self.calcular_resultado(j_jug, j_op))
            # Atención: Usamos j_jug primero, para obtener el resultado desde la perspectiva del JUGADOR

        conteo_E_despues_E = 0
        conteo_total_E_previos = 0

        # 2. Iterar sobre los resultados para calcular la frecuencia
        # Iteramos hasta el penúltimo resultado (índice -2), ya que necesitamos el siguiente (índice + 1)
        for i in range(len(resultados_historial) - 1):
            resultado_ronda_i = resultados_historial[i]
            resultado_ronda_i_mas_1 = resultados_historial[i + 1]

            # Contamos cuántas veces hubo un empate en la ronda i
            if resultado_ronda_i == "empate":
                conteo_total_E_previos += 1

                # Y si el siguiente resultado también fue empate
                if resultado_ronda_i_mas_1 == "empate":
                    conteo_E_despues_E += 1

        # 3. Calcular la frecuencia
        if conteo_total_E_previos > 0:
            # Frecuencia: (Empates Dobles) / (Total de Empates Previos)
            freq_empate_despues_empate = conteo_E_despues_E / conteo_total_E_previos
        else:
            # Si nunca ha habido un empate, la frecuencia es 0
            freq_empate_despues_empate = 0.0

        features['freq_E_despues_E'] = freq_empate_despues_empate
        """
        # =============================================================================
        # NUEVA FEATURE: Jugada más frecuente del Oponente tras Victoria del Jugador (JOTV)
        # =============================================================================

        # 1. Analizar el historial de resultados para encontrar las victorias del jugador (tú)
        indices_victoria_jugador = []

        for i in range(len(historial_oponente)):
            # Usar la jugada del jugador (tú) y oponente de la ronda i
            j_jug = historial_jugador.iloc[i]
            j_op = historial_oponente.iloc[i]

            # Verificamos si el jugador (tú) ganó en la ronda 'i'
            if self.calcular_resultado(j_jug, j_op) == "victoria":
                indices_victoria_jugador.append(i)

        jugadas_oponente_despues_victoria = []

        # 2. Recolectar la jugada del oponente en el turno inmediatamente posterior
        # Si la ronda 'i' fue victoria, la ronda 'i+1' es la reacción que queremos predecir.
        for i in indices_victoria_jugador:
            # Nos interesa la jugada del oponente en el turno 'i+1'.
            # Como estamos dentro del historial (que solo va hasta 'n-1'),
            # necesitamos asegurarnos de que el índice 'i+1' existe.
            if i + 1 < len(historial_oponente):
                jugadas_oponente_despues_victoria.append(historial_oponente.iloc[i + 1])

        # 3. Determinar la jugada más frecuente
        if jugadas_oponente_despues_victoria:
            # Calcula el conteo de cada jugada y toma la que tenga el conteo más alto (la primera si hay empate)
            conteo_reaccion = pd.Series(jugadas_oponente_despues_victoria).value_counts()
            jugada_mas_frecuente = conteo_reaccion.index[0]
        else:
            # Si nunca has ganado, no tenemos patrón de reacción. Usamos la jugada más frecuente global del oponente.
            # Aquí podrías usar una heurística: la jugada más frecuente del historial global.
            conteo_global = historial_oponente.value_counts()
            if not conteo_global.empty:
                jugada_mas_frecuente = conteo_global.index[0]
            else:
                jugada_mas_frecuente = 'ninguna'  # o un valor neutro

        features['reaccion_oponente_a_mi_victoria'] = jugada_mas_frecuente

        # FEATURE 2: Frecuencia de Repetición de Jugada (oponente)
        # -----------------------------------------------------------------------------
        n = len(historial_oponente)
        repeticiones = 0
        total_rondas_analizables = n - 1

        if total_rondas_analizables > 0:
            # Comparamos la jugada en i con la jugada en i-1
            for i in range(1, n):
                if historial_oponente.iloc[i] == historial_oponente.iloc[i - 1]:
                    repeticiones += 1

            freq_repeticion = repeticiones / total_rondas_analizables
        else:
            freq_repeticion = 0.0

        features['freq_repeticion_oponente'] = freq_repeticion


        # FEATURE 6: Jugada más frecuente del Oponente tras un Empate
        # -----------------------------------------------------------------------------

        indices_empate = []

        # 1. Identificar en qué rondas hubo un empate
        for i in range(len(historial_oponente)):
            j_jug = historial_jugador.iloc[i]
            j_op = historial_oponente.iloc[i]

            # Si el resultado es 'empate' (desde tu perspectiva), lo registramos.
            if self.calcular_resultado(j_jug, j_op) == "empate":
                indices_empate.append(i)

        jugadas_oponente_despues_empate = []

        # 2. Recolectar la jugada del oponente en el turno inmediatamente posterior (i+1)
        for i in indices_empate:
            # Asegurarse de que el índice i + 1 existe en el historial
            if i + 1 < len(historial_oponente):
                jugadas_oponente_despues_empate.append(historial_oponente.iloc[i + 1])

        # 3. Determinar la jugada más frecuente
        if jugadas_oponente_despues_empate:
            conteo_reaccion = pd.Series(jugadas_oponente_despues_empate).value_counts()
            jugada_mas_frecuente_empate = conteo_reaccion.index[0]
        else:
            # Si nunca ha habido un empate, no tenemos patrón. Usamos un valor neutro.
            jugada_mas_frecuente_empate = 'ninguna'

        features['reaccion_oponente_a_empate'] = jugada_mas_frecuente_empate




        # TODO: Implementa features de rachas
        # TU CÓDIGO AQUÍ
        racha_victorias = 0
        racha_derrotas = 0
        n = len(historial_oponente)
        if n >= 1:

            # 1. Determinar el índice inicial (el último elemento del historial, que es la Ronda n)
            # Usaremos la posición relativa (índices 0, 1, 2, ... n-1)

            # 2. Iterar hacia atrás desde la última jugada (posición n-1)

            # Evaluar la última jugada (Lag 1)
            jugada_oponente_n = historial_oponente.iloc[n - 1]
            jugada_jugador_n = historial_jugador.iloc[n - 1]

            rtdo = self.calcular_resultado(jugada_jugador_n, jugada_oponente_n)  # Resultado del JUGADOR

            # 3. Calcular la racha
            if rtdo == "derrota":  # Racha de derrotas del jugador (victorias del oponente)
                racha_derrotas = 1

                # Iterar hacia atrás para rachas > 1
                for i in range(n - 2, -1, -1):  # Desde la penúltima (n-2) hasta el inicio (0)
                    rtdo_previo = self.calcular_resultado(historial_jugador.iloc[i], historial_oponente.iloc[i])
                    if rtdo_previo == "derrota":
                        racha_derrotas += 1
                    else:
                        break

            elif rtdo == "victoria":  # Racha de victorias del jugador
                racha_victorias = 1

                for i in range(n - 2, -1, -1):
                    rtdo_previo = self.calcular_resultado(historial_jugador.iloc[i], historial_oponente.iloc[i])
                    if rtdo_previo == "victoria":
                        racha_victorias += 1
                    else:
                        break

        # Asignar features (si no hay historial, serán 0 por defecto)
        features['racha_victorias'] = racha_victorias
        features['racha_derrotas'] = racha_derrotas



        return features


# =============================================================================
# PARTE 2: PREPARAR DATOS
# =============================================================================

def cargar_y_preparar_datos(archivo_csv):
    """
    Carga datos de partidas y genera features

    Args:
        archivo_csv: Ruta a TU archivo CSV con datos de partidas
                     Debe tener columnas: jugada_jugador, jugada_oponente, numero_ronda

    TODO: Implementa esta función
    1. Cargar CSV con pandas
    2. Verificar que tiene las columnas necesarias
    3. Para cada ronda, generar features basadas en historial previo
    4. Target = jugada actual del oponente
    5. Retornar X (features), y (target)

    IMPORTANTE: Usa solo el historial HASTA cada ronda (no hacer trampa)
    """
    print("=" * 70)
    print("PASO 1: CARGAR Y PREPARAR DATOS")
    print("=" * 70)

    # TODO: Cargar TU CSV
    # TU CÓDIGO AQUÍ
    # df = pd.read_csv(archivo_csv)
    df = pd.read_csv(archivo_csv)
    # TODO: Verificar columnas
    columnas_necesarias = ['num_ronda','jugada_jugador','jugada_oponente']
    # columnas_necesarias = ['jugada_jugador', 'jugada_oponente', 'numero_ronda']
    # if not all(col in df.columns for col in columnas_necesarias):
    #     print(f"ERROR: El CSV debe tener columnas: {columnas_necesarias}")
    #     return None, None
    # columnas_necesarias = ['jugada_jugador', 'jugada_oponente', 'numero_ronda']
    if not all(col in df.columns for col in columnas_necesarias):
         print(f"ERROR: El CSV debe tener columnas: {columnas_necesarias}")
         return None, None
    # TODO: Generar features para cada ronda
    # TU CÓDIGO AQUÍ
    feature_gen = PPTFeatureGenerator()
    lista_features = []
    lista_targets = []
    for i in range(1, len(df)):
        # 1. Obtener el historial de jugadas ANTERIORES a la ronda actual (índice i)
        historial_oponente = df['jugada_oponente'][:i]
        historial_jugador = df['jugada_jugador'][:i]
        numero_ronda_actual = df['num_ronda'].iloc[i]  # o simplemente i + 1 si empieza en 1

        # 2. Generar las features usando el historial
        features = feature_gen.generar_features_basicas(
            historial_oponente,
            historial_jugador,
            numero_ronda_actual
        )

        # 3. El TARGET (y) es la jugada del oponente en la ronda actual (índice i)
        target = df['jugada_oponente'].iloc[i]

        # 4. Almacenar
        lista_features.append(features)
        lista_targets.append(target)

    # TODO: Crear DataFrames X e y
    # TU CÓDIGO AQUÍ
    # X = pd.DataFrame(lista_features)
    # y = pd.Series(lista_targets)
    X = pd.DataFrame(lista_features)
    y = pd.Series(lista_targets)
    # print(f"\n✓ Features generadas: {X.shape[0]} muestras, {X.shape[1]} features")

    return X, y





# =============================================================================
# PARTE 3: ENTRENAR MODELOS
# =============================================================================

def entrenar_y_comparar_modelos(X_train, X_test, y_train, y_test):
    """
    Entrena múltiples modelos y los compara

    TODO: Implementa esta función
    1. Definir diccionario con al menos 3 modelos
    2. Entrenar cada modelo
    3. Evaluar en train y test
    4. Mostrar tabla comparativa
    5. Retornar el mejor modelo
    """
    print("\n" + "=" * 70)
    print("PASO 2: ENTRENAR Y COMPARAR MODELOS")
    print("=" * 70)

    # TODO: Define modelos a probar
    # TU CÓDIGO AQUÍ
    # modelos = {
    #     'KNN (K=5)': KNeighborsClassifier(n_neighbors=5),
    #     'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    #     'Random Forest': RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    # }
    modelos = {
        'KNN (K=5)': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    }

    resultados = {}
    mejor_modelo = None
    mejor_accuracy = -1
    nombre_mejor_modelo = ""

    # TODO: Entrena y evalúa cada modelo
    # TU CÓDIGO AQUÍ
    print("Entrenando y evaluando modelos...")
    for nombre, modelo in modelos.items():
        # Entrenar
        modelo.fit(X_train, y_train)

        # Evaluar en Train
        y_train_pred = modelo.predict(X_train)
        acc_train = accuracy_score(y_train, y_train_pred)

        # Evaluar en Test
        y_test_pred = modelo.predict(X_test)
        acc_test = accuracy_score(y_test, y_test_pred)

        resultados[nombre] = {
            'Modelo': modelo,
            'Accuracy_Train': acc_train,
            'Accuracy_Test': acc_test
        }
        if acc_test > mejor_accuracy:
            mejor_accuracy = acc_test
            mejor_modelo = modelo
            nombre_mejor_modelo = nombre
    # TODO: Muestra resultados en tabla
    # TU CÓDIGO AQUÍ
    print("\n### 📋 Tabla Comparativa de Modelos ###")

    # Crear un DataFrame para mostrar los resultados
    df_resultados = pd.DataFrame(
        [{
            'Modelo': nombre,
            'Train Accuracy': res['Accuracy_Train'],
            'Test Accuracy': res['Accuracy_Test']
        } for nombre, res in resultados.items()]
    ).set_index('Modelo')

    # Formatear los resultados para mejor visualización
    df_resultados['Train Accuracy'] = (df_resultados['Train Accuracy'] * 100).map('{:.2f}%'.format)
    df_resultados['Test Accuracy'] = (df_resultados['Test Accuracy'] * 100).map('{:.2f}%'.format)

    print(df_resultados)

    print(
        f"\n🏆 Mejor modelo (en Test Accuracy): **{nombre_mejor_modelo}** con una precisión de **{mejor_accuracy * 100:.2f}%**")
    print(f"⚠️ Objetivo: Lograr una precisión superior al 33.33% (aleatorio)")
    # TODO: Retorna el mejor modelo
    # TU CÓDIGO AQUÍ
    return mejor_modelo
    pass


# =============================================================================
# PARTE 4: EVALUACIÓN DETALLADA
# =============================================================================

def evaluar_mejor_modelo(nombre_modelo, modelo, X_test, y_test):
    """
    Evaluación detallada del mejor modelo

    TODO: Implementa esta función
    1. Hacer predicciones en test
    2. Calcular accuracy
    3. Mostrar confusion matrix
    4. Mostrar classification report
    5. (Bonus) Mostrar feature importance si es árbol/RF
    """
    print("\n" + "=" * 70)
    print(f"PASO 3: EVALUACIÓN DETALLADA - {nombre_modelo}")
    print("=" * 70)

    # TODO: Predicciones
    # TU CÓDIGO AQUÍ

    # TODO: Accuracy
    # TU CÓDIGO AQUÍ

    # TODO: Confusion Matrix
    # TU CÓDIGO AQUÍ

    # TODO: Classification Report
    # TU CÓDIGO AQUÍ

    pass


# =============================================================================
# PARTE 5: JUGADOR IA (BONUS)
# =============================================================================


def entrenar_modelo(X, y, test_size: float = 0.2):
    """
    Entrena el modelo de prediccion.

    TODO: Implementa esta funcion
    - Divide los datos en train/test
    - Entrena al menos 2 modelos diferentes
    - Evalua cada modelo y selecciona el mejor
    - Muestra metricas de evaluacion

    Args:
        X: Features
        y: Target (proxima jugada del oponente)
        test_size: Proporcion de datos para test

    Returns:
        El mejor modelo entrenado
    """
    # TODO: Divide los datos
    # X_train, X_test, y_train, y_test = train_test_split(...)

    # TODO: Entrena varios modelos
    # modelos = {
    #     'KNN': KNeighborsClassifier(n_neighbors=5),
    #     'DecisionTree': DecisionTreeClassifier(),
    #     'RandomForest': RandomForestClassifier()
    # }

    # TODO: Evalua cada modelo
    # Para cada modelo:
    #   - Entrena con fit()
    #   - Predice con predict()
    #   - Calcula accuracy con accuracy_score()
    #   - Muestra classification_report()

    # TODO: Selecciona y retorna el mejor modelo

    pass  # Elimina esta linea cuando implementes


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





class JugadorIA:
    def __init__(self, modelo, feature_generator):
        # El modelo de predicción (Mock o real)

        self.modelo = modelo
        # El generador de features (Mock o real)
        self.feature_gen = feature_generator
        # Historial de jugadas del oponente (última es el final de la lista)
        self.historial_oponente = []
        # Historial de jugadas propias
        self.historial_propio = []
        # Contador de rondas
        self.numero_ronda = 0

        # Reglas básicas para ganar: el "counter"
        self.counter = {
            'piedra': 'papel',
            'papel': 'tijera',
            'tijera': 'piedra'
        }
        self.jugadas_posibles = list(self.counter.keys())

    def predecir_y_jugar(self):
        """
        Predice la próxima jugada del oponente y devuelve el counter

        1. Incrementar número de ronda
        2. Si es ronda 1, jugar aleatorio
        3. Generar features basadas en historial
        4. Predecir jugada del oponente con el modelo
        5. Jugar el counter
        6. Retornar tu jugada

        Returns:
            str: jugada a realizar ('piedra', 'papel', o 'tijera')
        """
        # 1. Incrementar número de ronda
        self.numero_ronda += 1

        if self.numero_ronda == 1:
            # 2. Si es ronda 1, jugar aleatorio (no hay historial para predecir)
            mi_jugada = random.choice(self.jugadas_posibles)
            print(f"Ronda {self.numero_ronda}: Iniciando con jugada aleatoria: {mi_jugada}")
        else:
            try:
                # 3. Generar features basadas en historial
                features = self.feature_gen.generar(self.historial_oponente, self.historial_propio)

                # 4. Predecir jugada del oponente con el modelo
                prediccion_oponente = self.modelo.predecir(features)

                # 5. Jugar el counter de la predicción
                mi_jugada = self.counter[prediccion_oponente]

                print(
                    f"Ronda {self.numero_ronda}: El modelo predice que el oponente jugará '{prediccion_oponente}'. Jugando counter: {mi_jugada}")

            except Exception as e:
                # 4. Fallback: Si la predicción falla (ej. error en el modelo), jugar aleatorio
                print(f"Error en la predicción (Ronda {self.numero_ronda}): {e}. Jugando aleatorio como fallback.")
                mi_jugada = random.choice(self.jugadas_posibles)

        # 6. Retornar tu jugada
        return mi_jugada

    def registrar_resultado(self, mi_jugada, jugada_oponente):
        """
        Registra el resultado de una ronda

        Actualiza los historiales
        """
        # Actualiza los historiales
        self.historial_propio.append(mi_jugada)
        self.historial_oponente.append(jugada_oponente)
        print(f"   -> Resultado registrado. Oponente jugó: {jugada_oponente}\n")


# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

def main():
    print("\n" + "█" * 35)
    print("EJERCICIO: ENTRENAR MODELO PARA PPT")
    print("█" * 35)

    # 1. Especificar ruta a TU archivo CSV con datos reales
    # ----------------------------------------------------
    # TODO: CAMBIA ESTA RUTA por la ubicación de TU archivo CSV
    archivo_csv = 'resultados.csv'
    # Si tu archivo se llama 'resultados.csv' como en tu intento anterior:
    # archivo_csv = 'resultados.csv'

    print(f"\n📂 Usando archivo de datos: {archivo_csv}")

    # 2. Cargar y preparar datos
    # --------------------------
    X, y = cargar_y_preparar_datos(archivo_csv)

    if X is None or y is None:
        print("❌ Terminando el ejercicio debido a un error en la carga/preparación de datos.")
        return

    # 2.1 Preprocesamiento de variables categóricas
    # ============================================
    # Antes de dividir, debemos codificar las columnas categóricas (como 'lag_1' y 'lag_2').
    # Si estas columnas no existen aún (porque no has implementado la Parte 1),
    # este bloque puede fallar. Si es el caso, coméntalo temporalmente.

    try:
        # Identificar columnas categóricas (ej. las features 'lag')
        categorical_features = ['lag_1', 'lag_2', 'reaccion_oponente_a_mi_victoria','reaccion_oponente_a_empate']
        # Definir el transformador: aplica OneHotEncoder a las categóricas
        preprocessor = ColumnTransformer(
            transformers=[
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)],
            remainder='passthrough'  # Mantener el resto de columnas (frecuencias, rachas, num_ronda)
        )

        # Aplicar el preprocesamiento a X
        X = pd.DataFrame(preprocessor.fit_transform(X))
        # Opcional: renombrar columnas para mejor lectura si es necesario (es complejo con ColumnTransformer)

        print(f"✓ Preprocesamiento (One-Hot) aplicado a las features: X.shape={X.shape}")
    except KeyError as e:
        print(
            f"⚠️  ADVERTENCIA: No se pudo aplicar One-Hot Encoding. Asegúrate de implementar las features {e} en PPTFeatureGenerator.")
        # Si no se pudo preprocesar, el modelo KNN/DT/RF no funcionará bien con strings.
        # Es CRUCIAL implementar la Parte 1 (PPTFeatureGenerator) para que esto funcione.
        pass

    # 3. Train/test split
    # -------------------
    # Dividir el conjunto de datos en entrenamiento (70%) y prueba (30%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"\n📊 División de datos: Train={X_train.shape}, Test={X_test.shape}")

    # 4. Entrenar y comparar modelos
    # ------------------------------
    nombre_mejor_modelo, mejor_modelo = entrenar_y_comparar_modelos(
        X_train, X_test, y_train, y_test
    )



    """
    Flujo completo del ejercicio

    TODO: Implementa el flujo completo:
    1. Especificar ruta a TU archivo CSV con datos reales
    2. Cargar y preparar datos
    3. Train/test split
    4. Entrenar y comparar modelos
    5. Evaluación detallada
    6. (Bonus) Simular partida con JugadorIA

    print("\n" + "█" * 35)
    print("EJERCICIO: ENTRENAR MODELO PARA PPT")
    print("█" * 35)

    # TODO: Especifica la ruta a TU CSV con datos de partidas
    # Ejemplo: archivo_csv = 'mis_partidas_ppt.csv'
    # archivo_csv = '???'  # ← Cambia esto

    print("\n⚠️  IMPORTANTE: Debes tener tus datos de partidas en CSV")
    print("    Mínimo 100 rondas jugadas contra compañeros")
    print("    Columnas necesarias: jugada_jugador, jugada_oponente, numero_ronda")
    print()

    # TODO: Implementa el flujo completo aquí
    # TU CÓDIGO AQUÍ

    print("\n" + "█" * 35)
    print("¡EJERCICIO COMPLETADO!")
    print("█" * 35)
    """

if __name__ == "__main__":
    # Descomenta cuando hayas implementado main()
    main()
"""
    print("=" * 70)
    print("INSTRUCCIONES")
    print("=" * 70)
    print("\n1. ANTES DE EMPEZAR:")
    print("   - Debes tener un CSV con tus partidas (mínimo 100 rondas)")
    print("   - Si no tienes datos, juega más partidas con compañeros")
    print("   - Formato CSV necesario:")
    print("     numero_ronda,jugada_jugador,jugada_oponente")
    print("     1,piedra,papel")
    print("     2,tijera,tijera")
    print("     ...")
    print("\n2. Implementa todas las funciones marcadas con TODO")
    print("\n3. Ejecuta este script:")
    print("   python 04_ejercicio_ppt_ALUMNO.py")
    print("\n4. OBJETIVO: Lograr >50% accuracy en test")
    print("\n" + "=" * 70)
    print("RECURSOS:")
    print("- Repasa ejemplos 01-03")
    print("- Revisa Clase 06 (Feature Engineering)")
    print("- Lee clases/07-entrenamiento-modelos/README.md")
    print("=" * 70)
"""