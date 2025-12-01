"""
RPSAI - Modelo de IA para Piedra, Papel o Tijera
=================================================

INSTRUCCIONES PARA EL ALUMNO:
-----------------------------
Este archivo contiene la plantilla para tu modelo de IA.
Debes completar las secciones marcadas con TODO.

El objetivo es crear un modelo que prediga la PROXIMA jugada del oponente
y responda con la jugada que le gana.

FORMATO DEL CSV (minimo requerido):
-----------------------------------
Tu archivo data/partidas.csv debe tener AL MENOS estas columnas:
    - numero_ronda: Numero de la ronda (1, 2, 3...)
    - jugada_j1: Jugada del jugador 1 (piedra/papel/tijera)
    - jugada_j2: Jugada del jugador 2/oponente (piedra/papel/tijera)

Ejemplo:
    numero_ronda,jugada_j1,jugada_j2
    1,piedra,papel
    2,tijera,piedra
    3,papel,papel

Si has capturado datos adicionales (tiempo_reaccion, timestamp, etc.),
puedes usarlos para crear features extra.

EVALUACION:
- 30% Extraccion de datos (documentado en DATOS.md)
- 30% Feature Engineering
- 40% Entrenamiento y funcionamiento del modelo

FLUJO:
1. Cargar datos del CSV
2. Crear features (caracteristicas predictivas)
3. Entrenar modelo(s)
4. Evaluar y seleccionar el mejor
5. Usar el modelo para predecir y jugar
"""

import os
import pickle
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

# Descomenta esta linea si te molesta el warning de sklearn sobre feature names:
# warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Importa aqui los modelos que vayas a usar
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

# TODO: Importa los modelos que necesites (KNN, DecisionTree, RandomForest, etc.)
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier


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


def calcular_resultado(jugada_jugador, jugada_oponente):
    """
    Calcula el resultado de una ronda

    TODO: Implementa esta función
    Debe retornar: "victoria", "derrota", o "empate"
    """
    # TU CÓDIGO AQUÍ
    if (jugada_jugador == "piedra" and jugada_oponente == "tijera") or (
            jugada_jugador == "tijera" and jugada_oponente == "papel") or (
            jugada_jugador == "papel" and jugada_oponente == "piedra"):
        return "derrota"  # del oponente
    elif (jugada_jugador == "piedra" and jugada_oponente == "papel") or (
            jugada_jugador == "papel" and jugada_oponente == "tijera") or (
            jugada_jugador == "tijera" and jugada_oponente == "piedra"):
        return "victoria"  # del oponente
    else:
        return "empate"


def generar_features_basicas(historial_oponente, historial_jugador, numero_ronda) ->dict :
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
        conteo = historial_oponente.value_counts()
        conteo = conteo.reindex(
            ['piedra', 'papel', 'tijera'],
            fill_value=0
        )
        frecuencia = conteo / total
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
    features['lag_1'] = historial_oponente.iloc[-1]
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
        if calcular_resultado(j_jug, j_op) == "victoria":
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



    if jugadas_oponente_despues_victoria:
        conteo_reaccion = pd.Series(jugadas_oponente_despues_victoria).value_counts()
        jugada_mas_frecuente = conteo_reaccion.index[0]
    else:
        # ⚠️ POSIBLE FUENTE DE SESGO: La jugada más frecuente global podría ser siempre la misma.
        conteo_global = historial_oponente.value_counts()
        if not conteo_global.empty:
            jugada_mas_frecuente = conteo_global.index[0]
        else:
            jugada_mas_frecuente = 'ninguna'
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
        if calcular_resultado(j_jug, j_op) == "empate":
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

        rtdo = calcular_resultado(jugada_jugador_n, jugada_oponente_n)  # Resultado del JUGADOR

        # 3. Calcular la racha
        if rtdo == "derrota":  # Racha de derrotas del jugador (victorias del oponente)
            racha_derrotas = 1

            # Iterar hacia atrás para rachas > 1
            for i in range(n - 2, -1, -1):  # Desde la penúltima (n-2) hasta el inicio (0)
                rtdo_previo = calcular_resultado(historial_jugador.iloc[i], historial_oponente.iloc[i])
                if rtdo_previo == "derrota":
                    racha_derrotas += 1
                else:
                    break

        elif rtdo == "victoria":  # Racha de victorias del jugador
            racha_victorias = 1

            for i in range(n - 2, -1, -1):
                rtdo_previo = calcular_resultado(historial_jugador.iloc[i], historial_oponente.iloc[i])
                if rtdo_previo == "victoria":
                    racha_victorias += 1
                else:
                    break

    # Asignar features (si no hay historial, serán 0 por defecto)
    features['racha_victorias'] = racha_victorias
    features['racha_derrotas'] = racha_derrotas

    return features


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
        'Decision Tree': DecisionTreeClassifier(max_depth=None, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
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




# =============================================================================
# PARTE 1: EXTRACCION DE DATOS (30% de la nota)
# =============================================================================

def cargar_datos(ruta_csv: str = None) -> pd.DataFrame:
    """
    Carga los datos del CSV de partidas.

    TODO: Implementa esta funcion
    - Usa pandas para leer el CSV
    - Maneja el caso de que el archivo no exista
    - Verifica que tenga las columnas necesarias

    Args:
        ruta_csv: Ruta al archivo CSV (usa RUTA_DATOS por defecto)

    Returns:
        DataFrame con los datos de las partidas
    """
    if ruta_csv is None:
        ruta_csv = RUTA_DATOS

    df = pd.read_csv(ruta_csv)
    return df
    # TODO: Implementa la carga de datos
    # Pista: usa pd.read_csv()


    #pass  # Elimina esta linea cuando implementes


def preparar_datos(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Crear el TARGET (jugada futura del oponente).
    # Usamos la columna 'jugada_oponente' para esto.
    df['proxima_jugada_j2'] = df['jugada_oponente'].shift(-1)

    # NOTA: Ya no es necesario renombrar, asumimos que 'num_ronda'
    # ya existe con ese nombre en el DataFrame de entrada.

    # 2. Eliminar la última fila que contiene NaN en el target
    df = df.dropna(subset=['proxima_jugada_j2'])

    # 3. Mapear el TARGET a números (0, 1, 2)
    df['proxima_jugada_j2'] = df['proxima_jugada_j2'].map(JUGADA_A_NUM)

    # 4. Limpieza final: Eliminar filas donde el mapeo haya fallado.
    df = df.dropna(subset=['proxima_jugada_j2'])

    return df
    # Pistas:
    # - Usa map() con JUGADA_A_NUM para convertir jugadas a numeros
    # - Usa shift(-1) para crear la columna de proxima jugada
    # - Usa dropna() para eliminar filas con NaN

    #pass  # Elimina esta linea cuando implementes


# =============================================================================
# PARTE 2: FEATURE ENGINEERING (30% de la nota)
# =============================================================================

def crear_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    lista_features = []

    # Asegúrate de que df['jugada_oponente'] y df['jugada_jugador'] son Series de strings.
    # Si 'preparar_datos' las mapeó a números, debes cargarlas de nuevo.
    # En este código, asumimos que preparacion de datos es correcta para el target,
    # y que las columnas 'jugada_oponente' y 'jugada_jugador' son strings.

    # 🟢 CAMBIO CLAVE 3: Iterar desde la fila 1 para tener al menos 1 jugada previa.
    # Los índices de Python son 0-based.
    # El rango debe ser desde 1 hasta len(df) - 1, si df ya fue dropeado en 'preparar_datos'.
    # Pero si df es el resultado de 'preparar_datos', los índices ya están alineados.

    # La X de features debe ser 1 fila más corta que el df preparado (porque 'lag_1' no existe en la ronda 0)
    # y porque la jugada a predecir es la i+1 (por eso se usa shift(-1)).

    # Si df tiene N filas, queremos N-1 features (de la ronda 1 a la N-1, para predecir de 2 a N).

    # Iteramos sobre todas las filas *excepto* la primera (ronda 1), ya que no tienen historial.
    for i in range(1, len(df)):
        # 1. Obtener el historial de jugadas ANTERIORES a la ronda actual (índice i)
        # El historial va de la ronda 0 hasta la ronda i-1
        historial_oponente = df['jugada_oponente'].iloc[:i]
        historial_jugador = df['jugada_jugador'].iloc[:i]

        # Asumiendo que tu CSV tiene 'numero_ronda' (no 'num_ronda')
        numero_ronda_actual = df['num_ronda'].iloc[i]

        # 2. Generar las features usando el historial
        features = generar_features_basicas(
            historial_oponente,
            historial_jugador,
            numero_ronda_actual
        )

        # 4. Almacenar
        lista_features.append(features)

    X = pd.DataFrame(lista_features)

    # 🟢 CAMBIO CLAVE 4: Asignar los índices correctos a X.
    # X tiene len(df) - 1 filas. Sus índices deben coincidir con las filas de df
    # que contienen el target (df.iloc[1:]).
    X.index = df.iloc[1:].index

    return X
    # ------------------------------------------
    # TODO: Feature 1 - Frecuencia de jugadas
    # ------------------------------------------
    # Calcula que porcentaje de veces j2 juega cada opcion
    # Pista: usa expanding().mean() o rolling()

    # ------------------------------------------
    # TODO: Feature 2 - Lag features (jugadas anteriores)
    # ------------------------------------------
    # Crea columnas con las ultimas 1, 2, 3 jugadas
    # Pista: usa shift(1), shift(2), etc.

    # ------------------------------------------
    # TODO: Feature 3 - Resultado anterior
    # ------------------------------------------
    # Crea una columna con el resultado de la ronda anterior
    # Esto puede revelar patrones (ej: siempre cambia despues de perder)

    # ------------------------------------------
    # TODO: Mas features (opcional pero recomendado)
    # ------------------------------------------
    # Agrega mas features que creas utiles
    # Recuerda: mas features relevantes = mejor prediccion


def seleccionar_features(df_preparado: pd.DataFrame, X_features: pd.DataFrame) -> tuple:
    """
    Selecciona y alinea las features (X_features) y el target (proxima_jugada_j2).

    Args:
        df_preparado: DataFrame resultante de preparar_datos (contiene el TARGET).
        X_features: DataFrame generado por crear_features (contiene las FEATURES).

    Returns:
        (X, y) - Features y target limpios y alineados.
    """
    # 1. Definir la columna target
    TARGET_COL = 'proxima_jugada_j2'

    # --- VERIFICACIONES INICIALES ---
    if TARGET_COL not in df_preparado.columns:
        # Esto debería haberse evitado en main(), pero es una buena defensa
        raise ValueError(f"La columna target '{TARGET_COL}' no se encuentra en el DataFrame preparado.")

    # 2. Obtener la columna TARGET (y) y alinearlo con el índice de X_features
    # La X_features ya tiene los índices correctos de las filas que no son NaN
    # y que tienen historial suficiente para las features de lag.
    y = df_preparado[TARGET_COL].loc[X_features.index]

    # 3. Features (X)
    X = X_features

    # 4. Asegurarse de que las columnas features utilizadas son las finales
    # Esta parte asume que X_features ya viene limpia y transformada
    # y que las columnas originales de feature_cols solo se usaron para validación.

    # 5. Verificación de alineación (crucial)
    if len(X) != len(y):
        raise ValueError(
            f"ERROR DE ALINEACIÓN: X tiene {len(X)} filas, pero y tiene {len(y)}. "
            "Revise la lógica de indexación en crear_features y preparar_datos."
        )

    print(f"✅ Features y Target alineados: {len(X)} filas listas para entrenar.")

    return X, y

    #pass  # Elimina esta linea cuando implementes


# =============================================================================
# PARTE 3: ENTRENAMIENTO Y FUNCIONAMIENTO (40% de la nota)
# =============================================================================

def entrenar_modelo(X, y, test_size: float = 0.2):
    # ... (código existente para definir categorical_features)

    # 1. Definir y ajustar el preprocessor
    categorical_features = ['lag_1', 'lag_2', 'reaccion_oponente_a_mi_victoria', 'reaccion_oponente_a_empate']
    preprocessor = ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)],
        remainder='passthrough'
    )

    # AJUSTAR (fit) el preprocessor a X antes de dividir
    preprocessor.fit(X)

    # Aplicar la transformación a X
    X_processed = pd.DataFrame(preprocessor.transform(X))

    # 2. División de datos con datos ya transformados
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=test_size, random_state=42, stratify=y
    )

    # ... (entrenamiento de modelos) ...

    mejor_modelo = entrenar_y_comparar_modelos(
        X_train, X_test, y_train, y_test
    )

    # Devolver el mejor modelo Y el preprocessor ajustado
    return mejor_modelo, preprocessor  # <--- CAMBIO CLAVE
    # TODO: Selecciona y retorna el mejor modelo

    #pass  # Elimina esta linea cuando implementes


def guardar_modelo(modelo, preprocessor, ruta: str = None):  # <--- Acepta preprocessor
    """Guarda el modelo entrenado y el preprocesador."""
    if ruta is None:
        ruta = RUTA_MODELO

    ruta_preprocessor = ruta.with_name("preprocessor_entrenado.pkl")  # Nuevo archivo

    os.makedirs(os.path.dirname(ruta), exist_ok=True)

    # Guardar modelo
    with open(ruta, "wb") as f:
        pickle.dump(modelo, f)
    print(f"Modelo guardado en: {ruta}")

    # Guardar preprocessor
    with open(ruta_preprocessor, "wb") as f:
        pickle.dump(preprocessor, f)
    print(f"Preprocessor guardado en: {ruta_preprocessor}")


# En src/modelo.py - Modificar cargar_modelo (para cargar ambos)

def cargar_modelo(ruta: str = None):
    """Carga un modelo previamente entrenado y su preprocesador."""
    if ruta is None:
        ruta = RUTA_MODELO
    ruta_preprocessor = ruta.with_name("preprocessor_entrenado.pkl")

    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No se encontro el modelo en: {ruta}")
    if not os.path.exists(ruta_preprocessor):
        raise FileNotFoundError(f"No se encontro el preprocesador en: {ruta_preprocessor}")

    with open(ruta, "rb") as f:
        modelo = pickle.load(f)

    with open(ruta_preprocessor, "rb") as f:
        preprocessor = pickle.load(f)

    return modelo, preprocessor



# =============================================================================
# PARTE 4: PREDICCION Y JUEGO
# =============================================================================

class JugadorIA:
    """
    Clase que encapsula el modelo para jugar.

    TODO: Completa esta clase para que pueda:
    - Cargar un modelo entrenado
    - Mantener historial de la partida actual
    - Predecir la proxima jugada del oponente
    - Decidir que jugada hacer para ganar
    """

    def __init__(self, ruta_modelo: str = None):
        self.modelo = None
        self.preprocessor = None # Nuevo atributo
        self.historial = []

        try:
            # Cargar ahora retorna modelo Y preprocessor
            self.modelo, self.preprocessor = cargar_modelo(ruta_modelo) # <--- CAMBIO
        except FileNotFoundError:
            print("Modelo o Preprocesador no encontrados. Entrena primero.")

    def registrar_ronda(self, jugada_j1: str, jugada_j2: str):
        """
        Registra una ronda jugada para actualizar el historial.

        Args:
            jugada_j1: Jugada del jugador 1
            jugada_j2: Jugada del oponente
        """
        self.historial.append((jugada_j1, jugada_j2))

    def obtener_features_actuales(self) -> np.ndarray:
        if self.preprocessor is None:
            raise RuntimeError("El preprocesador no se ha cargado. No se puede predecir.")

        # 1. Generar features a partir del historial
        if not self.historial:
            features_dict = {
                # ... (features numéricas) ...
                'lag_1': 'ninguna',  # Debe ser el string 'ninguna'
                'lag_2': 'ninguna',  # Debe ser el string 'ninguna'
                'reaccion_oponente_a_mi_victoria': 'ninguna',
                'freq_repeticion_oponente': 0.0,
                'reaccion_oponente_a_empate': 'ninguna',
                # ... (features de racha) ...
            }
            X_actual = pd.DataFrame([features_dict])
        else:
            # Convertir historial a DF para generar features
            historial_df = pd.DataFrame(
                self.historial,
                columns=['jugada_jugador', 'jugada_oponente']
            )
            # Llamar a generar_features_basicas (asumiendo que está disponible globalmente)
            features_dict = generar_features_basicas(
                historial_df['jugada_oponente'],
                historial_df['jugada_jugador'],
                len(self.historial) + 1
            )
            X_actual = pd.DataFrame([features_dict])

        # 2. Aplicar el preprocesador AJUSTADO (transform)
        # Esto convierte los strings como 'ninguna' a números 0s y 1s.
        X_processed = self.preprocessor.transform(X_actual)

        # 3. Devolver el array 2D listo para el modelo
        return X_processed

    def predecir_jugada_oponente(self) -> str:
        """
        Predice la proxima jugada del oponente.
        """
        if self.modelo is None:
            return np.random.choice(["piedra", "papel", "tijera"])
        else:
            features = self.obtener_features_actuales()

            # CORRECCIÓN: Extraer el primer elemento (el número entero) del array
            prediccion_array = self.modelo.predict(features)
            prediccion_entero = prediccion_array[0]  # O .item() si el array es unidimensional

            # Usar el entero (0, 1, o 2) como clave
            return NUM_A_JUGADA[prediccion_entero]

    def decidir_jugada(self) -> str:
        """
        Decide que jugada hacer para ganar al oponente.

        Returns:
            La jugada que gana a la prediccion del oponente
        """
        prediccion_oponente = self.predecir_jugada_oponente()

        if prediccion_oponente is None:
            return np.random.choice(["piedra", "papel", "tijera"])

        # Juega lo que le gana a la prediccion
        return PIERDE_CONTRA[prediccion_oponente]


# =============================================================================
# FUNCION PRINCIPAL
# =============================================================================

def main():
    """
    Funcion principal para entrenar el modelo.

    Ejecuta: python src/modelo.py
    """
    print("="*50)
    print("   RPSAI - Entrenamiento del Modelo")
    print("="*50)

    # TODO: Implementa el flujo completo:
    # 1. Cargar datos
    df=cargar_datos()
    # 2. Preparar datos
    df = preparar_datos(df)
    # 3. Crear features
    x_features = crear_features(df)
    # 4. Seleccionar features
    X, y = seleccionar_features(df,x_features)
    # 5. Entrenar modelo
    modelo_best, preprocessor_best = entrenar_modelo(X, y, 0.2)  # <--- CAMBIO
    # 6. Guardar modelo (Necesita guardar ambos)
    guardar_modelo(modelo_best, preprocessor_best)  # <--

    print("\n[!] Implementa las funciones marcadas con TODO")
    print("[!] Luego ejecuta este script para entrenar tu modelo")


if __name__ == "__main__":
    main()
"""
RPSAI - Modelo de IA para Piedra, Papel o Tijera
=================================================

INSTRUCCIONES PARA EL ALUMNO:
-----------------------------
Este archivo contiene la plantilla para tu modelo de IA.
Debes completar las secciones marcadas con TODO.

El objetivo es crear un modelo que prediga la PROXIMA jugada del oponente
y responda con la jugada que le gana.

FORMATO DEL CSV (minimo requerido):
-----------------------------------
Tu archivo data/partidas.csv debe tener AL MENOS estas columnas:
    - numero_ronda: Numero de la ronda (1, 2, 3...)
    - jugada_j1: Jugada del jugador 1 (piedra/papel/tijera)
    - jugada_j2: Jugada del jugador 2/oponente (piedra/papel/tijera)

Ejemplo:
    numero_ronda,jugada_j1,jugada_j2
    1,piedra,papel
    2,tijera,piedra
    3,papel,papel

Si has capturado datos adicionales (tiempo_reaccion, timestamp, etc.),
puedes usarlos para crear features extra.

EVALUACION:
- 30% Extraccion de datos (documentado en DATOS.md)
- 30% Feature Engineering
- 40% Entrenamiento y funcionamiento del modelo

FLUJO:
1. Cargar datos del CSV
2. Crear features (caracteristicas predictivas)
3. Entrenar modelo(s)
4. Evaluar y seleccionar el mejor
5. Usar el modelo para predecir y jugar
"""

import os
import pickle
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

# Descomenta esta linea si te molesta el warning de sklearn sobre feature names:
# warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Importa aqui los modelos que vayas a usar
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

# TODO: Importa los modelos que necesites (KNN, DecisionTree, RandomForest, etc.)
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier


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


def calcular_resultado(jugada_jugador, jugada_oponente):
    """
    Calcula el resultado de una ronda

    TODO: Implementa esta función
    Debe retornar: "victoria", "derrota", o "empate"
    """
    # TU CÓDIGO AQUÍ
    if (jugada_jugador == "piedra" and jugada_oponente == "tijera") or (
            jugada_jugador == "tijera" and jugada_oponente == "papel") or (
            jugada_jugador == "papel" and jugada_oponente == "piedra"):
        return "derrota"  # del oponente
    elif (jugada_jugador == "piedra" and jugada_oponente == "papel") or (
            jugada_jugador == "papel" and jugada_oponente == "tijera") or (
            jugada_jugador == "tijera" and jugada_oponente == "piedra"):
        return "victoria"  # del oponente
    else:
        return "empate"


def generar_features_basicas(historial_oponente, historial_jugador, numero_ronda) ->dict :
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
        conteo = historial_oponente.value_counts()
        conteo = conteo.reindex(
            ['piedra', 'papel', 'tijera'],
            fill_value=0
        )
        frecuencia = conteo / total
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
    features['lag_1'] = historial_oponente.iloc[-1]
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
        if calcular_resultado(j_jug, j_op) == "victoria":
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



    if jugadas_oponente_despues_victoria:
        conteo_reaccion = pd.Series(jugadas_oponente_despues_victoria).value_counts()
        jugada_mas_frecuente = conteo_reaccion.index[0]
    else:
        # ⚠️ POSIBLE FUENTE DE SESGO: La jugada más frecuente global podría ser siempre la misma.
        conteo_global = historial_oponente.value_counts()
        if not conteo_global.empty:
            jugada_mas_frecuente = conteo_global.index[0]
        else:
            jugada_mas_frecuente = 'ninguna'
    features['reaccion_oponente_a_mi_victoria'] = jugada_mas_frecuente
    """
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
        if calcular_resultado(j_jug, j_op) == "empate":
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

    features['reaccion_oponente_a_empate'] = jugada_mas_frecuente_empate"""

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

        rtdo = calcular_resultado(jugada_jugador_n, jugada_oponente_n)  # Resultado del JUGADOR

        # 3. Calcular la racha
        if rtdo == "derrota":  # Racha de derrotas del jugador (victorias del oponente)
            racha_derrotas = 1

            # Iterar hacia atrás para rachas > 1
            for i in range(n - 2, -1, -1):  # Desde la penúltima (n-2) hasta el inicio (0)
                rtdo_previo = calcular_resultado(historial_jugador.iloc[i], historial_oponente.iloc[i])
                if rtdo_previo == "derrota":
                    racha_derrotas += 1
                else:
                    break

        elif rtdo == "victoria":  # Racha de victorias del jugador
            racha_victorias = 1

            for i in range(n - 2, -1, -1):
                rtdo_previo = calcular_resultado(historial_jugador.iloc[i], historial_oponente.iloc[i])
                if rtdo_previo == "victoria":
                    racha_victorias += 1
                else:
                    break

    # Asignar features (si no hay historial, serán 0 por defecto)
    features['racha_victorias'] = racha_victorias
    features['racha_derrotas'] = racha_derrotas

    return features


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
        'Decision Tree': DecisionTreeClassifier(max_depth=None, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
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




# =============================================================================
# PARTE 1: EXTRACCION DE DATOS (30% de la nota)
# =============================================================================

def cargar_datos(ruta_csv: str = None) -> pd.DataFrame:
    """
    Carga los datos del CSV de partidas.

    TODO: Implementa esta funcion
    - Usa pandas para leer el CSV
    - Maneja el caso de que el archivo no exista
    - Verifica que tenga las columnas necesarias

    Args:
        ruta_csv: Ruta al archivo CSV (usa RUTA_DATOS por defecto)

    Returns:
        DataFrame con los datos de las partidas
    """
    if ruta_csv is None:
        ruta_csv = RUTA_DATOS

    df = pd.read_csv(ruta_csv)
    return df
    # TODO: Implementa la carga de datos
    # Pista: usa pd.read_csv()


    #pass  # Elimina esta linea cuando implementes


def preparar_datos(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. CREAR el TARGET: Usar la columna 'jugada_oponente' para crear la
    # columna 'proxima_jugada_j2' usando shift(-1).
    df['proxima_jugada_j2'] = df['jugada_oponente'].shift(-1)

    # 2. Eliminar la última fila que contiene NaN en el target
    # Esta línea limpia el NaN generado por shift(-1).
    df = df.dropna(subset=['proxima_jugada_j2'])

    # 3. Mapear el TARGET a números (0, 1, 2)
    # ESTA LÍNEA AHORA FUNCIONARÁ porque la columna ya existe.
    df['proxima_jugada_j2'] = df['proxima_jugada_j2'].map(JUGADA_A_NUM)

    # 4. Limpieza final: Eliminar filas donde el mapeo haya fallado.
    df = df.dropna(subset=['proxima_jugada_j2'])

    return df


# =============================================================================
# PARTE 2: FEATURE ENGINEERING (30% de la nota)
# =============================================================================

def crear_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    lista_features = []

    # Asegúrate de que df['jugada_oponente'] y df['jugada_jugador'] son Series de strings.
    # Si 'preparar_datos' las mapeó a números, debes cargarlas de nuevo.
    # En este código, asumimos que preparacion de datos es correcta para el target,
    # y que las columnas 'jugada_oponente' y 'jugada_jugador' son strings.

    # 🟢 CAMBIO CLAVE 3: Iterar desde la fila 1 para tener al menos 1 jugada previa.
    # Los índices de Python son 0-based.
    # El rango debe ser desde 1 hasta len(df) - 1, si df ya fue dropeado en 'preparar_datos'.
    # Pero si df es el resultado de 'preparar_datos', los índices ya están alineados.

    # La X de features debe ser 1 fila más corta que el df preparado (porque 'lag_1' no existe en la ronda 0)
    # y porque la jugada a predecir es la i+1 (por eso se usa shift(-1)).

    # Si df tiene N filas, queremos N-1 features (de la ronda 1 a la N-1, para predecir de 2 a N).

    # Iteramos sobre todas las filas *excepto* la primera (ronda 1), ya que no tienen historial.
    for i in range(1, len(df)):
        # 1. Obtener el historial de jugadas ANTERIORES a la ronda actual (índice i)
        # El historial va de la ronda 0 hasta la ronda i-1
        historial_oponente = df['jugada_oponente'].iloc[:i]
        historial_jugador = df['jugada_jugador'].iloc[:i]

        # Asumiendo que tu CSV tiene 'numero_ronda' (no 'num_ronda')
        numero_ronda_actual = df['num_ronda'].iloc[i]

        # 2. Generar las features usando el historial
        features = generar_features_basicas(
            historial_oponente,
            historial_jugador,
            numero_ronda_actual
        )

        # 4. Almacenar
        lista_features.append(features)

    X = pd.DataFrame(lista_features)

    # 🟢 CAMBIO CLAVE 4: Asignar los índices correctos a X.
    # X tiene len(df) - 1 filas. Sus índices deben coincidir con las filas de df
    # que contienen el target (df.iloc[1:]).
    X.index = df.iloc[1:].index

    return X
    # ------------------------------------------
    # TODO: Feature 1 - Frecuencia de jugadas
    # ------------------------------------------
    # Calcula que porcentaje de veces j2 juega cada opcion
    # Pista: usa expanding().mean() o rolling()

    # ------------------------------------------
    # TODO: Feature 2 - Lag features (jugadas anteriores)
    # ------------------------------------------
    # Crea columnas con las ultimas 1, 2, 3 jugadas
    # Pista: usa shift(1), shift(2), etc.

    # ------------------------------------------
    # TODO: Feature 3 - Resultado anterior
    # ------------------------------------------
    # Crea una columna con el resultado de la ronda anterior
    # Esto puede revelar patrones (ej: siempre cambia despues de perder)

    # ------------------------------------------
    # TODO: Mas features (opcional pero recomendado)
    # ------------------------------------------
    # Agrega mas features que creas utiles
    # Recuerda: mas features relevantes = mejor prediccion


def seleccionar_features(df_preparado: pd.DataFrame, X_features: pd.DataFrame) -> tuple:
    """
    Selecciona y alinea las features (X_features) y el target (proxima_jugada_j2).

    Args:
        df_preparado: DataFrame resultante de preparar_datos (contiene el TARGET).
        X_features: DataFrame generado por crear_features (contiene las FEATURES).

    Returns:
        (X, y) - Features y target limpios y alineados.
    """
    # 1. Definir la columna target
    TARGET_COL = 'proxima_jugada_j2'

    # --- VERIFICACIONES INICIALES ---
    if TARGET_COL not in df_preparado.columns:
        # Esto debería haberse evitado en main(), pero es una buena defensa
        raise ValueError(f"La columna target '{TARGET_COL}' no se encuentra en el DataFrame preparado.")

    # 2. Obtener la columna TARGET (y) y alinearlo con el índice de X_features
    # La X_features ya tiene los índices correctos de las filas que no son NaN
    # y que tienen historial suficiente para las features de lag.
    y = df_preparado[TARGET_COL].loc[X_features.index]

    # 3. Features (X)
    X = X_features

    # 4. Asegurarse de que las columnas features utilizadas son las finales
    # Esta parte asume que X_features ya viene limpia y transformada
    # y que las columnas originales de feature_cols solo se usaron para validación.

    # 5. Verificación de alineación (crucial)
    if len(X) != len(y):
        raise ValueError(
            f"ERROR DE ALINEACIÓN: X tiene {len(X)} filas, pero y tiene {len(y)}. "
            "Revise la lógica de indexación en crear_features y preparar_datos."
        )

    print(f"✅ Features y Target alineados: {len(X)} filas listas para entrenar.")

    return X, y

    #pass  # Elimina esta linea cuando implementes


# =============================================================================
# PARTE 3: ENTRENAMIENTO Y FUNCIONAMIENTO (40% de la nota)
# =============================================================================

def entrenar_modelo(X, y, test_size: float = 0.2):
    # ... (código existente para definir categorical_features)

    # 1. Definir y ajustar el preprocessor
    categorical_features = ['lag_1', 'lag_2', 'reaccion_oponente_a_mi_victoria']
    preprocessor = ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)],
        remainder='passthrough'
    )

    # AJUSTAR (fit) el preprocessor a X antes de dividir
    preprocessor.fit(X)

    # Aplicar la transformación a X
    X_processed = pd.DataFrame(preprocessor.transform(X))

    # 2. División de datos con datos ya transformados
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=test_size, random_state=42, stratify=y
    )

    # ... (entrenamiento de modelos) ...

    mejor_modelo = entrenar_y_comparar_modelos(
        X_train, X_test, y_train, y_test
    )

    # Devolver el mejor modelo Y el preprocessor ajustado
    return mejor_modelo, preprocessor  # <--- CAMBIO CLAVE
    # TODO: Selecciona y retorna el mejor modelo

    #pass  # Elimina esta linea cuando implementes


def guardar_modelo(modelo, preprocessor, ruta: str = None):  # <--- Acepta preprocessor
    """Guarda el modelo entrenado y el preprocesador."""
    if ruta is None:
        ruta = RUTA_MODELO

    ruta_preprocessor = ruta.with_name("preprocessor_entrenado.pkl")  # Nuevo archivo

    os.makedirs(os.path.dirname(ruta), exist_ok=True)

    # Guardar modelo
    with open(ruta, "wb") as f:
        pickle.dump(modelo, f)
    print(f"Modelo guardado en: {ruta}")

    # Guardar preprocessor
    with open(ruta_preprocessor, "wb") as f:
        pickle.dump(preprocessor, f)
    print(f"Preprocessor guardado en: {ruta_preprocessor}")


# En src/modelo.py - Modificar cargar_modelo (para cargar ambos)

def cargar_modelo(ruta: str = None):
    """Carga un modelo previamente entrenado y su preprocesador."""
    if ruta is None:
        ruta = RUTA_MODELO
    ruta_preprocessor = ruta.with_name("preprocessor_entrenado.pkl")

    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No se encontro el modelo en: {ruta}")
    if not os.path.exists(ruta_preprocessor):
        raise FileNotFoundError(f"No se encontro el preprocesador en: {ruta_preprocessor}")

    with open(ruta, "rb") as f:
        modelo = pickle.load(f)

    with open(ruta_preprocessor, "rb") as f:
        preprocessor = pickle.load(f)

    return modelo, preprocessor



# =============================================================================
# PARTE 4: PREDICCION Y JUEGO
# =============================================================================

class JugadorIA:
    """
    Clase que encapsula el modelo para jugar.

    TODO: Completa esta clase para que pueda:
    - Cargar un modelo entrenado
    - Mantener historial de la partida actual
    - Predecir la proxima jugada del oponente
    - Decidir que jugada hacer para ganar
    """

    def __init__(self, ruta_modelo: str = None):
        self.modelo = None
        self.preprocessor = None # Nuevo atributo
        self.historial = []

        try:
            # Cargar ahora retorna modelo Y preprocessor
            self.modelo, self.preprocessor = cargar_modelo(ruta_modelo) # <--- CAMBIO
        except FileNotFoundError:
            print("Modelo o Preprocesador no encontrados. Entrena primero.")

    def registrar_ronda(self, jugada_j1: str, jugada_j2: str):
        """
        Registra una ronda jugada para actualizar el historial.

        Args:
            jugada_j1: Jugada del jugador 1
            jugada_j2: Jugada del oponente
        """
        self.historial.append((jugada_j1, jugada_j2))

    def obtener_features_actuales(self) -> np.ndarray:
        if self.preprocessor is None:
            raise RuntimeError("El preprocesador no se ha cargado. No se puede predecir.")

        # 1. Generar features a partir del historial
        if not self.historial:
            # 🟢 CORRECCIÓN: Inicializar TODAS las features con valores neutros (0, 'ninguna')
            features_dict = {
                # Features Categóricas (las que esperan OneHotEncoder):
                'lag_1': 'ninguna',
                'lag_2': 'ninguna',
                'reaccion_oponente_a_mi_victoria': 'ninguna',

                # Features Numéricas (las que esperan 'remainder' o 'passthrough'):
                'freq_piedra': 0.0,
                'freq_papel': 0.0,
                'freq_tijera': 0.0,
                'freq_5_piedra': 0.0,
                'freq_5_papel': 0.0,
                'freq_5_tijera': 0.0,
                'racha_victorias': 0,
                'racha_derrotas': 0,
                # Asegúrate de incluir cualquier otra feature numérica que uses (ej: 'num_ronda')
                'num_ronda': 1  # Si empiezas a contar desde la ronda 1
            }
            X_actual = pd.DataFrame([features_dict])
        else:
            # ... (el código para cuando sí hay historial, que llama a generar_features_basicas)
            # ... (este bloque ya parece estar bien si generar_features_basicas funciona)
            historial_df = pd.DataFrame(
                self.historial,
                columns=['jugada_jugador', 'jugada_oponente']
            )
            features_dict = generar_features_basicas(
                historial_df['jugada_oponente'],
                historial_df['jugada_jugador'],
                len(self.historial) + 1
            )
            X_actual = pd.DataFrame([features_dict])

            # 2. Aplicar el preprocesador AJUSTADO (transform)
        X_processed = self.preprocessor.transform(X_actual)

        # 3. Devolver el array 2D listo para el modelo
        return X_processed

    def predecir_jugada_oponente(self) -> str:
        """
        Predice la proxima jugada del oponente.
        """
        if self.modelo is None:
            return np.random.choice(["piedra", "papel", "tijera"])
        else:
            features = self.obtener_features_actuales()

            # CORRECCIÓN: Extraer el primer elemento (el número entero) del array
            prediccion_array = self.modelo.predict(features)
            prediccion_entero = prediccion_array[0]  # O .item() si el array es unidimensional

            # Usar el entero (0, 1, o 2) como clave
            return NUM_A_JUGADA[prediccion_entero]

    def decidir_jugada(self) -> str:
        """
        Decide que jugada hacer para ganar al oponente.

        Returns:
            La jugada que gana a la prediccion del oponente
        """
        prediccion_oponente = self.predecir_jugada_oponente()

        if prediccion_oponente is None:
            return np.random.choice(["piedra", "papel", "tijera"])

        # Juega lo que le gana a la prediccion
        return PIERDE_CONTRA[prediccion_oponente]


# =============================================================================
# FUNCION PRINCIPAL
# =============================================================================

def main():
    """
    Funcion principal para entrenar el modelo.

    Ejecuta: python src/modelo.py
    """
    print("="*50)
    print("   RPSAI - Entrenamiento del Modelo")
    print("="*50)

    # TODO: Implementa el flujo completo:
    # 1. Cargar datos
    df=cargar_datos()
    # 2. Preparar datos
    df = preparar_datos(df)
    # 3. Crear features
    x_features = crear_features(df)
    # 4. Seleccionar features
    X, y = seleccionar_features(df,x_features)
    # 5. Entrenar modelo
    modelo_best, preprocessor_best = entrenar_modelo(X, y, 0.2)  # <--- CAMBIO
    # 6. Guardar modelo (Necesita guardar ambos)
    guardar_modelo(modelo_best, preprocessor_best)  # <--

    print("\n[!] Implementa las funciones marcadas con TODO")
    print("[!] Luego ejecuta este script para entrenar tu modelo")


if __name__ == "__main__":
    main()
