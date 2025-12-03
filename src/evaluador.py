"""
RPSAI - Evaluador de Winrate
============================

Este script evalua el rendimiento de tu modelo de IA
jugando partidas contra un humano y calculando el winrate.

Uso:
    python src/evaluador.py

El evaluador:
1. Carga tu modelo entrenado
2. Juega N partidas contra un humano
3. Calcula y muestra el winrate final
4. Muestra la nota segun los criterios de evaluacion
"""

import sys
from pathlib import Path
import random # Necesario para fallback aleatorio

# Agregar el directorio src al path para importar modelo
sys.path.insert(0, str(Path(__file__).parent))

# Importa las clases y constantes del módulo modelo.py
# El modelo corregido debe estar en el mismo directorio (src)
try:
    from modelo import JugadorIA, JUGADA_A_NUM, NUM_A_JUGADA, GANA_A
except ImportError:
    print("ERROR: No se pudo importar 'modelo.py'. Asegúrate de que el archivo existe en src/.")
    sys.exit(1)


# Mapeo de entrada a jugada
ENTRADA_A_JUGADA = {
    "1": "piedra", "2": "papel", "3": "tijera",
    "p": "piedra", "a": "papel", "t": "tijera",
    "piedra": "piedra", "papel": "papel", "tijera": "tijera"
}


def obtener_nota(winrate: float) -> tuple:
    """Calcula la nota basada en el winrate."""
    if winrate >= 55:
        return 10, "BONUS: Fallos en examen no restan!"
    elif winrate >= 50:
        return 10, None
    elif winrate >= 49:
        return 9, None
    elif winrate >= 48:
        return 8, None
    elif winrate >= 46:
        return 7, None
    elif winrate >= 44:
        return 6, None
    elif winrate >= 42:
        return 5, None
    elif winrate >= 40:
        return 4, None
    elif winrate >= 39:
        return 3, None
    elif winrate >= 37:
        return 2, None
    elif winrate >= 35:
        return 1, None
    else:
        return 0, None


def obtener_resultado(jugada_ia: str, jugada_humano: str) -> str:
    """Obtiene el resultado desde la perspectiva de la IA."""
    if jugada_ia == jugada_humano:
        return "empate"
    # Si la jugada de la IA gana a la jugada del humano
    elif GANA_A[jugada_ia] == jugada_humano:
        return "victoria"
    # En cualquier otro caso, el humano gana a la IA
    else:
        return "derrota"


def leer_jugada_humano() -> str:
    """Lee la jugada del humano."""
    while True:
        print("\n[1/p] Piedra  [2/a] Papel  [3/t] Tijera")
        entrada = input("Tu jugada: ").lower().strip()

        if entrada in ENTRADA_A_JUGADA:
            return ENTRADA_A_JUGADA[entrada]

        print("Jugada no valida. Intenta de nuevo.")


def mostrar_ronda(ronda: int, jugada_ia: str, jugada_humano: str, resultado: str):
    """Muestra el resultado de una ronda con emojis."""
    simbolos = {"piedra": "🪨", "papel": "📃", "tijera": "✂️"}

    print(f"\n--- Ronda {ronda} ---")
    print(f"🧑 Tu: {simbolos.get(jugada_humano, '❓')} ({jugada_humano})")
    print(f"🤖 IA: {simbolos.get(jugada_ia, '❓')} ({jugada_ia})")

    if resultado == "victoria":
        print(">>> 🏆 IA GANA <<<")
    elif resultado == "derrota":
        print(">>> 🎉 Tu ganas <<<")
    else:
        print(">>> 🤝 Empate <<<")


def mostrar_progreso(victorias: int, derrotas: int, empates: int, total: int):
    """Muestra el progreso actual."""
    jugadas = victorias + derrotas + empates
    restantes = total - jugadas

    # Solo calculamos winrate si hay jugadas decisivas (no empates)
    total_decisivas = victorias + derrotas
    if total_decisivas > 0:
        winrate_actual = victorias / total_decisivas * 100
        print(f"\n[Progreso: {jugadas}/{total}] "
              f"IA: {victorias}V-{derrotas}D-{empates}E "
              f"(Winrate: {winrate_actual:.1f}%) "
              f"| Quedan: {restantes}")
    else:
        # Muestra 0% al inicio
        print(f"\n[Progreso: {jugadas}/{total}] IA: 0V-0D-0E (Winrate: 0.0%) | Quedan: {restantes}")


def evaluar(num_rondas: int = 50):
    """
    Ejecuta la evaluacion del modelo.
    """
    print("="*60)
    print("   RPSAI - EVALUACION DE WINRATE")
    print("="*60)
    print(f"\nSe jugaran {num_rondas} rondas contra tu modelo de IA.")
    print("Juega de forma natural, como lo harias normalmente.\n")

    ia = None
    try:
        ia = JugadorIA()
        if ia.modelo is None:
            print("[!] ADVERTENCIA: No se cargo ningun modelo (modelo.pkl no encontrado).")
            print("[!] La IA jugara de forma ALEATORIA.")
            print("[!] Entrena tu modelo primero con: python src/modelo.py\n")
            raise RuntimeError("Modelo no cargado.")
    except Exception as e:
        # En caso de error crítico (ej. FileNotFoundError en el init), usar un fallback
        print(f"[!] Error al inicializar JugadorIA o al cargar el modelo: {e}")
        print("[!] Usando Fallback IA aleatoria.\n")
        class FallbackIA:
            def decidir_jugada(self): return random.choice(["piedra", "papel", "tijera"])
            def registrar_ronda(self, j1, j2): pass
        ia = FallbackIA()


    input("Presiona ENTER para comenzar la evaluacion...")

    victorias = 0
    derrotas = 0
    empates = 0

    for ronda in range(1, num_rondas + 1):
        # La IA decide su jugada (usando el modelo o aleatorio si no carga)
        jugada_ia = ia.decidir_jugada()

        # El humano juega
        jugada_humano = leer_jugada_humano()

        # Determinar resultado (desde perspectiva IA)
        resultado = obtener_resultado(jugada_ia, jugada_humano)

        # Mostrar resultado
        mostrar_ronda(ronda, jugada_ia, jugada_humano, resultado)

        # Registrar en el historial de la IA
        # IMPORTANTE: El humano es J1, la IA es J2, esto alimenta el historial para la próxima ronda.
        ia.registrar_ronda(jugada_ia, jugada_humano)

        # Actualizar contadores
        if resultado == "victoria":
            victorias += 1
        elif resultado == "derrota":
            derrotas += 1
        else:
            empates += 1

        # Mostrar progreso
        mostrar_progreso(victorias, derrotas, empates, num_rondas)

    # Resultados finales
    print("\n" + "="*60)
    print("   RESULTADOS FINALES")
    print("="*60)

    total_decisivas = victorias + derrotas
    if total_decisivas > 0:
        winrate = victorias / total_decisivas * 100
    else:
        winrate = 0

    print(f"\n🎯 Rondas jugadas: {num_rondas}")
    print(f"🤖 Victorias IA: {victorias}")
    print(f"🧑 Derrotas IA: {derrotas}")
    print(f"🤝 Empates: {empates}")
    print(f"\n📊 WINRATE DE LA IA: {winrate:.1f}%")

    nota, bonus = obtener_nota(winrate)
    print(f"\n{'='*60}")
    print(f"   NOTA MODO 1 (Winrate): {nota}/10")
    if bonus:
        print(f"   {bonus}")
    print(f"{'='*60}")

    # Tabla de referencia
    print("\nTabla de referencia:")
    print("  33%- = 0  |  35% = 1  |  37% = 2  |  39% = 3")
    print("  40% = 4   |  42% = 5  |  44% = 6  |  46% = 7")
    print("  48% = 8   |  49% = 9  |  50%+ = 10 | 55%+ = BONUS")


def main():
    """Funcion principal."""
    import argparse

    parser = argparse.ArgumentParser(description="Evalua el winrate de tu modelo de IA")
    parser.add_argument("-n", "--rondas", type=int, default=50,
                        help="Numero de rondas a jugar (default: 50)")
    args = parser.parse_args()

    evaluar(args.rondas)


if __name__ == "__main__":
    main()