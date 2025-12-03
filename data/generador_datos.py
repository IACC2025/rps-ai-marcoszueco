import csv
import time
import keyboard  # pip install keyboard


controles_j1 = {"a": "piedra", "s": "papel", "d": "tijera"}
controles_j2 = {"4": "piedra", "5": "papel", "6": "tijera"}

def ganador(j1, j2):
    if j1 == j2:
        return "Empate"
    elif (j1 == "piedra" and j2 == "tijera") or \
         (j1 == "papel" and j2 == "piedra") or \
         (j1 == "tijera" and j2 == "papel"):
        return "Jugador 1"
    else:
        return "Jugador 2"

print("Piedra, Papel o Tijera simultáneo!")
print("Jugador 1 usa a/s/d   -> piedra/papel/tijera")
print("Jugador 2 usa 4/5/6   -> piedra/papel/tijera")
print()

rondas = int(input("¿Cuántas rondas quieren jugar? "))


with open("resultados1.csv", "w", newline="") as archivo:
    escritor = csv.writer(archivo)
    escritor.writerow(["num_ronda", "jugada_jugador", "jugada_oponente"])

    puntos1 = 0
    puntos2 = 0

    for r in range(1, rondas + 1):
        print(f"\n--- Ronda {r} ---")
        print("Preparados...")
        time.sleep(1)
        print("3...")
        time.sleep(1)
        print("2...")
        time.sleep(1)
        print("1...")
        time.sleep(1)
        print("¡YA! Pulsen sus teclas!")

        tecla1 = None
        tecla2 = None


        while tecla1 not in controles_j1 or tecla2 not in controles_j2:
            for t in controles_j1:
                if keyboard.is_pressed(t):
                    tecla1 = t
            for t in controles_j2:
                if keyboard.is_pressed(t):
                    tecla2 = t

        j1 = controles_j1[tecla1]
        j2 = controles_j2[tecla2]
        g = ganador(j1, j2)

        print(f"Jugador 1 -> {j1}")
        print(f"Jugador 2 -> {j2}")
        print("Resultado:", g)

        if g == "Jugador 1":
            puntos1 += 1
        elif g == "Jugador 2":
            puntos2 += 1

        escritor.writerow([r, j1, j2])
        time.sleep(1)

print("\n--- FINAL ---")
print("Jugador 1:", puntos1)
print("Jugador 2:", puntos2)

if puntos1 > puntos2:
    print("Ganador final: Jugador 1")
elif puntos2 > puntos1:
    print("Ganador final: Jugador 2")
else:
    print("¡Empate!")

print("\nResultados guardados en resultados1.csv")
