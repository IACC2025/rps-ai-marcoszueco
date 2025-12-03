# Documentacion de Recogida de Datos

**Alumno:** Marcos Zueco Llera y Carlos Ramirez Garcia

## Formato del CSV

Tu archivo `data/resultados.csv` debe tener **minimo** estas columnas:

| Columna           | Descripcion | Ejemplo |
|-------------------|-------------|---------|
| `num_ronda`       | Numero de la ronda (1, 2, 3...) | 1 |
| `jugada_jugador`  | Jugada del jugador 1 | piedra |
| `jugada_oponente` | Jugada del jugador 2 (oponente) | papel |

### Ejemplo de CSV minimo:

```csv
num_ronda,jugada_jugador,jugada_oponente
1,piedra,papel
2,tijera,piedra
3,papel,papel
4,piedra,tijera
...
```

---

## Como recogiste los datos?

Marca con [x] el metodo usado y describe brevemente:

### Metodo de recogida:

- [x] **Programa propio**: Cree un programa para jugar y guardar datos
- [ ] **Manual**: Jugue partidas y apunte los resultados a mano
- [ ] **Aplicacion/Web externa**: Use una app y exporte los datos
- [ ] **Otro**: _________________

### Descripcion del proceso:

```
(Explica aqui como recogiste los datos. Si usaste un programa,
describe brevemente como funciona. Si fue manual, explica el proceso.)
Hemos usado un programa en el que indicamos cuantas rondas queremos jugar 
y jugamos  todas rondas seguidas, cuando terminan se guardan los resultados en un archivo .csv




```
---

## Datos adicionales capturados

Si capturaste datos extra ademas de los basicos, marcalos aqui:

- [ ] `tiempo_reaccion_ms` - Tiempo que tardo el jugador en responder
- [ ] `timestamp` - Fecha/hora de cada jugada
- [ ] `sesion` - ID de sesion de juego
- [ (este se genera en el propio codigo como feature)] `resultado` - victoria/derrota/empate
- [ ] Otro: _________________

### Descripcion de datos adicionales:

```
(Si capturaste datos extra, explica aqui por que y como los usas)


```

---

## Estadisticas del dataset

- **Total de rondas:** 500
- **Numero de sesiones/partidas:** 1 (hemos jugado otras pero nos hemos quedado con la tirada de 500)
- **Contra cuantas personas diferentes:** 1

### Tipo de IA:

- [ x] **IA Especifica**: Entrenada para ganar a UNA persona concreta
  - Nombre/identificador del oponente: Carlos Ramirez
- [ ] **IA General**: Entrenada para ganar a cualquier oponente
