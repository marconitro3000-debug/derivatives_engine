# Comandos: todo lo que se puede ejecutar en este repo

Referencia completa de **cada** cosa ejecutable: los cuatro modos de `main.py`,
todos sus flags, los tests, el notebook y la API de Python. Si algo se puede
correr, está aquí y dice qué hace.

Regla general: `python main.py` sin argumentos abre el menú y no cambia nada de
lo que había antes. Cada opción del menú es también un comando, y **cada knob de
[`config.py`](../config.py) es también un flag** — la lista de flags la genera
[`cli.py`](../cli.py) leyendo la propia dataclass, así que no puede quedarse
desincronizada.

```bash
python main.py --help        # la lista completa, generada
```

---

## Instalación

| Comando | Qué hace |
|---|---|
| `pip install -e .` | Instala el paquete en modo editable con sus dependencias (torch, numpy, scipy, pandas, matplotlib, yfinance). Es el único paso de setup. |
| `pip install -e ".[dev]"` | Lo mismo más `pytest`, si vas a correr los tests. |
| `python -c "import volsurface; print(volsurface.__file__)"` | Comprueba que el paquete se importa. |

---

## Los cuatro modos

### `python main.py` — el menú

Pregunta qué hacer y luego qué subyacente. Sin argumentos, sin nada que
recordar. Es la vía por defecto y sigue funcionando exactamente igual que antes
de que existiera la CLI.

```
  [1]  Train a surface        fit a chain, score it, plot everything
  [2]  Price off a surface    query the one trained last time
  [3]  Compare trained models which archived run to use, and why
  [4]  Sweep architectures    is the network too small? fit several
                              sizes on one chain and table the answer
```

### `python main.py train` — ajustar una superficie

Descarga la cadena viva, la limpia, calibra los baselines paramétricos, entrena
la red bajo las penalizaciones de no-arbitraje, entrena también la ablación de
prior plano, puntúa las cuatro superficies con el mismo código y escribe todo.

**Produce:** `results/report.txt`, `results/<ticker>_surface.pt`,
`results/<ticker>_chain.npz`, `results/training.png`, `results/smiles.png`, un
`results/arbitrage_*.png` por superficie, y un archivo completo de la corrida en
`models/<ticker>_<timestamp>/` (pesos, cadena, historia, `metrics.json`).

**Tarda** unos 3-5 minutos: la descarga y la de-americanización son la mitad, y
la calibración de SVI por slice es el paso más lento del scoring.

```bash
python main.py train                              # SPY, todo por defecto
python main.py train --ticker AAPL                # otro subyacente
python main.py train --epochs 3000 --hidden 128x4 # otra red
python main.py train --no-run-baselines           # rápido: sólo la red
python main.py train --no-use-live-data           # cadena sintética, sin red
python main.py train --chain-file results/spy_chain.npz   # reajustar la de ayer
python main.py train --ticker QQQ --seed 7 --no-show-plots
```

### `python main.py price` — valorar con la superficie entrenada

Carga `results/<ticker>_surface.pt` (instantáneo, sin red) y responde a strike +
vencimiento con vol implícita, precio, griegas y el chequeo local de
no-arbitraje, más la sonrisa con tu strike marcado para que veas si la respuesta
es interpolación entre cotizaciones reales o extrapolación.

```bash
python main.py price                                    # interactivo
python main.py price --ticker AAPL
python main.py price --price-maturities 0.25,0.5,1.0    # no interactivo
python main.py price --price-strikes 700,750,800 --price-maturities 0.5
```

Sin nada escuchando en stdin (un cron, un pipe) usa `price_maturities` y
`price_strikes`; en una sesión interactiva ésos son sólo los valores por defecto
del primer prompt.

### `python main.py compare` — elegir entre corridas archivadas

Lee todo `models/`, las ordena por error de validación y superpone sus curvas de
entrenamiento. Es la respuesta a "¿qué checkpoint uso?" cuando llevas quince
experimentos.

```bash
python main.py compare
python main.py compare --ticker AAPL
python main.py compare --models-dir models_backup
```

### `python main.py sweep` — ¿es la red demasiado pequeña?

Ajusta varias arquitecturas **a la misma cadena**, con el mismo split, el mismo
presupuesto de épocas, la misma semilla y las mismas penalizaciones. Lo único
que cambia es la forma de `net`. Saca error de validación contra número de
parámetros, que es la curva que decide si la capacidad es la restricción que
manda o si lo es el bid-ask.

**Produce:** `results/sweep.txt`, `results/sweep.png` y
`results/<ticker>_sweep_chain.npz` — esta última es la cadena exacta que vieron
todos los brazos, para que cualquier arm se pueda repetir con `--chain-file`.

```bash
python main.py sweep                                     # la rejilla por defecto
python main.py sweep --sweep-architectures 64x3,256x4,1024x4
python main.py sweep --sweep-seeds 3                     # 3 semillas por arquitectura
python main.py sweep --chain-file results/spy_chain.npz  # sobre una cadena fija
python main.py sweep --epochs 500 --sweep-architectures 32x2,64x3  # rápido
```

La cadena se descarga **una vez** y se reutiliza. Darle a cada brazo su propia
descarga confundiría la arquitectura con lo que hizo el mercado en esos minutos,
que en una cadena viva es fácilmente mayor que el efecto que se quiere medir. El
split también se fija: si la semilla moviera además qué cotizaciones se retienen,
las semillas no serían comparables entre sí y esa dispersión se comería el efecto
que el sweep existe para medir.

Resultado de la corrida por defecto (18 ajustes, en
[`docs/example_sweep.txt`](example_sweep.txt)): **360x más parámetros compran
2.1bp**, un 4.7% del error, por ~40x el tiempo de entrenamiento — y la dispersión
entre semillas de una misma arquitectura es de 1 a 4bp. Detalle y lectura en
[`docs/RED_NEURONAL.md`](RED_NEURONAL.md), sección 6.

---

## Todos los flags

Uno por campo de `RunConfig`. Los booleanos aceptan `--x` y `--no-x`. Las tuplas
van separadas por comas. `--hidden` acepta `256x4` (cuatro capas de 256) o
`64,64,64`.

### Qué hacer

| Flag | Por defecto | Qué hace |
|---|---|---|
| `mode` (posicional) | menú | `train`, `price`, `compare` o `sweep`. |
| `--show-plots / --no-show-plots` | `True` | Abrir ventanas además de guardar los PNG. Siempre off si nadie está mirando. |

### Qué ajustar

| Flag | Por defecto | Qué hace |
|---|---|---|
| `--ticker` | `SPY` | Subyacente. ETFs líquidos y large caps funcionan; los ilíquidos no sobreviven los filtros de cotización. |
| `--use-live-data / --no-use-live-data` | `True` | Descargar de Yahoo Finance. Si falla, cae a una cadena generada y lo dice en voz alta. |
| `--chain-file PATH` | ninguno | Ajustar un `*_chain.npz` ya descargado en vez de bajar uno. Manda sobre `--use-live-data`. **Es lo que hace reproducible un experimento.** |
| `--max-expiries` | `8` | Vencimientos, muestreados uniformemente en `sqrt(T)` sobre toda la curva, no tomados del frente. |
| `--min-maturity-years` | `0.02` | Nada más corto: lo domina el tick size. |
| `--max-maturity-years` | `2.0` | Nada más largo: apenas cotiza. |

### Filtros de cotización

| Flag | Por defecto | Qué hace |
|---|---|---|
| `--min-open-interest` | `10` | Interés abierto mínimo. |
| `--max-relative-spread` | `0.25` | Una cotización más ancha que el 25% de su propio mid no lleva volatilidad usable. |
| `--de-americanize / --no-de-americanize` | `True` | Quitar la prima de ejercicio anticipado en árbol binomial. Apagarlo cobra esa prima a la volatilidad: ~13bp de media en SPY, ~25bp más allá del año, casi todo en los puts. |
| `--lattice-steps` | `150` | Pasos del árbol. La prima es una diferencia de dos precios del mismo árbol, así que el error de discretización se cancela. |

### La red

| Flag | Por defecto | Qué hace |
|---|---|---|
| `--epochs` | `1500` | Épocas de Adam full-batch. |
| `--hidden` / `--hidden-layers` | `64,64,64` | Anchos de las capas ocultas. `256x4` = cuatro de 256. |
| `--alpha` | `0.35` | Corrección relativa máxima al prior SSVI. 0.35 en varianza total es ~16% en vol: es la cota de error del modelo, fijada antes de entrenar. |
| `--learning-rate` | `0.003` | LR inicial de Adam, con decaimiento coseno. |
| `--validation-fraction` | `0.2` | Fracción de strikes retenida **dentro de cada vencimiento**, cubriendo toda la sonrisa. |
| `--seed` | `0` | Semilla de los pesos, del split y de los puntos de colocación. |

### Penalizaciones de no-arbitraje

| Flag | Por defecto | Qué hace |
|---|---|---|
| `--calendar-weight` | `10.0` | Peso sobre `relu(-dw/dT)^2`. |
| `--butterfly-weight` | `1.0` | Peso sobre `relu(-g)^2`, con `g` la función de Durrleman. |
| `--collocation-points` | `2048` | Puntos por época donde se imponen las restricciones, sorteados en una región más ancha que las cotizaciones y remuestreados en cada paso. |

### Qué producir

| Flag | Por defecto | Qué hace |
|---|---|---|
| `--output-dir` | `results` | El *cursor*: se sobrescribe en cada train, es de donde lee `price`. |
| `--models-dir` | `models` | El *archivo*: una carpeta por corrida, nunca se sobrescribe, es lo que compara `compare`. |
| `--make-plots / --no-make-plots` | `True` | Escribir los PNG. |
| `--save-model / --no-save-model` | `True` | Guardar pesos y archivar la corrida. |
| `--run-baselines / --no-run-baselines` | `True` | Calibrar SVI y SSVI para la comparación. Es el paso más lento; apagarlo es la forma de iterar rápido. |
| `--run-prior-ablation / --no-run-prior-ablation` | `True` | Reentrenar la misma red contra un prior de vol constante y meterla en la tabla. Es el experimento que dice cuánto del resultado es el prior y cuánto la red. Cuesta un segundo entrenamiento. |
| `--verbose / --no-verbose` | `True` | Imprimir el log de entrenamiento. |

### Sweep

| Flag | Por defecto | Qué hace |
|---|---|---|
| `--sweep-architectures` | `32x2,64x3,128x3,256x4,512x4` | Arquitecturas a comparar, en `ANCHOxPROFUNDIDAD`. |
| `--sweep-seeds` | `1` | Semillas por arquitectura. Una basta para ver la forma de la curva; tres es lo que quieres antes de afirmar que dos arquitecturas realmente difieren, porque la dispersión entre corridas en una cadena real es de unos pocos puntos básicos. |

### Precio no interactivo

| Flag | Por defecto | Qué hace |
|---|---|---|
| `--price-maturities` | `0.25,0.5,1.0` | Vencimientos (años) a valorar cuando nadie escucha en stdin. |
| `--price-strikes` | vacío | Strikes a valorar. Vacío = escalera alrededor del forward de cada vencimiento. |

---

## Tests

```bash
pytest                            # los 128, sin red
pytest -q                         # sin ruido
pytest tests/test_neural_surface.py -v          # sólo la red
pytest -k "arbitrage or feasible" -v            # por nombre
pytest --basetemp=/tmp/pt         # si tu %TEMP% da PermissionError en Windows
```

Qué cubre cada archivo:

| Archivo | Qué fija |
|---|---|
| [`tests/test_neural_surface.py`](../tests/test_neural_surface.py) | El prior torch reproduce el numpy; el modelo sin entrenar *es* el prior; la corrección respeta `alpha`; autograd coincide con diferencias finitas; el caché de derivadas devuelve lo correcto; construir un modelo no toca el RNG global; el split llega a las alas; la paciencia no se gasta en el warm-up; el epoch elegido es factible; historia y pesos describen el mismo epoch. |
| [`tests/test_black_scholes.py`](../tests/test_black_scholes.py) | Paridad put-call, griegas contra diferencias finitas, casos límite. |
| [`tests/test_marketdata.py`](../tests/test_marketdata.py) | La inversión de IV se niega cuando la vol no es identificable; forward y descuento implícitos por paridad; filtros de cotización; de-americanización. |
| [`tests/test_numerical_pricers.py`](../tests/test_numerical_pricers.py) | Árbol binomial y Monte Carlo convergen a Black-Scholes. |
| [`tests/test_baselines.py`](../tests/test_baselines.py) | Calibración de SVI y SSVI, y las condiciones de no-arbitraje de cada una. |

---

## El notebook

```bash
jupyter lab notebook.ipynb        # o jupyter notebook
```

El mismo estudio escrito como un paper, cada afirmación respaldada por una celda
ejecutable, commiteado con sus salidas para que se renderice entero en GitHub.
No hace falta ejecutarlo para leerlo.

---

## Como librería

Todo lo que hace `main.py` está disponible como API. Referencia completa en
[`docs/USAGE.md`](USAGE.md).

```python
from volsurface import fetch_chain, train_surface, compare, comparison_table

chain  = fetch_chain("SPY")                    # descarga + limpieza + de-americanización
result = train_surface(chain)                  # entrena la red
print(comparison_table(compare(chain, result)))
print(result.model.implied_vol([-0.1, 0.0, 0.1], [0.5, 0.5, 0.5]))
```

Piezas sueltas que quizá quieras por separado:

```python
from volsurface import synthetic_snapshot, scan_arbitrage, fit_report
from volsurface.svi import SVISliceSurface, SSVISurface
from volsurface.neural import TrainConfig, ModelConfig, PenaltyWeights

chain = synthetic_snapshot(noise_bps=25.0)     # cadena generada, sin red
svi   = SVISliceSurface.fit(chain)             # el baseline flexible
ssvi  = SSVISurface.fit(chain)                 # el baseline seguro
print(scan_arbitrage(svi))                     # escaneo en malla densa
print(fit_report(ssvi, chain))                 # error en vol y en precio

cfg = TrainConfig(epochs=3000, model=ModelConfig(hidden=(256,)*4, alpha=0.25))
cfg.penalties = PenaltyWeights(calendar=10.0, butterfly=5.0, n_points=4096)
result = train_surface(chain, cfg)
ablation = train_surface(chain, cfg, prior="flat")   # sin el prior SSVI
```

Recargar una corrida archivada sin volver a entrenar:

```python
from volsurface import list_runs, load_run, load_history

runs = list_runs("models", "SPY")              # ordenadas, la más reciente primero
model, chain = load_run(runs[0])
history = load_history(runs[0])                # curvas por época
```

---

## Recetas

**Iterar rápido sobre la red sin volver a descargar ni recalibrar SVI:**
```bash
python main.py train --chain-file results/spy_chain.npz --no-run-baselines \
                     --no-run-prior-ablation --epochs 600 --no-make-plots
```

**Comprobar que un cambio no rompió nada, sin tocar la red:**
```bash
python main.py train --no-use-live-data --epochs 300 --output-dir /tmp/check
```
La cadena sintética viene de una superficie SSVI libre de arbitraje conocida, así
que cualquier violación que reporten los diagnósticos sobre ella es un defecto
del *modelo*, no una característica del mercado.

**Repetir exactamente una corrida archivada:**
```bash
python main.py train --chain-file models/spy_20260905T000741/chain.npz --seed 0
```

**Barrer la fuerza de las restricciones en vez de la arquitectura:**
```bash
for w in 1 5 20; do
  python main.py train --chain-file results/spy_chain.npz --butterfly-weight $w \
                       --no-run-baselines --no-run-prior-ablation \
                       --output-dir results/bfly_$w
done
```

**Correrlo desatendido (cron, tarea programada):** con nada conectado a stdin,
`main.py` no pregunta nada y usa la configuración tal cual.
```bash
python main.py train --ticker SPY --no-show-plots
```

---

## Dónde acaba cada cosa

```
results/                    el cursor - sobrescrito en cada train
    report.txt              la corrida entera, tal cual salió por consola
    sweep.txt / sweep.png   el estudio de capacidad
    <ticker>_surface.pt     pesos + prior + estadísticos de features
    <ticker>_chain.npz      la cadena limpia que vio el modelo
    training.png            fit, gap de generalización, violaciones, penalizaciones
    smiles.png              las tres superficies contra las cotizaciones
    arbitrage_*.png         g y dw/dT en todo el plano, una por superficie

models/<ticker>_<stamp>/    el archivo - nunca se sobrescribe
    surface.pt  chain.npz  history.npz  metrics.json
```

`results/` es el cursor y `models/` es la historia. `price` lee del primero;
`compare` lee del segundo.
