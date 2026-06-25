# Commands

## Pricing

```bash
python price_option.py TICKER --strike K --expiry T --model MODEL --type call|put
```

`--expiry` en años. Ejemplos: `0.083` = 1 mes, `0.25` = 3 meses, `0.5` = 6 meses, `1.0` = 1 año.

`--model` puede ser: `all` (default), `black-scholes`, `heston`, `svi`, `monte-carlo`, `binomial`.

Por defecto coge el spot y la IV ATM en vivo de Yahoo Finance. Se pueden forzar con `--spot` y `--vol`.

---

### Ejemplos

Precio de una call de AAPL con todos los modelos:
```bash
python price_option.py AAPL --strike 185 --expiry 0.25
```

Solo Heston (calibra a la chain real de SPY antes de pricear, tarda ~40-60s):
```bash
python price_option.py SPY --strike 550 --expiry 0.5 --model heston
```

SVI (calibra por maturity slice, tarda ~1s):
```bash
python price_option.py TSLA --strike 250 --expiry 1.0 --model svi
```

Black-Scholes analítico con la IV del mercado:
```bash
python price_option.py NVDA --strike 900 --expiry 0.25 --model black-scholes
```

Monte Carlo (100k paths GBM):
```bash
python price_option.py GS --strike 520 --expiry 0.75 --model monte-carlo
```

Binomial CRR 500 pasos, europeo y americano:
```bash
python price_option.py MSFT --strike 420 --expiry 0.5 --model binomial
```

Put en vez de call:
```bash
python price_option.py AAPL --strike 185 --expiry 0.25 --type put --model heston
```

Forzar vol del 18% en vez de coger la del mercado:
```bash
python price_option.py SPY --strike 550 --expiry 0.5 --vol 0.18
```

Forzar spot y vol manualmente (sin red):
```bash
python price_option.py SPY --strike 550 --expiry 0.5 --spot 540 --vol 0.20
```

---

## Tests

```bash
pytest                            # los 61 tests
pytest tests/test_pricer.py       # BS, IV, MC, binomial, vol surface
pytest tests/test_calibration.py  # calibracion y engine
pytest -k "heston"                # filtrar por nombre
```

## Install

```bash
pip install -e ".[dev]"   # instalacion editable con pytest
pip install -e .           # solo runtime
```
