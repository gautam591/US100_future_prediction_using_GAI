# US100 (Nasdaq-100) OHLCV → Candlestick Chart + Generative AI Future Scenarios

This package:
- Loads **daily OHLCV** for Nasdaq-100 index (`^NDX`) from `data/us100_ohlcv.csv` (US100 proxy),
- Builds a **candlestick chart** from recent OHLCV,
- Trains a **Conditional VAE (CVAE)** (a generative model) to learn future return distributions,
- Samples many **future scenarios** and plots them as future paths + 90% uncertainty band.

## Install
```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run end-to-end
```bash
python src/run_all.py
```

## Outputs
- `outputs/us100_forecast.png`
- `models/cvae_us100.pt`

> Thesis framing: outputs are **probabilistic scenarios**, not guaranteed exact future prices.
