#!/usr/bin/env python3
"""
infer_minimal.py
Inferência Prophet ultra minimalista:
- Ajuste as variáveis no topo.
- Executa e gera um CSV com previsões futuras.

Requisitos:
  pip install pandas prophet joblib
Uso:
  python infer_minimal.py
"""

import joblib
import os
import pandas as pd
import math
from pandas.tseries.frequencies import to_offset

# ------------------ CONFIGURAÇÃO MÍNIMA ------------------
MODEL_PATH = "./models/general_prophet.pkl"   # ou modelo específico ex: ./models/Medidor_de_Energia_P9_QDFI_prophet.pkl
HORIZON = "1H"        # exemplo: "1H", "7D", "30D"
RESOLUTION = "15min"  # exemplo: "15min", "H", "D"
OUTPUT_CSV = "./dados/forecasts/forecast_minimal.csv"
# ----------------------------------------------------------

print(f"Carregando modelo: {MODEL_PATH}")
model = joblib.load(MODEL_PATH)

history = model.history.copy()
if "ds" not in history.columns:
    raise ValueError("Histórico do modelo não contém coluna 'ds'.")
last_ds = history["ds"].max()
print(f"Última data do histórico no modelo: {last_ds}")

# Normalizar entradas para evitar FutureWarning (usar minúsculas)
horizon_norm = str(HORIZON).strip().lower()
resolution_norm = str(RESOLUTION).strip().lower()

# Calcular número de períodos
total_horizon = pd.Timedelta(horizon_norm)
freq_offset = to_offset(resolution_norm)
periods = int(math.ceil(total_horizon / freq_offset))
if periods <= 0:
    raise ValueError("HORIZON e RESOLUTION resultaram em periods <= 0.")

# Construir datas futuras (inicia após a última observação)
start_future = last_ds + freq_offset
future_ds = pd.date_range(start=start_future, periods=periods, freq=resolution_norm)
future_df = pd.DataFrame({"ds": future_ds})

print(f"Gerando previsão futura: horizon={HORIZON} (periods={periods}) resolution={RESOLUTION}")
pred = model.predict(future_df)[["ds", "yhat", "yhat_lower", "yhat_upper"]]

# Garantir diretório de saída
out_dir = os.path.dirname(OUTPUT_CSV)
if out_dir and not os.path.exists(out_dir):
    os.makedirs(out_dir, exist_ok=True)

pred.to_csv(OUTPUT_CSV, index=False)
print(f"Forecast salvo em: {OUTPUT_CSV}")
print("Amostra final:")
print(pred.tail().to_string(index=False))