#!/usr/bin/env python3
"""
infer.py - Inferência minimalista usando modelos Prophet treinados.

- Lê config.json para localizar models_dir e o arquivo extraído (CSV.gz).
- Carrega o modelo específico do equipamento se existir, senão usa o modelo "general".
- Gera previsões para horizon/resolution passados como parâmetros.
- Salva CSV comprimido do forecast e um PNG do gráfico (opcional).

Uso básico:
  Ajuste parâmetros em __main__ ou importe infer(...) de outro script.
  pip install pandas prophet joblib matplotlib
  python infer.py
"""

import os, json, math, joblib
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from pandas.tseries.frequencies import to_offset

# carregar config
with open("config.json", "r", encoding="utf-8") as f:
    cfg = json.load(f)

MODELS_DIR = cfg.get("models_defaults", {}).get("model_dir", "./models")
IMAGES_DIR = cfg.get("models_defaults", {}).get("images_dir", "./images")
EXT_OUT = cfg.get("extractor", {}).get("output", {})
INPUT_DIR = EXT_OUT.get("output_dir", "./dados")
BASENAME = EXT_OUT.get("output_basename", "product_extraction")
INPUT_PATH = os.path.join(INPUT_DIR, BASENAME + ".csv.gz")
META_INPUT_PATH = os.path.join(INPUT_DIR, BASENAME + ".csv.gz.meta.json")

INFER_CFG = cfg.get("infer", {})
FORECAST_DIR = INFER_CFG.get("forecast_output_dir", "./dados/forecasts")
GENERAL_MODEL_NAME = INFER_CFG.get("default_models", {}).get("general_name", "general_prophet.pkl")
SPECIFIC_TEMPLATE = INFER_CFG.get("default_models", {}).get("specific_template", "{slug}_prophet.pkl")

# util simples
def slugify(s):
    s = str(s or "")
    out = []
    for ch in s.strip():
        if ch.isalnum() or ch in ("_", "-"):
            out.append(ch)
        elif ch.isspace():
            out.append("_")
        else:
            out.append("_")
    return "".join(out) or "model"

# localizar modelo (específico -> geral)
def find_model(equipment_name):
    specific = os.path.join(MODELS_DIR, SPECIFIC_TEMPLATE.format(slug=slugify(equipment_name)))
    general = os.path.join(MODELS_DIR, GENERAL_MODEL_NAME)
    if equipment_name and os.path.exists(specific):
        return specific
    if os.path.exists(general):
        return general
    raise FileNotFoundError(f"Nenhum modelo encontrado: tentei '{specific}' e '{general}'")

# ler last_ds do meta do modelo, se existir
def model_last_ds(meta_path):
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as fm:
            mm = json.load(fm)
        if mm.get("last_ds"):
            return pd.to_datetime(mm.get("last_ds"))
    return None

# calcular periods a partir de horizon/resolution
def compute_periods(horizon, resolution):
    th = pd.Timedelta(horizon)
    off = to_offset(resolution)
    periods = int(math.ceil(th / off))
    return periods

# função principal de inferência
def infer(equipment_name=None, start=None, horizon="1H", resolution="15min", save_image=True, image_name=None):
    model_path = find_model(equipment_name)
    print(f"[{datetime.utcnow().isoformat()}Z] Carregando modelo: {model_path}")
    model = joblib.load(model_path)

    # tentar obter last_ds do meta do modelo
    meta_path = model_path + ".meta.json"
    last_ds = model_last_ds(meta_path)

    # carregar observados (para plot e pré-predição)
    obs = pd.DataFrame(columns=["ds","y"])
    if os.path.exists(INPUT_PATH):
        obs = pd.read_csv(INPUT_PATH, compression="gzip", parse_dates=[cfg.get("extractor", {}).get("train", {}).get("ds_column", "Date")])
        # normalizar nomes: Date -> ds, Value -> y
        colmap = {}
        for c in obs.columns:
            lc = c.strip().lower()
            if lc in ("date","ds","timestamp"):
                colmap[c] = "ds"
            if lc in ("value","valor","val"):
                colmap[c] = "y"
        obs = obs.rename(columns=colmap)
        if "ds" in obs.columns and "y" in obs.columns:
            obs = obs[["ds","y"]].dropna().sort_values("ds").reset_index(drop=True)
        else:
            obs = pd.DataFrame(columns=["ds","y"])

    # determinar start
    if start:
        start_ts = pd.to_datetime(start)
    else:
        start_ts = last_ds if last_ds is not None else (obs["ds"].max() if not obs.empty else pd.Timestamp.now())

    periods = compute_periods(horizon, resolution)

    # predição in-sample (histórico) - tentamos prever exatamente nas ds observadas
    hist_pred = None
    if not obs.empty:
        try:
            hist_pred = model.predict(obs[["ds"]].rename(columns={"ds":"ds"}))[["ds","yhat","yhat_lower","yhat_upper"]]
            hist_pred = hist_pred[hist_pred["ds"].isin(obs["ds"])]
        except Exception as e:
            print(f"Aviso: predição in-sample falhou: {e}")
            hist_pred = None

    # construir futuro iniciando em start + resolution
    res_off = to_offset(resolution)
    start_next = pd.to_datetime(start_ts) + res_off
    future_idx = pd.date_range(start=start_next, periods=periods, freq=resolution)
    future_df = pd.DataFrame({"ds": future_idx})
    fut_pred = model.predict(future_df)[["ds","yhat","yhat_lower","yhat_upper"]]

    # concatenar
    if hist_pred is not None:
        forecast_df = pd.concat([hist_pred, fut_pred], ignore_index=True).sort_values("ds").reset_index(drop=True)
    else:
        forecast_df = fut_pred.sort_values("ds").reset_index(drop=True)

    # salvar forecast (gzip)
    os.makedirs(FORECAST_DIR, exist_ok=True)
    fname = f"{slugify(equipment_name or 'general')}_forecast_{horizon}_{resolution}.csv.gz"
    out_path = os.path.join(FORECAST_DIR, fname)
    forecast_df.to_csv(out_path, index=False, compression="gzip")
    print(f"[{datetime.utcnow().isoformat()}Z] Forecast salvo: {out_path}")

    # plot simples
    plt.figure(figsize=(11,6))
    last_date = obs['ds'].max() if not obs.empty else forecast_df['ds'].min()
    is_future = forecast_df['ds'] > last_date
    is_past = ~is_future
    if is_past.any():
        plt.fill_between(forecast_df.loc[is_past,'ds'], forecast_df.loc[is_past,'yhat_lower'], forecast_df.loc[is_past,'yhat_upper'], color='lightgray', alpha=0.4)
    if is_future.any():
        plt.fill_between(forecast_df.loc[is_future,'ds'], forecast_df.loc[is_future,'yhat_lower'], forecast_df.loc[is_future,'yhat_upper'], color='salmon', alpha=0.35)
    if not obs.empty:
        plt.scatter(obs['ds'], obs['y'], color='black', s=18, label='Observado')
    plt.plot(forecast_df['ds'], forecast_df['yhat'], color='lightblue', linewidth=1, label='yhat')
    if is_future.any():
        plt.plot(forecast_df.loc[is_future,'ds'], forecast_df.loc[is_future,'yhat'], color='red', linewidth=2, label='Futuro')
    plt.xlabel("Data")
    plt.ylabel("Value")
    title = f"Forecast - {equipment_name or 'general'} - horizon {horizon} res {resolution}"
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_image:
        os.makedirs(IMAGES_DIR, exist_ok=True)
        img_name = image_name if (image_name := None) else f"forecast_{slugify(equipment_name or 'general')}_{horizon}_{resolution}.png"
        img_path = os.path.join(IMAGES_DIR, img_name)
        plt.savefig(img_path)
        print(f"[{datetime.utcnow().isoformat()}Z] Gráfico salvo: {img_path}")
    plt.show()

    return forecast_df

# exemplos de uso mínimo (gera 3 gráficos como no pipeline antigo)
if __name__ == "__main__":
    ex = INFER_CFG.get("examples", {})
    # 1h / 15min
    try:
        e1 = ex.get("hour", {"horizon":"1H","resolution":"15min"})
        infer(equipment_name=None, start=None, horizon=e1["horizon"], resolution=e1["resolution"], save_image=True)
    except Exception as err:
        print("Erro exemplo 1:", err)
    # 1w / hora
    try:
        e2 = ex.get("week", {"horizon":"7D","resolution":"H"})
        infer(equipment_name=None, start=None, horizon=e2["horizon"], resolution=e2["resolution"], save_image=True)
    except Exception as err:
        print("Erro exemplo 2:", err)
    # 1m / dia
    try:
        e3 = ex.get("month", {"horizon":"30D","resolution":"D"})
        infer(equipment_name=None, start=None, horizon=e3["horizon"], resolution=e3["resolution"], save_image=True)
    except Exception as err:
        print("Erro exemplo 3:", err)