#!/usr/bin/env python3
"""
infer.py - Inferência minimalista usando modelos Prophet treinados.

- Lê config.json para localizar models_dir e o arquivo extraído (CSV.gz).
- Carrega o modelo específico do equipamento se existir, senão usa o modelo "general".
- Gera previsões para horizon/resolution passados como parâmetros.
- Salva CSV comprimido do forecast (futuro + histórico exibido) e um PNG do gráfico (opcional).
- A janela exibida é limitada conforme o horizonte:
    * <=1H  -> últimas 24H (a janela termina no último timestamp projetado)
    * <=7D  -> últimas 2 semanas (14D)
    * <=30D -> últimos 3 meses (~90D)

Alteração solicitada: exibir rótulos de hora com sufixo "h" (ex: "14h") quando a janela for curta / quando for sensible mostrar horas.
Além disso: se equipment_name for None, tenta detectar EquipmentName único no CSV (conforme extra_columns em config) e usar o modelo específico.

Uso:
  pip install pandas prophet joblib matplotlib
  python infer.py
"""

import os, json, math, joblib
from datetime import datetime, timezone
import pandas as pd
import matplotlib.pyplot as plt
from pandas.tseries.frequencies import to_offset
import matplotlib.dates as mdates

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

def find_model(equipment_name):
    specific = os.path.join(MODELS_DIR, SPECIFIC_TEMPLATE.format(slug=slugify(equipment_name)))
    general = os.path.join(MODELS_DIR, GENERAL_MODEL_NAME)
    if equipment_name and os.path.exists(specific):
        return specific
    if os.path.exists(general):
        return general
    raise FileNotFoundError(f"Nenhum modelo encontrado: tentei '{specific}' e '{general}'")

def model_last_ds(meta_path):
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as fm:
                mm = json.load(fm)
            if mm.get("last_ds"):
                return pd.to_datetime(mm.get("last_ds"))
        except Exception:
            pass
    return None

def compute_periods(horizon, resolution):
    # Normalizar strings para evitar FutureWarning
    horizon_norm = str(horizon).strip().lower()
    resolution_norm = str(resolution).strip().lower()
    th = pd.Timedelta(horizon_norm)
    off = to_offset(resolution_norm)
    return int(math.ceil(th / off))

def infer(equipment_name=None, start=None, horizon="1H", resolution="15min", save_image=True, image_name=None):
    """
    Se equipment_name for None, tenta detectar EquipmentName único no CSV (conforme config.extractor.train.extra_columns).
    Depois carrega o modelo (específico se existir, senão general) e segue com a previsão.
    """
    # carregar observados (mantendo extras configuradas)
    obs = pd.DataFrame(columns=["ds","y"])
    if os.path.exists(INPUT_PATH):
        raw = pd.read_csv(
            INPUT_PATH,
            compression="gzip",
            parse_dates=[cfg.get("extractor", {}).get("train", {}).get("ds_column", "Date")]
        )
        colmap = {}
        for c in raw.columns:
            lc = c.strip().lower()
            if lc in ("date","ds","timestamp"):
                colmap[c] = "ds"
            if lc in ("value","valor","val"):
                colmap[c] = "y"
        raw = raw.rename(columns=colmap)

        # incluir colunas extras configuradas (ex: EquipmentName) se existirem no CSV
        train_extra_cols = cfg.get("extractor", {}).get("train", {}).get("extra_columns", []) or []
        available_extra = [c for c in train_extra_cols if c in raw.columns]

        # preparar lista de colunas para obs: ds, y + extras presentes
        use_cols = ["ds", "y"] + available_extra
        use_cols = [c for c in use_cols if c in raw.columns]

        if "ds" in raw.columns and "y" in raw.columns:
            obs = raw[use_cols].dropna(subset=["ds", "y"]).sort_values("ds").reset_index(drop=True)

            # Auto-detect equipment_name se não foi passado e houver exatamente 1 equipamento no CSV
            if equipment_name is None and "EquipmentName" in obs.columns:
                unique_eq = obs["EquipmentName"].dropna().unique()
                if len(unique_eq) == 1:
                    equipment_name = unique_eq[0]
                    print(f"[{datetime.now(timezone.utc).isoformat().replace('+00:00','Z')}] EquipmentName detectado no CSV: '{equipment_name}' -> usando modelo específico.")

    # Agora que possivelmente ajustamos equipment_name, localizar e carregar o modelo
    model_path = find_model(equipment_name)
    print(f"[{datetime.now(timezone.utc).isoformat().replace('+00:00','Z')}] Carregando modelo: {model_path}")
    model = joblib.load(model_path)

    meta_path = model_path + ".meta.json"
    last_ds = model_last_ds(meta_path)

    # decidir start_ts (prioridade: start param -> last_ds do modelo -> obs max -> agora)
    if start:
        start_ts = pd.to_datetime(start)
    else:
        start_ts = last_ds if last_ds is not None else (obs["ds"].max() if not obs.empty else pd.Timestamp.now())

    # normalizar antes do cálculo
    horizon_norm = str(horizon).strip().lower()
    resolution_norm = str(resolution).strip().lower()

    periods = compute_periods(horizon_norm, resolution_norm)
    if periods <= 0:
        raise ValueError("HORIZON/RESOLUTION resultaram em periods <= 0.")

    # histórico (in-sample)
    hist_pred = None
    if not obs.empty:
        try:
            hist_pred = model.predict(obs[["ds"]])[["ds","yhat","yhat_lower","yhat_upper"]]
            hist_pred = hist_pred[hist_pred["ds"].isin(obs["ds"])]
        except Exception as e:
            print(f"Aviso: predição in-sample falhou: {e}")

    # futuro
    res_off = to_offset(resolution_norm)
    future_start = pd.to_datetime(start_ts) + res_off
    future_idx = pd.date_range(start=future_start, periods=periods, freq=resolution_norm)
    fut_pred = model.predict(pd.DataFrame({"ds": future_idx}))[["ds","yhat","yhat_lower","yhat_upper"]]

    if hist_pred is not None:
        forecast_df = pd.concat([hist_pred, fut_pred], ignore_index=True).sort_values("ds").reset_index(drop=True)
    else:
        forecast_df = fut_pred.sort_values("ds").reset_index(drop=True)

    # ---------------- NOVO BLOCO: limitar janela de exibição ----------------
    display_start = None
    display_end = None
    try:
        display_end = forecast_df["ds"].max()
        h_td = pd.Timedelta(horizon_norm)
        if h_td <= pd.Timedelta("1h"):
            display_start = display_end - pd.Timedelta("24h") + res_off
        elif h_td <= pd.Timedelta("7d"):
            display_start = display_end - pd.Timedelta("14d")
        elif h_td <= pd.Timedelta("30d"):
            display_start = display_end - pd.Timedelta("90d")
    except Exception:
        display_start = None
        display_end = None

    if display_start is not None:
        if not obs.empty:
            obs = obs[obs["ds"] >= display_start].copy()
        forecast_df = forecast_df[forecast_df["ds"] >= display_start].reset_index(drop=True)
    # ------------------------------------------------------------------------

    # salvar forecast
    os.makedirs(FORECAST_DIR, exist_ok=True)
    fname = f"{slugify(equipment_name or 'general')}_forecast_{horizon}_{resolution}.csv.gz"
    out_path = os.path.join(FORECAST_DIR, fname)
    forecast_df.to_csv(out_path, index=False, compression="gzip")
    print(f"[{datetime.now(timezone.utc).isoformat().replace('+00:00','Z')}] Forecast salvo: {out_path}")

    # plot (apenas janela filtrada)
    plt.figure(figsize=(11,6))
    ax = plt.gca()
    last_hist_date = obs['ds'].max() if not obs.empty else forecast_df['ds'].min()
    is_future = forecast_df['ds'] > last_hist_date
    is_past = ~is_future

    if is_past.any():
        plt.fill_between(forecast_df.loc[is_past,'ds'],
                         forecast_df.loc[is_past,'yhat_lower'],
                         forecast_df.loc[is_past,'yhat_upper'],
                         color='lightgray', alpha=0.4, label='IC (histórico)')
    if is_future.any():
        plt.fill_between(forecast_df.loc[is_future,'ds'],
                         forecast_df.loc[is_future,'yhat_lower'],
                         forecast_df.loc[is_future,'yhat_upper'],
                         color='salmon', alpha=0.35, label='IC (futuro)')
    if not obs.empty:
        plt.scatter(obs['ds'], obs['y'], color='black', s=18, label='Observado')
    plt.plot(forecast_df['ds'], forecast_df['yhat'], color='lightblue', linewidth=1, label='yhat')
    if is_future.any():
        plt.plot(forecast_df.loc[is_future,'ds'], forecast_df.loc[is_future,'yhat'],
                 color='red', linewidth=2, label='Futuro')

    # --- ajuste mínimo para rótulos de hora com sufixo "h" ---
    # Se a janela exibida for curta (até ~48H) e a resolução for horária/minutar,
    # ajustamos o formatter para mostrar "14h" em vez de "14:00".
    try:
        display_span = None
        if display_start is not None and display_end is not None:
            display_span = display_end - display_start
        short_window = (display_span is not None and display_span <= pd.Timedelta("2d"))
        res_lower = resolution_norm
        is_hour_like = ("h" in res_lower and "min" not in res_lower) or ("min" in res_lower) or (res_lower in ("t","min"))
        if short_window and is_hour_like:
            # formatador de horas com 'h'
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Hh"))
            # usar locator automático para ficar legível
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.gcf().autofmt_xdate()
    except Exception:
        pass
    # -------------------------------------------------------------

    plt.xlabel("Data")
    plt.ylabel("Value")
    plt.title(f"Forecast - {equipment_name or 'general'} - horizon {horizon} res {resolution}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_image:
        os.makedirs(IMAGES_DIR, exist_ok=True)
        img_name = image_name or f"forecast_{slugify(equipment_name or 'general')}_{horizon}_{resolution}.png"
        img_path = os.path.join(IMAGES_DIR, img_name)
        plt.savefig(img_path)
        print(f"[{datetime.now(timezone.utc).isoformat().replace('+00:00','Z')}] Gráfico salvo: {img_path}")

    plt.show()
    return forecast_df

# exemplos de uso (mantidos)
if __name__ == "__main__":
    ex = INFER_CFG.get("examples", {})
    try:
        e1 = ex.get("hour", {"horizon":"1H","resolution":"15min"})
        infer(equipment_name=None, start=None, horizon=e1["horizon"], resolution=e1["resolution"], save_image=True)
    except Exception as err:
        print("Erro exemplo 1:", err)

    try:
        e2 = ex.get("week", {"horizon":"7D","resolution":"H"})
        infer(equipment_name=None, start=None, horizon=e2["horizon"], resolution=e2["resolution"], save_image=True)
    except Exception as err:
        print("Erro exemplo 2:", err)

    try:
        e3 = ex.get("month", {"horizon":"30D","resolution":"D"})
        infer(equipment_name=None, start=None, horizon=e3["horizon"], resolution=e3["resolution"], save_image=True)
    except Exception as err:
        print("Erro exemplo 3:", err)