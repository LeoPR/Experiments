#!/usr/bin/env python3
"""
model.py
Miolo de lógica de previsão (antes embutido em infer.py):

Responsabilidades:
- Carregar config.json
- Funções utilitárias (slugify, find_model, etc.)
- Carregar dados observados do CSV
- Auto-detectar equipamento (quando equipment_name == None e existir só 1 EquipmentName)
- Carregar modelo (específico ou geral)
- Gerar previsões (histórico + futuro) e aplicar corte de janela de exibição
- Retornar objetos para que infer.py cuide de salvar CSV e plotar

Função principal:
  generate_forecast(equipment_name=None, start=None, horizon="1H", resolution="15min")

Retorno:
  forecast_df (DataFrame filtrado para janela de exibição)
  obs_filtered (DataFrame de observados filtrados para a janela)
  equipment_name_effective (str ou None)
  meta (dict) -> inclui model_path, periods, horizon_norm, resolution_norm, display_start, display_end, last_ds_model

Observação:
- Não salva arquivos nem gera gráficos aqui; isso fica em infer.py
"""

import os
import json
import math
import joblib
from datetime import datetime, timezone
import pandas as pd
from pandas.tseries.frequencies import to_offset

# Carregar config uma única vez
with open("config.json", "r", encoding="utf-8") as f:
    CFG = json.load(f)

# Constantes de config (expostas para o infer.py)
MODELS_DIR = CFG.get("models_defaults", {}).get("model_dir", "./models")
IMAGES_DIR = CFG.get("models_defaults", {}).get("images_dir", "./images")
EXT_OUT = CFG.get("extractor", {}).get("output", {})
INPUT_DIR = EXT_OUT.get("output_dir", "./dados")
BASENAME = EXT_OUT.get("output_basename", "product_extraction")
INPUT_PATH = os.path.join(INPUT_DIR, BASENAME + ".csv.gz")

INFER_CFG = CFG.get("infer", {})
FORECAST_DIR = INFER_CFG.get("forecast_output_dir", "./dados/forecasts")
GENERAL_MODEL_NAME = INFER_CFG.get("default_models", {}).get("general_name", "general_prophet.pkl")
SPECIFIC_TEMPLATE = INFER_CFG.get("default_models", {}).get("specific_template", "{slug}_prophet.pkl")

def utcnow_z():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

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
    """
    Retorna o caminho do modelo específico se existir; senão o geral.
    Lança FileNotFoundError se nenhum existir.
    """
    specific = os.path.join(MODELS_DIR, SPECIFIC_TEMPLATE.format(slug=slugify(equipment_name))) if equipment_name else None
    general = os.path.join(MODELS_DIR, GENERAL_MODEL_NAME)
    if equipment_name and specific and os.path.exists(specific):
        return specific
    if os.path.exists(general):
        return general
    raise FileNotFoundError(f"Nenhum modelo encontrado: tentei '{specific}' e '{general}'")

def model_last_ds(meta_path):
    """
    Lê meta JSON do modelo (se existir) para tentar pegar 'last_ds'.
    """
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
    """
    Calcula quantidade de períodos futuros baseado no horizonte e resolução normalizados (lowercase).
    """
    horizon_norm = str(horizon).strip().lower()
    resolution_norm = str(resolution).strip().lower()
    th = pd.Timedelta(horizon_norm)
    off = to_offset(resolution_norm)
    periods = int(math.ceil(th / off))
    return periods, horizon_norm, resolution_norm, off

def _load_observations(equipment_name):
    """
    Carrega o CSV de dados observados e:
      - renomeia colunas ds/y
      - preserva colunas extras definidas em config.extractor.train.extra_columns
      - auto-detecta equipment_name se não foi passado e houver exatamente um (coluna 'EquipmentName')
    Retorna: obs_df, equipment_name_eff
    """
    obs = pd.DataFrame(columns=["ds", "y"])
    equipment_eff = equipment_name
    if not os.path.exists(INPUT_PATH):
        return obs, equipment_eff

    raw = pd.read_csv(
        INPUT_PATH,
        compression="gzip",
        parse_dates=[CFG.get("extractor", {}).get("train", {}).get("ds_column", "Date")]
    )

    colmap = {}
    for c in raw.columns:
        lc = c.strip().lower()
        if lc in ("date", "ds", "timestamp"):
            colmap[c] = "ds"
        if lc in ("value", "valor", "val"):
            colmap[c] = "y"
    raw = raw.rename(columns=colmap)

    train_extra_cols = CFG.get("extractor", {}).get("train", {}).get("extra_columns", []) or []
    available_extra = [c for c in train_extra_cols if c in raw.columns]

    use_cols = ["ds", "y"] + available_extra
    use_cols = [c for c in use_cols if c in raw.columns]

    if "ds" in raw.columns and "y" in raw.columns:
        obs = raw[use_cols].dropna(subset=["ds", "y"]).sort_values("ds").reset_index(drop=True)

        # auto-detect
        if equipment_eff is None and "EquipmentName" in obs.columns:
            uniq = obs["EquipmentName"].dropna().unique()
            if len(uniq) == 1:
                equipment_eff = uniq[0]
                print(f"[{utcnow_z()}] EquipmentName detectado: '{equipment_eff}' -> tentando modelo específico.")
    return obs, equipment_eff

def generate_forecast(equipment_name=None, start=None, horizon="1H", resolution="15min"):
    """
    Gera forecast (histórico + futuro), aplicando as mesmas regras de janela de exibição do infer original.

    Parâmetros:
      equipment_name: (str|None) nome do equipamento; se None tenta auto-detectar.
      start: data inicial para gerar futuro (string ou datetime). Se None, usa last_ds do modelo ou max ds observado ou agora.
      horizon: string tipo "1H", "7D"...
      resolution: frequência de previsão ("15min", "H", "D", etc.)

    Retorno:
      forecast_df_filtrado, obs_filtrado, equipment_name_efetivo, meta_dict
    """
    # 1) Carregar observados e possivelmente detectar equipamento
    obs, equipment_eff = _load_observations(equipment_name)

    # 2) Carregar modelo
    model_path = find_model(equipment_eff)
    print(f"[{utcnow_z()}] Carregando modelo: {model_path}")
    model = joblib.load(model_path)

    # 3) last_ds do meta/modelo
    meta_path = model_path + ".meta.json"
    last_ds = model_last_ds(meta_path)
    if start:
        start_ts = pd.to_datetime(start)
    else:
        # prioridade: last_ds do modelo > max ds observado > agora
        start_ts = last_ds if last_ds is not None else (obs["ds"].max() if not obs.empty else pd.Timestamp.now())

    # 4) Calcular períodos
    periods, horizon_norm, resolution_norm, res_off = compute_periods(horizon, resolution)
    if periods <= 0:
        raise ValueError("HORIZON/RESOLUTION resultaram em periods <= 0.")

    # 5) Predição in-sample (histórico) — tenta replicar lógica anterior
    hist_pred = None
    if not obs.empty:
        try:
            hist_pred = model.predict(obs[["ds"]])[["ds", "yhat", "yhat_lower", "yhat_upper"]]
            hist_pred = hist_pred[hist_pred["ds"].isin(obs["ds"])]
        except Exception as e:
            print(f"Aviso: predição in-sample falhou: {e}")

    # 6) Futuro
    future_start = pd.to_datetime(start_ts) + res_off
    future_idx = pd.date_range(start=future_start, periods=periods, freq=resolution_norm)
    fut_pred = model.predict(pd.DataFrame({"ds": future_idx}))[["ds", "yhat", "yhat_lower", "yhat_upper"]]

    if hist_pred is not None:
        forecast_df = pd.concat([hist_pred, fut_pred], ignore_index=True).sort_values("ds").reset_index(drop=True)
    else:
        forecast_df = fut_pred.sort_values("ds").reset_index(drop=True)

    # 7) Corte de janela de exibição (mesma regra do infer original)
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

    meta = {
        "model_path": model_path,
        "equipment_name_effective": equipment_eff,
        "periods": periods,
        "horizon_norm": horizon_norm,
        "resolution_norm": resolution_norm,
        "display_start": display_start,
        "display_end": display_end,
        "last_ds_model": last_ds.isoformat() if last_ds is not None else None,
        "start_ts_used": start_ts.isoformat()
    }

    return forecast_df, obs, equipment_eff, meta