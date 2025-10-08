#!/usr/bin/env python3
"""
model.py
Miolo de lógica de previsão (antes embutido em infer.py).

Responsabilidades:
- Carregar config.json
- Funções utilitárias (slugify, find_model, etc.)
- Carregar dados observados do CSV
- Auto-detectar equipamento (quando equipment_name == None e existir só 1 Entity)
- Carregar modelo (específico ou geral)
- Gerar previsões (histórico + futuro) e aplicar corte de janela de exibição

Alterações mínimas realizadas:
- adicionada configuração extractor.train.identity_columns (padrão: ["EquipmentName"])
- _load_observations agora usa identity_columns para detectar uma identidade única (identity_map)
- adicionadas funções utilitárias: list_identity_columns(), detect_identity(), list_entities()
- mantida compatibilidade: _load_observations continua retornando (obs_df, equipment_eff)
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

# Novidade: colunas que representam a identidade (equipamento / grupo / etc.)
IDENTITY_COLUMNS = CFG.get("extractor", {}).get("train", {}).get("identity_columns", ["EquipmentName"])

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
    Retorna: (periods, horizon_norm, resolution_norm, offset)
    """
    horizon_norm = str(horizon).strip().lower()
    resolution_norm = str(resolution).strip().lower()
    th = pd.Timedelta(horizon_norm)
    off = to_offset(resolution_norm)
    periods = int(math.ceil(th / off))
    return periods, horizon_norm, resolution_norm, off

def detect_identity(obs_df):
    """
    Inspeciona obs_df e tenta detectar uma identidade única com base em IDENTITY_COLUMNS.
    Retorna um dict {col: value} se houver exatamente uma combinação distinta não-nula,
    caso contrário retorna None.
    """
    if obs_df is None or obs_df.empty:
        return None
    cols = [c for c in IDENTITY_COLUMNS if c in obs_df.columns]
    if not cols:
        return None
    # eliminar linhas com NaN nas colunas de identidade e obter combinações únicas
    uniq = obs_df.dropna(subset=cols)[cols].drop_duplicates()
    if len(uniq) == 1:
        row = uniq.iloc[0]
        return {col: row[col] for col in cols}
    return None

def list_identity_columns():
    """
    Retorna as colunas configuradas como identity (ex: ['EquipmentName']).
    """
    return list(IDENTITY_COLUMNS)

def list_entities(column):
    """
    Lista valores distintos da coluna pedida no arquivo de entrada (INPUT_PATH).
    Retorna lista vazia se o arquivo não existir ou a coluna não estiver presente.
    """
    if not os.path.exists(INPUT_PATH):
        return []
    try:
        df = pd.read_csv(INPUT_PATH, compression="gzip", usecols=[column])
    except Exception:
        return []
    vals = df[column].dropna().unique().tolist()
    return vals

def _load_observations(equipment_name):
    """
    Carrega o CSV de dados observados e:
      - renomeia colunas ds/y
      - preserva colunas extras definidas em config.extractor.train.extra_columns
      - auto-detecta equipamento/identidade se não foi passado e houver exatamente uma combinação
    Retorna: (obs_df, equipment_name_effective)
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

        # tentativa genérica de auto-detect usando identity columns
        identity_map = detect_identity(obs)
        if identity_map is not None:
            # se não foi passado equipment_name explicitamente, e existir EquipmentName na identity_map,
            # definimos equipment_eff para manter compatibilidade com código antigo.
            if equipment_eff is None and "EquipmentName" in identity_map:
                equipment_eff = identity_map.get("EquipmentName")
            print(f"[{utcnow_z()}] Identidade detectada: {identity_map} -> tentando modelo específico.")

    return obs, equipment_eff

# ----------------- Novas funções expostas (Passo 1) -----------------
def load_model(equipment_name=None):
    """
    Carrega e retorna (model, model_path, last_ds).
    Lança FileNotFoundError se nenhum modelo encontrado.
    """
    model_path = find_model(equipment_name)
    print(f"[{utcnow_z()}] Carregando modelo: {model_path}")
    model = joblib.load(model_path)
    last_ds = model_last_ds(model_path + ".meta.json")
    return model, model_path, last_ds

def project(model,
            start_ts,
            horizon,
            resolution,
            obs_df=None):
    """
    Gera a projeção (histórico + futuro) a partir de um objeto Prophet já carregado.
    - model: instância Prophet (treinada)
    - start_ts: timestamp (ou None) — usado apenas para definir início do futuro
    - horizon/resolution: como em generate_forecast
    - obs_df: DataFrame opcional com coluna 'ds' para predição in-sample (pode ser None)

    Retorna:
      forecast_df, obs_df_filtered, meta
    """
    periods, horizon_norm, resolution_norm, res_off = compute_periods(horizon, resolution)
    if periods <= 0:
        raise ValueError("HORIZON/RESOLUTION resultaram em periods <= 0.")

    # histórico (in-sample)
    hist_pred = None
    if obs_df is not None and not obs_df.empty:
        try:
            hist_pred = model.predict(obs_df[["ds"]])[["ds", "yhat", "yhat_lower", "yhat_upper"]]
            hist_pred = hist_pred[hist_pred["ds"].isin(obs_df["ds"])]
        except Exception as e:
            print(f"Aviso: predição in-sample falhou: {e}")

    # futuro
    future_start = pd.to_datetime(start_ts) + res_off
    future_idx = pd.date_range(start=future_start, periods=periods, freq=resolution_norm)
    fut_pred = model.predict(pd.DataFrame({"ds": future_idx}))[["ds", "yhat", "yhat_lower", "yhat_upper"]]

    if hist_pred is not None:
        forecast_df = pd.concat([hist_pred, fut_pred], ignore_index=True).sort_values("ds").reset_index(drop=True)
    else:
        forecast_df = fut_pred.sort_values("ds").reset_index(drop=True)

    # aplicar regra de janela de exibição (mesma usada antes)
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
        if obs_df is not None and not obs_df.empty:
            obs_df = obs_df[obs_df["ds"] >= display_start].copy()
        forecast_df = forecast_df[forecast_df["ds"] >= display_start].reset_index(drop=True)

    meta = {
        "periods": periods,
        "horizon_norm": horizon_norm,
        "resolution_norm": resolution_norm,
        "display_start": display_start,
        "display_end": display_end,
    }
    return forecast_df, obs_df, meta

# ----------------- Função compatível existente -----------------
def generate_forecast(equipment_name=None, start=None, horizon="1H", resolution="15min"):
    """
    Função compatível que encapsula o fluxo completo:
      - carrega observados (e auto-detecta equipamento)
      - carrega modelo
      - decide start_ts (prioridade: start param -> last_ds do modelo -> max obs -> now)
      - chama project() e retorna os mesmos valores que o infer antigo esperava

    Retorno:
      forecast_df, obs_filtered, equipment_name_effective, meta
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
        start_ts = last_ds if last_ds is not None else (obs["ds"].max() if not obs.empty else pd.Timestamp.now())

    # 4) delegar para project()
    forecast_df, obs_filtered, meta = project(model, start_ts, horizon, resolution, obs_df=obs)

    # enriquecer meta com info do modelo
    meta.update({
        "model_path": model_path,
        "equipment_name_effective": equipment_eff,
        "last_ds_model": last_ds.isoformat() if last_ds is not None else None,
        "start_ts_used": pd.to_datetime(start_ts).isoformat()
    })

    return forecast_df, obs_filtered, equipment_eff, meta