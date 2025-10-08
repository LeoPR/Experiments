#!/usr/bin/env python3
"""
train.py - Treina modelos Prophet a partir do CSV extraído pelo extrator.

Refactor mínimo:
- Helpers: ts_now(), normalize_freq(), resample_series(), load_input_csv()
- Mantive comportamento, validações e metadados como antes.
- Objetivo: reduzir duplicação (reamostragem) e centralizar parsing do CSV.
"""

import os
import json
from datetime import datetime, timezone
import joblib
import pandas as pd
from prophet import Prophet

CONFIG_PATH = "config.json"

# carregar config
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = json.load(f)

ext = cfg.get("extractor", {})
out = ext.get("output", {})
inp_dir = out.get("output_dir")
basename = out.get("output_basename")
if not inp_dir or not basename:
    raise ValueError("Config: extractor.output.output_dir e output_basename devem estar definidos no config.json")

INPUT_PATH = os.path.join(inp_dir, basename + ".csv.gz")
META_PATH = os.path.join(inp_dir, basename + ".csv.gz.meta.json")

train_cfg = ext.get("train", {})
Y_COLUMN = train_cfg.get("y_column")
DS_COLUMN = train_cfg.get("ds_column")
if not Y_COLUMN or not DS_COLUMN:
    raise ValueError("Config: extractor.train.y_column e extractor.train.ds_column são obrigatórios no config.json")

EXTRA_COLUMNS = train_cfg.get("extra_columns", []) or []
TRAIN_PER_EQUIPMENT = bool(train_cfg.get("train_per_equipment", True))
EQUIPMENT_FILTER = train_cfg.get("equipment_filter")

TRAIN_RESAMPLE_ENABLED = bool(train_cfg.get("train_resample_enabled", False))
TRAIN_RESAMPLE_FREQ = train_cfg.get("train_resample_freq", None)
TRAIN_MAX_HISTORY_DAYS = train_cfg.get("train_max_history_days", None)

MODELS_DIR = cfg.get("models_defaults", {}).get("model_dir")
if not MODELS_DIR:
    raise ValueError("Config: models_defaults.model_dir deve estar definido no config.json")

PROPHET_PARAMS = cfg.get("prophet", {}).get("params", {})
EXTRA_SEASONALITIES = cfg.get("prophet", {}).get("extra_seasonalities", [])

# ------------------ HELPERS ------------------
def ts_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def normalize_freq(freq):
    return str(freq).strip().lower() if freq is not None else None

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

def resample_series(df, freq, value_col="y"):
    """
    Reamostra DataFrame com índice temporal em 'ds' e coluna de valor.
    Retorna DataFrame com colunas ['ds','y'] prontas para treino.
    """
    if df is None or df.empty:
        return df
    freq_norm = normalize_freq(freq)
    series = df.set_index("ds").resample(freq_norm)[value_col].mean()
    series = series.interpolate(method="time").ffill().bfill().reset_index()
    series = series.rename(columns={value_col: "y"})
    # garantir a ordem das colunas
    series = series[["ds", "y"]]
    return series

def load_input_csv(path, usecols, ds_col, y_col):
    """
    Carrega CSV.gz com parse de datas, renomeia colunas para 'ds' e 'y',
    converte y para numérico e remove NaNs.
    """
    df = pd.read_csv(path, compression='gzip', usecols=usecols, parse_dates=[ds_col])
    df = df.rename(columns={ds_col: "ds", y_col: "y"})
    df = df.dropna(subset=["ds", "y"])
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["y"])
    df = df.sort_values("ds").reset_index(drop=True)
    return df

# ------------------ TRAIN UTILITIES ------------------
def apply_extra_seasonalities(model, extra_list):
    """
    Aplica sazonalidades extras definidas no config.
    Cada item esperado:
      {"name": "...", "period": <float>, "fourier_order": <int>, "prior_scale": <float opcional>, "mode": <...>}
    """
    applied = []
    for seas in extra_list:
        try:
            kwargs = {
                "name": seas["name"],
                "period": seas["period"],
                "fourier_order": seas["fourier_order"],
                "prior_scale": seas.get("prior_scale", 10.0)
            }
            if "mode" in seas and seas["mode"]:
                kwargs["mode"] = seas["mode"]
            model.add_seasonality(**kwargs)
            applied.append(seas)
        except Exception as e:
            print(f"Aviso: falha ao adicionar sazonalidade {seas.get('name')}: {e}")
    return applied

def train_and_save(df, label, meta_extra=None):
    os.makedirs(MODELS_DIR, exist_ok=True)
    slug = slugify(label)
    pkl = os.path.join(MODELS_DIR, f"{slug}_prophet.pkl")
    print(f"[{ts_now()}] Treinando '{label}' ({len(df)} pontos)...")

    m = Prophet(**PROPHET_PARAMS)
    applied = apply_extra_seasonalities(m, EXTRA_SEASONALITIES)

    m.fit(df)
    joblib.dump(m, pkl)
    meta = {
        "model_label": label,
        "saved_at": ts_now(),
        "rows_used": int(len(df)),
        "last_ds": df['ds'].max().isoformat() if not df['ds'].isnull().all() else None,
        "prophet_params": PROPHET_PARAMS,
        "extra_seasonalities_applied": applied
    }
    if meta_extra:
        meta.update(meta_extra)
    with open(pkl + ".meta.json", "w", encoding="utf-8") as fm:
        json.dump(meta, fm, indent=2, ensure_ascii=False)
    print(f"[{ts_now()}] Salvo: {pkl}")
    return pkl

# ------------------ MAIN FLOW ------------------
# Log das configurações efetivas
print(f"[{ts_now()}] CONFIG USADA:")
print(f"  INPUT_PATH = {INPUT_PATH}")
print(f"  MODELS_DIR = {MODELS_DIR}")
print(f"  DS_COLUMN = '{DS_COLUMN}', Y_COLUMN = '{Y_COLUMN}'")
print(f"  EXTRA_COLUMNS = {EXTRA_COLUMNS}")
print(f"  TRAIN_PER_EQUIPMENT = {TRAIN_PER_EQUIPMENT}")
print(f"  TRAIN_RESAMPLE_ENABLED = {TRAIN_RESAMPLE_ENABLED}, TRAIN_RESAMPLE_FREQ = {TRAIN_RESAMPLE_FREQ}")
if TRAIN_MAX_HISTORY_DAYS:
    print(f"  TRAIN_MAX_HISTORY_DAYS = {TRAIN_MAX_HISTORY_DAYS}")

if not os.path.exists(INPUT_PATH):
    raise FileNotFoundError(f"Arquivo de entrada não encontrado: {INPUT_PATH}")

meta = {}
if os.path.exists(META_PATH):
    with open(META_PATH, "r", encoding="utf-8") as fm:
        meta = json.load(fm)
else:
    print(f"Aviso: metadados não encontrados em {META_PATH}. Seguirei sem validação antecipada.")

available_cols = []
if meta and "columns" in meta:
    available_cols = [c.get("name") for c in meta.get("columns", []) if c.get("name")]

# validações mínimas usando estritamente o config.json
if available_cols:
    if Y_COLUMN not in available_cols:
        raise ValueError(f"Coluna y configurada ('{Y_COLUMN}') não encontrada nos metadados: {available_cols}")
    if DS_COLUMN not in available_cols:
        raise ValueError(f"Coluna ds configurada ('{DS_COLUMN}') não encontrada nos metadados: {available_cols}")

# decidir colunas a carregar (usa os valores do config)
usecols = [DS_COLUMN, Y_COLUMN]
extras_to_use = []
for col in EXTRA_COLUMNS:
    if not available_cols or col in available_cols:
        extras_to_use.append(col)
usecols += extras_to_use
usecols = list(dict.fromkeys(usecols))

print(f"[{ts_now()}] Carregando {INPUT_PATH} (colunas={usecols}) ...")
df = load_input_csv(INPUT_PATH, usecols, DS_COLUMN, Y_COLUMN)

if EQUIPMENT_FILTER:
    print(f"Atenção: filtro de equipamento ativo: {EQUIPMENT_FILTER}")
    if "EquipmentName" not in df.columns:
        raise ValueError("Filtro por equipamento solicitado, mas coluna 'EquipmentName' não está presente nos dados.")
    df = df[df["EquipmentName"] == EQUIPMENT_FILTER]
    if len(df) < 2:
        raise ValueError(f"Após aplicar filtro '{EQUIPMENT_FILTER}' restaram {len(df)} pontos — insuficiente para treinar.")

if len(df) < 2:
    raise ValueError("Dados insuficientes para treinar (menos de 2 pontos).")

# limitar histórico opcional
if TRAIN_MAX_HISTORY_DAYS:
    try:
        days = int(TRAIN_MAX_HISTORY_DAYS)
        cutoff = df["ds"].max() - pd.Timedelta(days=days)
        before = len(df)
        df = df[df["ds"] >= cutoff].copy()
        print(f"[{ts_now()}] Limitei histórico para últimos {days} dias: {before} -> {len(df)} pontos")
    except Exception:
        print("Aviso: falha ao aplicar train_max_history_days; ignorando.")

# série geral sem groupby
general_df = df[["ds", "y"]].sort_values("ds").reset_index(drop=True)

# resample opcional (usando helper resample_series)
meta_extra = {}
if TRAIN_RESAMPLE_ENABLED and TRAIN_RESAMPLE_FREQ:
    freq = normalize_freq(TRAIN_RESAMPLE_FREQ)
    print(f"[{ts_now()}] Resample ativado: freq={freq}. Preparando série (geral).")
    before = len(general_df)
    general_df = resample_series(general_df, freq, value_col="y")
    after = len(general_df)
    print(f"[{ts_now()}] Geral: {before} -> {after} pontos após resample")
    meta_extra["resampled"] = True
    meta_extra["resample_freq"] = freq
else:
    meta_extra["resampled"] = False

general_path = train_and_save(general_df, "general", meta_extra=meta_extra)

specific_models = {}
if TRAIN_PER_EQUIPMENT and "EquipmentName" in df.columns:
    equipments = df["EquipmentName"].dropna().unique().tolist()
    print(f"Treinando modelos por equipamento (encontrados {len(equipments)})...")
    for eq in equipments:
        sub = df[df["EquipmentName"] == eq][["ds", "y"]].sort_values("ds").reset_index(drop=True)
        if len(sub) < 2:
            print(f" - Pulando '{eq}' (insuficiente: {len(sub)})")
            continue
        if TRAIN_RESAMPLE_ENABLED and TRAIN_RESAMPLE_FREQ:
            freq = normalize_freq(TRAIN_RESAMPLE_FREQ)
            b = len(sub)
            sub = resample_series(sub, freq, value_col="y")
            print(f" - {eq}: {b} -> {len(sub)} após resample {freq}")
        try:
            p = train_and_save(
                sub,
                eq,
                meta_extra={
                    "resampled": TRAIN_RESAMPLE_ENABLED,
                    "resample_freq": (TRAIN_RESAMPLE_FREQ if TRAIN_RESAMPLE_ENABLED else None)
                }
            )
            specific_models[eq] = p
        except Exception as e:
            print(f" - Falha ao treinar '{eq}': {e}")

print("Treino finalizado.")
print("Modelo geral:", general_path)
print("Modelos específicos treinados:", list(specific_models.keys()))